"""
train.py — MultiDiffNILM 训练 & 验证主程序
用法:
    python train.py --dataset redd
    python train.py --dataset ukdale
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from config import DATASET_CONFIG, TRAIN
from data_loader import get_dataloaders
from model import MultiDiffNILM
from metrics import evaluate_all_appliances, print_metrics


# ───────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return ck["epoch"], ck["val_loss"]


# ───────────────────────────────────────────
def train_one_epoch(model, loader, optimizer,
                    lambda_physics, device, writer, global_step):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for mains, target, time_codes in pbar:
        mains      = mains.to(device)       # (B,1,L)
        target     = target.to(device)      # (B,N,L)
        time_codes = time_codes.to(device)  # (B,3,L)

        optimizer.zero_grad()
        losses = model.compute_loss(mains, target, time_codes,
                                    lambda_physics)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["loss"].item()
        n_batches  += 1
        global_step += 1

        # TensorBoard 日志
        writer.add_scalar("train/loss",         losses["loss"].item(),         global_step)
        writer.add_scalar("train/loss_denoise", losses["loss_denoise"].item(), global_step)
        writer.add_scalar("train/loss_physics", losses["loss_physics"].item(), global_step)

        pbar.set_postfix({
            "loss":     f"{losses['loss'].item():.4f}",
            "denoise":  f"{losses['loss_denoise'].item():.4f}",
            "physics":  f"{losses['loss_physics'].item():.4f}",
        })

    return total_loss / n_batches, global_step


@torch.no_grad()
def validate(model, loader, lambda_physics, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for mains, target, time_codes in loader:
        mains      = mains.to(device)
        target     = target.to(device)
        time_codes = time_codes.to(device)

        losses = model.compute_loss(mains, target, time_codes,
                                    lambda_physics)
        total_loss += losses["loss"].item()
        n_batches  += 1

    return total_loss / n_batches


# ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="redd",
                        choices=["redd", "ukdale"])
    parser.add_argument("--resume",     default=None,
                        help="继续训练的 checkpoint 路径")
    parser.add_argument("--epochs",     type=int,
                        default=TRAIN["max_epochs"])
    parser.add_argument("--batch_size", type=int,
                        default=TRAIN["batch_size"])
    parser.add_argument("--lr",         type=float,
                        default=TRAIN["learning_rate"])
    parser.add_argument("--lambda_p",   type=float,
                        default=TRAIN["lambda_physics"],
                        help="物理守恒损失权重")
    args = parser.parse_args()

    # ── 基础设置
    set_seed(TRAIN["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    cfg        = DATASET_CONFIG[args.dataset]
    appliances = cfg["appliances"]
    n_app      = len(appliances)

    # ── 保存目录
    save_dir = Path(TRAIN["save_dir"]) / args.dataset
    log_dir  = Path(TRAIN["log_dir"])  / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)
    best_ckpt = save_dir / "best.pt"
    last_ckpt = save_dir / "last.pt"

    # ── 数据加载
    print(f"[数据] 加载 {args.dataset.upper()} 数据集...")
    train_loader, val_loader, _ = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=TRAIN["num_workers"])

    # ── 模型 & 优化器
    model = MultiDiffNILM(n_appliances=n_app).to(device)
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"[模型] 参数量: {n_params:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(TRAIN["adam_beta1"], TRAIN["adam_beta2"]))

    writer      = SummaryWriter(log_dir=str(log_dir))
    global_step = 0
    start_epoch = 0
    best_val    = float("inf")

    # ── 断点续训
    if args.resume and os.path.exists(args.resume):
        print(f"[续训] 加载 {args.resume}")
        start_epoch, best_val = load_checkpoint(
            model, optimizer, args.resume, device)
        start_epoch += 1

    # ── 训练循环
    print(f"\n[训练] 开始，共 {args.epochs} 个 epoch\n")
    for epoch in range(start_epoch, args.epochs):
        # ─ 训练
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer,
            args.lambda_p, device, writer, global_step)

        # ─ 验证
        val_loss = validate(model, val_loader,
                            args.lambda_p, device)

        writer.add_scalar("val/loss", val_loss, epoch)
        print(f"Epoch [{epoch+1:>4}/{args.epochs}]  "
              f"train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}")

        # ─ 保存最优
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch,
                            best_val, best_ckpt)
            print(f"  ✓ 保存最优模型 val_loss={best_val:.5f}")

        # ─ 保存最新
        save_checkpoint(model, optimizer, epoch,
                        val_loss, last_ckpt)

    writer.close()
    print("\n[训练完成]")
    print(f"  最优验证损失: {best_val:.5f}")
    print(f"  模型保存于:   {save_dir}")


if __name__ == "__main__":
    main()
