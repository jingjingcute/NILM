"""
evaluate.py — MultiDiffNILM 测试集完整评估
用法:
    python evaluate.py --dataset redd   --ckpt checkpoints/redd/best.pt
    python evaluate.py --dataset ukdale --ckpt checkpoints/ukdale/best.pt
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from config import DATASET_CONFIG, TRAIN
from data_loader import get_dataloaders
from model import MultiDiffNILM
from metrics import evaluate_all_appliances, print_metrics


@torch.no_grad()
def run_inference(model, loader, device, n_app):
    """在整个数据集上跑推理，收集预测和真值"""
    model.eval()
    all_pred = []
    all_true = []

    for mains, target, time_codes in tqdm(loader, desc="推理"):
        mains      = mains.to(device)       # (B,1,L)
        target     = target.to(device)      # (B,N,L)
        time_codes = time_codes.to(device)  # (B,3,L)

        pred = model.disaggregate(mains, time_codes)  # (B,N,L)

        all_pred.append(pred.cpu().numpy())
        all_true.append(target.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)   # (N_total, N_app, L)
    all_true = np.concatenate(all_true, axis=0)
    return all_pred, all_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="redd",
                        choices=["redd", "ukdale"])
    parser.add_argument("--ckpt",    required=True,
                        help="checkpoint 文件路径")
    parser.add_argument("--batch_size", type=int,
                        default=TRAIN["batch_size"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    cfg        = DATASET_CONFIG[args.dataset]
    appliances = cfg["appliances"]
    n_app      = len(appliances)

    # ── 加载数据
    print(f"[数据] 加载 {args.dataset.upper()} 测试集...")
    _, _, test_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=TRAIN["num_workers"])

    # ── 加载模型
    model = MultiDiffNILM(n_appliances=n_app).to(device)
    ck    = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["state_dict"])
    print(f"[模型] 加载自 {args.ckpt}（第 {ck['epoch']+1} epoch）")

    # ── 推理
    print("[推理] 开始...")
    pred, true = run_inference(model, test_loader, device, n_app)

    # ── 归一化阈值（max_power 归一化后的 on_threshold）
    on_thr_norm = {
        app: cfg["on_power_threshold"][app] / cfg["max_power"][app]
        for app in appliances
    }

    # ── 评估
    print("[评估] 计算指标...")
    results = evaluate_all_appliances(
        y_true      = true,
        y_pred      = pred,
        appliances  = appliances,
        on_thresholds = on_thr_norm,
        max_powers    = cfg["max_power"],
    )

    # ── 打印结果
    print(f"\n{'='*60}")
    print(f"  数据集: {args.dataset.upper()}")
    print(f"  模型:   MultiDiffNILM")
    print_metrics(results)

    # ── 保存 CSV
    import pandas as pd
    df = pd.DataFrame(results).T
    out_path = Path(args.ckpt).parent / f"eval_{args.dataset}.csv"
    df.to_csv(out_path)
    print(f"\n[保存] 评估结果 → {out_path}")


if __name__ == "__main__":
    main()
