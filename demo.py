"""
demo.py — 快速冒烟测试（不需要真实数据集）
验证模型结构、前向传播、损失计算、采样全流程正确
用法: python demo.py
"""

import torch
import numpy as np
from model import MultiDiffNILM
from metrics import evaluate_all_appliances, print_metrics

# ─────────────────────────────────────────
print("=" * 55)
print("  MultiDiffNILM — 快速冒烟测试")
print("=" * 55)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[设备] {device}")

# ─────────────────────────────────────────
# 模拟 REDD 配置
APPLIANCES = ["microwave", "washer", "dishwasher", "refrigerator"]
N_APP      = len(APPLIANCES)
B, L       = 4, 480   # batch=4, 窗口长度=480

# ─────────────────────────────────────────
# 构造模型
model = MultiDiffNILM(n_appliances=N_APP).to(device)
n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"[模型] 参数量: {n_params:,}")

# ─────────────────────────────────────────
# 构造假数据
mains      = torch.randn(B, 1, L).to(device)
target     = torch.rand(B, N_APP, L).to(device)   # [0,1] 归一化功率
time_codes = torch.rand(B, 3,     L).sub(0.5).to(device)  # [-0.5,0.5]

# ─────────────────────────────────────────
# 测试前向传播 & 损失
print("\n[测试] 前向传播 + 损失计算...")
losses = model.compute_loss(mains, target, time_codes,
                             lambda_physics=0.1)
print(f"  loss_total   = {losses['loss'].item():.5f}")
print(f"  loss_denoise = {losses['loss_denoise'].item():.5f}")
print(f"  loss_physics = {losses['loss_physics'].item():.5f}")
assert not torch.isnan(losses["loss"]), "损失出现 NaN！"
print("  ✓ 损失正常")

# ─────────────────────────────────────────
# 测试反向传播
print("\n[测试] 反向传播...")
losses["loss"].backward()
print("  ✓ 反向传播正常")

# ─────────────────────────────────────────
# 测试采样（推理）
print("\n[测试] 扩散采样（推理）...")
model.eval()
with torch.no_grad():
    pred = model.disaggregate(mains, time_codes)
print(f"  输出形状: {tuple(pred.shape)}")
assert pred.shape == (B, N_APP, L), "输出形状错误！"
print(f"  输出范围: [{pred.min():.4f}, {pred.max():.4f}]")
print("  ✓ 采样正常")

# ─────────────────────────────────────────
# 测试评估指标
print("\n[测试] 评估指标计算...")
pred_np = pred.cpu().numpy()
true_np = target.cpu().numpy()

on_thr = {app: 0.1 for app in APPLIANCES}
max_pw = {"microwave": 1800, "washer": 3500,
          "dishwasher": 1200, "refrigerator": 400}

results = evaluate_all_appliances(
    y_true=true_np.reshape(1, N_APP, -1),
    y_pred=pred_np.reshape(1, N_APP, -1),
    appliances=APPLIANCES,
    on_thresholds=on_thr,
    max_powers=max_pw,
)
print_metrics(results)

# ─────────────────────────────────────────
# 输入输出形状汇总
print("\n[形状汇总]")
print(f"  mains      输入: {tuple(mains.shape)}")
print(f"  target     输入: {tuple(target.shape)}")
print(f"  time_codes 输入: {tuple(time_codes.shape)}")
print(f"  pred       输出: {tuple(pred.shape)}")
print(f"    → 同时输出 {N_APP} 个电器的功率波形 ✓")

print("\n" + "=" * 55)
print("  所有测试通过！模型可以正常训练。")
print("  运行训练: python train.py --dataset redd")
print("=" * 55)
