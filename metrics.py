"""
metrics.py — NILM 评估指标
分类指标：Accuracy, Precision, Recall, F1
回归指标：MAE, MRE
"""

import numpy as np
import torch
from typing import Dict


def compute_status(power: np.ndarray,
                   threshold: float) -> np.ndarray:
    """功率值 → 开关状态（0/1）"""
    return (power > threshold).astype(np.float32)


def classification_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            threshold: float = 0.5
                            ) -> Dict[str, float]:
    """
    y_true, y_pred: (T,) 功率归一化值 [0,1]
    threshold: 判定"开启"的功率阈值（归一化后）
    """
    s_true = compute_status(y_true, threshold)
    s_pred = compute_status(y_pred, threshold)

    TP = ((s_true == 1) & (s_pred == 1)).sum()
    FP = ((s_true == 0) & (s_pred == 1)).sum()
    FN = ((s_true == 1) & (s_pred == 0)).sum()
    TN = ((s_true == 0) & (s_pred == 0)).sum()

    P = s_true.sum()
    N = len(s_true) - P

    accuracy  = (TP + TN) / (P + N + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy":  float(accuracy),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


def regression_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        max_power: float = 1.0
                        ) -> Dict[str, float]:
    """
    y_true, y_pred: 归一化功率 [0,1]
    max_power: 还原到瓦特的缩放系数
    """
    # 还原到真实功率（W）
    t = y_true * max_power
    p = y_pred * max_power

    mae = np.abs(t - p).mean()
    denom = np.maximum(np.abs(t), np.abs(p))
    mre   = (np.abs(t - p) / (denom + 1e-8)).mean()

    return {"mae": float(mae), "mre": float(mre)}


def evaluate_all_appliances(
        y_true: np.ndarray,     # (N_samples, N_app, L)
        y_pred: np.ndarray,     # (N_samples, N_app, L)
        appliances: list,
        on_thresholds: dict,    # 归一化后的开启阈值
        max_powers: dict        # 最大功率（W）
        ) -> Dict[str, Dict[str, float]]:
    """
    对每个电器分别计算所有指标，并返回平均值
    """
    results = {}
    n_app = len(appliances)

    # 展平时间维
    y_true_flat = y_true.reshape(-1, n_app, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, n_app, y_pred.shape[-1])

    agg_acc, agg_f1, agg_mae, agg_mre = 0., 0., 0., 0.

    for i, app in enumerate(appliances):
        t = y_true_flat[:, i, :].flatten()
        p = y_pred_flat[:, i, :].flatten()

        thr  = on_thresholds.get(app, 0.1)
        maxp = max_powers.get(app, 1.0)

        clf = classification_metrics(t, p, threshold=thr)
        reg = regression_metrics(t, p, max_power=maxp)

        results[app] = {**clf, **reg}
        agg_acc += clf["accuracy"]
        agg_f1  += clf["f1"]
        agg_mae += reg["mae"]
        agg_mre += reg["mre"]

    # 平均指标
    results["average"] = {
        "accuracy": agg_acc / n_app,
        "f1":       agg_f1  / n_app,
        "mae":      agg_mae / n_app,
        "mre":      agg_mre / n_app,
    }
    return results


def print_metrics(results: Dict[str, Dict[str, float]]):
    """格式化打印评估结果"""
    header = f"{'Appliance':<16} {'Accuracy':>10} {'F1':>10}"
    header += f" {'MAE(W)':>10} {'MRE':>10}"
    print("\n" + "=" * 60)
    print(header)
    print("-" * 60)
    for app, m in results.items():
        print(f"{app:<16} {m['accuracy']:>10.4f} {m['f1']:>10.4f}"
              f" {m['mae']:>10.2f} {m['mre']:>10.4f}")
    print("=" * 60)
