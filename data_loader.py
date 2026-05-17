"""
data_loader.py — REDD / UKDALE 数据加载与预处理
MultiDiffNILM 项目
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from config import DATASET_CONFIG, PREPROCESS, TRAIN


# ═══════════════════════════════════════════════════════════
#  1. 原始数据读取
# ═══════════════════════════════════════════════════════════

class REDDReader:
    """读取 REDD 低频数据（low_freq 目录）"""

    def __init__(self, data_path: str):
        self.path = Path(data_path)

    def _read_channel(self, house: int, channel: int) -> pd.Series:
        fp = self.path / f"house_{house}" / f"channel_{channel}.dat"
        df = pd.read_csv(fp, sep=" ", header=None,
                         names=["timestamp", "power"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")["power"]
        return df

    def read_mains(self, house: int,
                   channels: List[int]) -> pd.Series:
        series = [self._read_channel(house, c) for c in channels]
        merged = pd.concat(series, axis=1).sum(axis=1)
        return merged

    def read_appliance(self, house: int,
                       channel: int) -> pd.Series:
        return self._read_channel(house, channel)

    def get_appliance_channels(self, house: int,
                               appliance: str) -> List[int]:
        """从 labels.dat 找到电器对应的通道号"""
        label_fp = self.path / f"house_{house}" / "labels.dat"
        channels = []
        with open(label_fp) as f:
            for line in f:
                idx, name = line.strip().split(" ", 1)
                if appliance.lower() in name.lower():
                    channels.append(int(idx))
        return channels


class UKDALEReader:
    """读取 UKDALE HDF5 或 dat 格式数据"""

    def __init__(self, data_path: str):
        self.path = Path(data_path)

    def _read_channel(self, house: int, channel: int) -> pd.Series:
        fp = self.path / f"house_{house}" / f"channel_{channel}.dat"
        df = pd.read_csv(fp, sep=" ", header=None,
                         names=["timestamp", "power"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")["power"]
        return df

    def read_mains(self, house: int,
                   channels: List[int]) -> pd.Series:
        series = [self._read_channel(house, c) for c in channels]
        if len(series) == 1:
            return series[0]
        merged = pd.concat(series, axis=1).sum(axis=1)
        return merged

    def read_appliance(self, house: int,
                       channel: int) -> pd.Series:
        return self._read_channel(house, channel)

    def get_appliance_channels(self, house: int,
                               appliance: str) -> List[int]:
        label_fp = self.path / f"house_{house}" / "labels.dat"
        channels = []
        with open(label_fp) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                idx, name = parts
                if appliance.lower() in name.lower():
                    channels.append(int(idx))
        return channels


# ═══════════════════════════════════════════════════════════
#  2. 预处理工具
# ═══════════════════════════════════════════════════════════

def resample_series(s: pd.Series, freq_sec: int) -> pd.Series:
    """重采样到固定频率"""
    rule = f"{freq_sec}S"
    return s.resample(rule).mean()


def fill_gaps(s: pd.Series, max_gap_sec: int) -> pd.Series:
    """短缺口前值填充，长缺口填 0"""
    s = s.copy()
    max_gap = pd.Timedelta(seconds=max_gap_sec)
    # 找到缺失区间
    s_filled = s.fillna(method="ffill", limit=max_gap_sec)
    s_filled = s_filled.fillna(0.0)
    return s_filled


def attach_status(power: pd.Series,
                  threshold: float,
                  min_on_sec: int,
                  sample_rate: int) -> pd.Series:
    """根据功率判断开关状态"""
    status = (power > threshold).astype(float)
    min_on_steps = max(1, min_on_sec // sample_rate)
    # 过滤过短的开启片段
    from scipy.ndimage import label as scipy_label
    labeled, n = scipy_label(status.values)
    for i in range(1, n + 1):
        seg = labeled == i
        if seg.sum() < min_on_steps:
            status.values[seg] = 0.0
    return status


def standardize(arr: np.ndarray,
                mean: float,
                std: float) -> np.ndarray:
    return (arr - mean) / (std + 1e-8)


def sliding_windows(mains: np.ndarray,
                    appliances: np.ndarray,
                    time_codes: np.ndarray,
                    win: int,
                    stride: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    滑动窗口切片
    mains:      (T,)
    appliances: (T, N_appliances)
    time_codes: (T, 3)   [hour, dow, month] → 编码后
    返回:
        X  (n_windows, win)
        Y  (n_windows, N_appliances, win)
        TC (n_windows, 3, win)
    """
    T = len(mains)
    indices = range(0, T - win + 1, stride)
    X, Y, TC = [], [], []
    for i in indices:
        X.append(mains[i:i + win])
        Y.append(appliances[i:i + win, :].T)      # (N, win)
        TC.append(time_codes[i:i + win, :].T)     # (3, win)
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            np.array(TC, dtype=np.float32))


def encode_time(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    多尺度时间编码：[hour/23, dow/6, month/11] → [-0.5, 0.5]
    返回 (T, 3)
    """
    hour  = timestamps.hour.values  / 23.0 - 0.5
    dow   = timestamps.dayofweek.values / 6.0  - 0.5
    month = (timestamps.month.values - 1) / 11.0 - 0.5
    return np.stack([hour, dow, month], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  3. 统一数据集构建
# ═══════════════════════════════════════════════════════════

def build_dataset(dataset_name: str,
                  houses: List[int],
                  cfg: dict
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取多个房屋，返回合并后的滑动窗口数据
    Returns:
        X  (N, L)          总功率窗口，已标准化
        Y  (N, n_app, L)   各电器功率窗口，已归一化到 [0,1]
        TC (N, 3, L)       时间编码
    """
    if dataset_name == "redd":
        reader = REDDReader(cfg["path"])
    else:
        reader = UKDALEReader(cfg["path"])

    win    = PREPROCESS["window_size"]
    stride = PREPROCESS["stride"]
    sr     = cfg["sample_rate"]
    apps   = cfg["appliances"]

    all_X, all_Y, all_TC = [], [], []

    for house in houses:
        print(f"  [House {house}] 读取数据...")

        # ── 读取主电表
        mains_raw = reader.read_mains(house, cfg["mains_channel"])
        mains_rs  = resample_series(mains_raw, sr)
        mains_rs  = fill_gaps(mains_rs, PREPROCESS["fill_gap_sec"])

        # ── 读取各电器
        app_series = {}
        for app in apps:
            chs = reader.get_appliance_channels(house, app)
            if not chs:
                # 该房屋没有此电器，用 0 填充
                app_series[app] = pd.Series(
                    0.0, index=mains_rs.index)
                continue
            raw  = reader.read_appliance(house, chs[0])
            rs   = resample_series(raw, sr)
            rs   = fill_gaps(rs, PREPROCESS["fill_gap_sec"])
            # 对齐到 mains 时间轴
            rs   = rs.reindex(mains_rs.index, method="nearest",
                              fill_value=0.0)
            app_series[app] = rs

        # ── 对齐共同时间范围
        common_idx = mains_rs.index
        mains_arr = mains_rs.values.astype(np.float32)

        # ── 各电器归一化到 [0,1]
        app_matrix = np.zeros((len(common_idx), len(apps)),
                              dtype=np.float32)
        for j, app in enumerate(apps):
            max_p = cfg["max_power"][app]
            arr   = np.clip(app_series[app].values, 0, max_p)
            app_matrix[:, j] = arr / max_p

        # ── 主电表标准化（均值-方差）
        mean_m = mains_arr.mean()
        std_m  = mains_arr.std()
        mains_norm = standardize(mains_arr, mean_m, std_m)

        # ── 时间编码
        tc = encode_time(common_idx)

        # ── 滑动窗口
        X, Y, TC = sliding_windows(mains_norm, app_matrix,
                                   tc, win, stride)
        all_X.append(X)
        all_Y.append(Y)
        all_TC.append(TC)
        print(f"     → {len(X)} 个窗口")

    return (np.concatenate(all_X, axis=0),
            np.concatenate(all_Y, axis=0),
            np.concatenate(all_TC, axis=0))


# ═══════════════════════════════════════════════════════════
#  4. PyTorch Dataset
# ═══════════════════════════════════════════════════════════

class NILMDataset(Dataset):
    """
    item 返回:
        mains  (1, L)       总功率（含通道维）
        target (N_app, L)   各电器功率
        time   (3, L)       时间编码
    """

    def __init__(self,
                 X: np.ndarray,   # (N, L)
                 Y: np.ndarray,   # (N, n_app, L)
                 TC: np.ndarray): # (N, 3, L)
        self.X  = torch.from_numpy(X).unsqueeze(1)   # (N,1,L)
        self.Y  = torch.from_numpy(Y)                 # (N,n_app,L)
        self.TC = torch.from_numpy(TC)                # (N,3,L)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.TC[idx]


# ═══════════════════════════════════════════════════════════
#  5. 公共接口
# ═══════════════════════════════════════════════════════════

def get_dataloaders(dataset_name: str,
                    batch_size: int = TRAIN["batch_size"],
                    num_workers: int = TRAIN["num_workers"]
                    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    返回 (train_loader, val_loader, test_loader)
    """
    cfg = DATASET_CONFIG[dataset_name]

    print(f"[数据集] {dataset_name.upper()} — 构建训练集...")
    X_tr, Y_tr, TC_tr = build_dataset(
        dataset_name, cfg["train_houses"], cfg)

    print(f"[数据集] {dataset_name.upper()} — 构建测试集...")
    X_te, Y_te, TC_te = build_dataset(
        dataset_name, cfg["test_houses"], cfg)

    # 划分验证集
    full_train = NILMDataset(X_tr, Y_tr, TC_tr)
    n_val  = int(len(full_train) * TRAIN["val_ratio"])
    n_tr   = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_tr, n_val],
        generator=torch.Generator().manual_seed(TRAIN["seed"]))

    test_ds = NILMDataset(X_te, Y_te, TC_te)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"  训练: {len(train_ds)}  验证: {len(val_ds)}"
          f"  测试: {len(test_ds)}")
    return train_loader, val_loader, test_loader
