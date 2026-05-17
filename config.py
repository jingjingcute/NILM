"""
config.py — 全局超参数配置
MultiDiffNILM: UNet多通道输出 + DiffNILM扩散骨干
支持 REDD 和 UKDALE 数据集
"""

# ───────────────────────────────────────────
# 数据集配置
# ───────────────────────────────────────────
DATASET_CONFIG = {
    "redd": {
        "path": "./data/redd",          # REDD 数据根目录
        "sample_rate": 6,               # 采样间隔（秒）
        "houses": [1, 2, 3, 4, 5, 6],
        "train_houses": [1, 2, 3, 5],
        "test_houses": [4, 6],
        "appliances": ["microwave", "washer", "dishwasher", "refrigerator"],
        "on_power_threshold": {         # 电器"开启"最低功率（W）
            "microwave":   200,
            "washer":       40,
            "dishwasher":   50,
            "refrigerator": 50,
        },
        "max_power": {                  # 用于归一化的最大功率（W）
            "microwave":   1800,
            "washer":      3500,
            "dishwasher":  1200,
            "refrigerator": 400,
        },
        "min_on_duration": {            # 最短开启时长（秒）
            "microwave":    12,
            "washer":     1800,
            "dishwasher": 1800,
            "refrigerator": 60,
        },
        "mains_channel": [1, 2],        # REDD 两相电源通道
    },
    "ukdale": {
        "path": "./data/ukdale",        # UKDALE 数据根目录
        "sample_rate": 6,
        "houses": [1, 2, 3, 4, 5],
        "train_houses": [1, 3, 4, 5],
        "test_houses": [2],
        "appliances": [
            "microwave", "washer", "dishwasher", "refrigerator", "kettle"
        ],
        "on_power_threshold": {
            "microwave":   200,
            "washer":       20,
            "dishwasher":   10,
            "refrigerator": 50,
            "kettle":      200,
        },
        "max_power": {
            "microwave":   3000,
            "washer":      2500,
            "dishwasher":  2500,
            "refrigerator": 400,
            "kettle":      3100,
        },
        "min_on_duration": {
            "microwave":    12,
            "washer":     1800,
            "dishwasher": 1800,
            "refrigerator": 60,
            "kettle":       12,
        },
        "mains_channel": [1],
    },
}

# ───────────────────────────────────────────
# 数据预处理配置
# ───────────────────────────────────────────
PREPROCESS = {
    "window_size":   480,   # 滑动窗口长度（步数）
    "stride":        120,   # 滑动步长
    "fill_gap_sec":  180,   # 短于此时长的缺口用前值填充（秒）
}

# ───────────────────────────────────────────
# 扩散模型超参数
# ───────────────────────────────────────────
DIFFUSION = {
    "T":              1000,     # 最大扩散步数
    "T_infer":           8,     # 快速采样步数
    "beta_start":    1e-6,      # 噪声调度起点
    "beta_end":      6e-3,      # 噪声调度终点
    # 快速采样噪声调度（8步）
    "infer_schedule": [
        1e-6, 2e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 9e-1
    ],
}

# ───────────────────────────────────────────
# 网络结构超参数
# ───────────────────────────────────────────
NETWORK = {
    "residual_channels":  128,   # 残差层通道数 C
    "residual_layers":     30,   # 残差层数 N
    "dilation_cycle":      10,   # 膨胀周期 n
    "time_embed_dim":     128,   # 时间特征嵌入维度
}

# ───────────────────────────────────────────
# 训练超参数
# ───────────────────────────────────────────
TRAIN = {
    "batch_size":      32,
    "learning_rate": 3e-5,
    "max_epochs":     200,
    "adam_beta1":     0.5,
    "adam_beta2":     0.999,
    "lambda_physics": 0.1,   # 物理守恒损失权重
    "val_ratio":      0.1,   # 验证集比例
    "seed":            42,
    "num_workers":      4,
    "save_dir":     "./checkpoints",
    "log_dir":      "./logs",
}
