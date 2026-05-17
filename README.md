# MultiDiffNILM

**UNet 多通道输出 + DiffNILM 扩散骨干**
一次推理同时输出所有电器功率波形，支持 REDD 和 UKDALE 数据集。

---

## 项目结构

```
multi_diff_nilm/
├── config.py        # 全局超参数配置
├── data_loader.py   # REDD / UKDALE 数据加载与预处理
├── model.py         # MultiDiffNILM 核心模型
├── metrics.py       # 分类 & 回归评估指标
├── train.py         # 训练主程序
├── evaluate.py      # 测试集评估
├── demo.py          # 快速冒烟测试（无需真实数据）
└── requirements.txt
```

---

## 核心创新点

| 模块 | 来源 | 作用 |
|------|------|------|
| Bi-DilConv 残差层 | DiffNILM | 扩大时序感受野 |
| 多通道输出头 | UNet-NILM | 同时输出 N 个电器功率 |
| 物理守恒损失 | 本文提出 | 约束各电器之和 ≈ 总功率 |
| 连续噪声水平 | DiffNILM | 训练更稳定 |
| 快速采样（8步） | DiffNILM | 推理加速 |

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 冒烟测试（不需要数据集）
```bash
python demo.py
```

### 3. 准备数据集

**REDD**
- 下载：http://redd.csail.mit.edu/
- 解压到 `./data/redd/`
- 目录结构：`data/redd/house_1/channel_1.dat` ...

**UKDALE**
- 下载：https://jack-kelly.com/data/
- 解压到 `./data/ukdale/`
- 目录结构：`data/ukdale/house_1/channel_1.dat` ...

### 4. 训练
```bash
# 训练 REDD 数据集
python train.py --dataset redd

# 训练 UKDALE 数据集
python train.py --dataset ukdale

# 自定义超参数
python train.py --dataset redd --epochs 300 --lr 1e-4 --lambda_p 0.2

# 断点续训
python train.py --dataset redd --resume checkpoints/redd/last.pt
```

### 5. 评估
```bash
python evaluate.py --dataset redd   --ckpt checkpoints/redd/best.pt
python evaluate.py --dataset ukdale --ckpt checkpoints/ukdale/best.pt
```

---

## 模型架构

```
输入:
  x_aggre  (B, 1, L)     ← 总有功功率（标准化）
  x_time   (B, 3, L)     ← 时间编码 [hour, dow, month]

扩散过程（训练）:
  x0 (B, N, L) → 加噪 → x_t
                            ↓
                     [DenoiseNet] ← noise_level √ᾱ
                            ↓
                         ε_hat (B, N, L)
                            ↓
                     loss = log‖ε - ε_hat‖ + λ·物理守恒损失

采样（推理）:
  xT ~ N(0,I) → 8步去噪 → x0_hat (B, N, L)

输出:
  [电器1功率, 电器2功率, ..., 电器N功率]  同时输出 ✓
```

---

## 超参数说明（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 480 | 滑动窗口步数（480×6s = 48min）|
| `T` | 1000 | 最大扩散步数 |
| `T_infer` | 8 | 快速采样步数 |
| `residual_channels` | 128 | 残差通道数 |
| `residual_layers` | 30 | 残差层数 |
| `lambda_physics` | 0.1 | 物理守恒损失权重 |
| `learning_rate` | 3e-5 | 初始学习率 |

---

## 参考文献

1. Sun et al., *DiffNILM*, Sensors 2023
2. Faustine et al., *UNet-NILM*, ACM BuildSys 2020
3. Kong et al., *DiffWave*, ICLR 2021
4. Ho et al., *DDPM*, NeurIPS 2020
