"""
model.py — MultiDiffNILM 核心网络
结构：DiffNILM 去噪骨干（Bi-DilConv 残差网络）
      + UNet-NILM 多通道输出头
      → 一次推理同时输出所有电器功率波形
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from config import NETWORK, DIFFUSION


# ═══════════════════════════════════════════════════════════
#  工具模块
# ═══════════════════════════════════════════════════════════

class SiLU(nn.Module):
    """Sigmoid Linear Unit"""
    def forward(self, x):
        return x * torch.sigmoid(x)


def sinusoidal_embedding(noise_level: torch.Tensor,
                         dim: int = 128) -> torch.Tensor:
    """
    将连续噪声水平 sqrt(ᾱ) 编码为正弦位置向量
    noise_level: (B,)
    返回:       (B, dim)
    """
    half = dim // 2
    # 频率因子与 DiffNILM 论文 Eq.(12) 一致
    freqs = 10 ** (-torch.arange(half, dtype=torch.float32,
                                  device=noise_level.device) / half * 4)
    args  = noise_level[:, None] * freqs[None] * 50000
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ═══════════════════════════════════════════════════════════
#  双向膨胀卷积残差层（来自 DiffWave / DiffNILM）
# ═══════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    参数
    ----
    C          : 残差通道数
    dilation   : 当前层的膨胀率
    cond_dim   : 条件输入通道数（聚合功率 + 时间特征）
    """

    def __init__(self, C: int, dilation: int, cond_dim: int):
        super().__init__()
        # 双向膨胀卷积：正向 + 反向拼接实现双向感受野
        self.conv_fwd  = nn.Conv1d(C, 2 * C, kernel_size=3,
                                   padding=dilation,
                                   dilation=dilation)
        self.conv_bwd  = nn.Conv1d(C, 2 * C, kernel_size=3,
                                   padding=dilation,
                                   dilation=dilation)
        # 条件投影
        self.cond_proj = nn.Conv1d(cond_dim, 2 * C, kernel_size=1)
        # 残差 & 跳跃输出
        self.res_conv  = nn.Conv1d(C, C, kernel_size=1)
        self.skip_conv = nn.Conv1d(C, C, kernel_size=1)

    def forward(self,
                x: torch.Tensor,        # (B, C, L)
                cond: torch.Tensor      # (B, cond_dim, L)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 双向卷积
        h_fwd = self.conv_fwd(x)
        h_bwd = self.conv_bwd(x.flip(-1)).flip(-1)
        h = h_fwd + h_bwd                          # (B, 2C, L)

        # 加入条件信息
        h = h + self.cond_proj(cond)               # (B, 2C, L)

        # 门控激活（tanh ⊙ sigmoid）
        h_tanh  = torch.tanh(h[:, :h.shape[1]//2])
        h_sig   = torch.sigmoid(h[:, h.shape[1]//2:])
        h = h_tanh * h_sig                         # (B, C, L)

        # 残差 & 跳跃
        res  = self.res_conv(h) + x                # (B, C, L)
        skip = self.skip_conv(h)                   # (B, C, L)
        return res, skip


# ═══════════════════════════════════════════════════════════
#  噪声预测网络（去噪骨干）
# ═══════════════════════════════════════════════════════════

class DenoiseNet(nn.Module):
    """
    输入:
        x_noisy    : (B, N_app, L)   含噪多电器功率
        noise_lvl  : (B,)            连续噪声水平 sqrt(ᾱ)
        x_aggre    : (B, 1, L)       聚合总功率（条件）
        x_time     : (B, 3, L)       时间编码（条件）
    输出:
        eps_hat    : (B, N_app, L)   预测噪声
    """

    def __init__(self,
                 n_appliances: int,
                 C:   int = NETWORK["residual_channels"],
                 N:   int = NETWORK["residual_layers"],
                 n:   int = NETWORK["dilation_cycle"],
                 emb_dim: int = NETWORK["time_embed_dim"]):
        super().__init__()
        self.n_app = n_appliances
        self.C     = C

        # ── 输入投影：多通道含噪信号 → C 通道
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_appliances, C, kernel_size=1),
            SiLU()
        )

        # ── 条件输入投影（聚合功率 + 时间编码）
        cond_raw_dim = 1 + 3   # 1(aggre) + 3(time)
        self.cond_proj = nn.Sequential(
            nn.Conv1d(cond_raw_dim, C, kernel_size=1),
            SiLU()
        )
        cond_dim = C

        # ── 噪声水平嵌入 → 映射到 C 维偏置
        self.noise_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), SiLU(),
            nn.Linear(emb_dim, C),       SiLU(),
        )

        # ── 残差层
        self.res_blocks = nn.ModuleList([
            ResidualBlock(C, dilation=2 ** (i % n),
                          cond_dim=cond_dim)
            for i in range(N)
        ])

        # ── 输出头：skip 求和 → 多通道噪声预测
        self.output_head = nn.Sequential(
            SiLU(),
            nn.Conv1d(C, C, kernel_size=1),
            SiLU(),
            nn.Conv1d(C, n_appliances, kernel_size=1),  # ← 多通道输出
        )

    def forward(self,
                x_noisy:   torch.Tensor,   # (B, N_app, L)
                noise_lvl: torch.Tensor,   # (B,)
                x_aggre:   torch.Tensor,   # (B, 1, L)
                x_time:    torch.Tensor,   # (B, 3, L)
                ) -> torch.Tensor:         # (B, N_app, L)

        B, _, L = x_noisy.shape

        # 输入投影
        h = self.input_proj(x_noisy)                     # (B, C, L)

        # 噪声水平嵌入 → 偏置项
        noise_emb = sinusoidal_embedding(
            noise_lvl, dim=NETWORK["time_embed_dim"])    # (B, emb_dim)
        bias = self.noise_mlp(noise_emb)                 # (B, C)
        h = h + bias.unsqueeze(-1)                       # (B, C, L)

        # 条件拼接（聚合功率 + 时间编码）
        cond = torch.cat([x_aggre, x_time], dim=1)       # (B,4,L)
        cond = self.cond_proj(cond)                       # (B,C,L)

        # 残差层前向传播，累积 skip
        skip_sum = torch.zeros_like(h)
        for block in self.res_blocks:
            h, skip = block(h, cond)
            skip_sum = skip_sum + skip

        # 输出头
        eps_hat = self.output_head(skip_sum)              # (B, N_app, L)
        return eps_hat


# ═══════════════════════════════════════════════════════════
#  扩散过程控制（前向 & 逆向）
# ═══════════════════════════════════════════════════════════

class DiffusionProcess(nn.Module):
    """
    封装噪声调度、前向加噪、逆向采样
    """

    def __init__(self):
        super().__init__()
        T         = DIFFUSION["T"]
        beta_s    = DIFFUSION["beta_start"]
        beta_e    = DIFFUSION["beta_end"]

        # 线性噪声调度
        betas     = torch.linspace(beta_s, beta_e, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        sqrt_ab   = torch.sqrt(alpha_bar)
        sqrt_1mab = torch.sqrt(1.0 - alpha_bar)

        # 注册为 buffer（不参与梯度，随模型迁移设备）
        self.register_buffer("betas",      betas)
        self.register_buffer("alphas",     alphas)
        self.register_buffer("alpha_bar",  alpha_bar)
        self.register_buffer("sqrt_ab",    sqrt_ab)
        self.register_buffer("sqrt_1mab",  sqrt_1mab)

    def q_sample(self,
                 x0: torch.Tensor,          # (B, N_app, L)
                 t:  torch.Tensor,          # (B,) int index
                 eps: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor,   # x_t
                            torch.Tensor,   # eps
                            torch.Tensor]:  # sqrt(ᾱ_t)
        """前向加噪（闭合公式）"""
        if eps is None:
            eps = torch.randn_like(x0)
        sab   = self.sqrt_ab[t].view(-1, 1, 1)
        s1mab = self.sqrt_1mab[t].view(-1, 1, 1)
        x_t   = sab * x0 + s1mab * eps
        noise_lvl = self.sqrt_ab[t]                      # (B,)
        return x_t, eps, noise_lvl

    def sample_noise_level(self,
                           t: torch.Tensor
                           ) -> torch.Tensor:
        """
        连续噪声水平采样：sqrt(ᾱ) ~ Uniform(sqrt(ᾱ_t), sqrt(ᾱ_{t-1}))
        """
        lo = self.sqrt_ab[t]
        hi = torch.where(t > 0,
                         self.sqrt_ab[t - 1],
                         torch.ones_like(lo))
        noise_lvl = lo + (hi - lo) * torch.rand_like(lo)
        return noise_lvl

    @torch.no_grad()
    def p_sample_loop(self,
                      net: DenoiseNet,
                      x_aggre: torch.Tensor,   # (B,1,L)
                      x_time:  torch.Tensor,   # (B,3,L)
                      n_app:   int,
                      device:  torch.device
                      ) -> torch.Tensor:       # (B, N_app, L)
        """快速逆向采样（T_infer 步）"""
        infer_betas = torch.tensor(
            DIFFUSION["infer_schedule"],
            dtype=torch.float32, device=device)
        T_inf   = len(infer_betas)
        alphas  = 1.0 - infer_betas
        ab      = torch.cumprod(alphas, dim=0)

        B, _, L = x_aggre.shape
        x = torch.randn(B, n_app, L, device=device)

        for step in reversed(range(T_inf)):
            t_tensor = torch.full((B,), ab[step],
                                  device=device)
            eps_hat = net(x, t_tensor, x_aggre, x_time)

            # 均值计算
            alpha_t = alphas[step]
            ab_t    = ab[step]
            mu = (x - infer_betas[step] / torch.sqrt(1 - ab_t)
                  * eps_hat) / torch.sqrt(alpha_t)

            if step > 0:
                ab_prev = ab[step - 1]
                beta_tilde = ((1 - ab_prev) / (1 - ab_t)
                              * infer_betas[step])
                z = torch.randn_like(x)
                x = mu + torch.sqrt(beta_tilde) * z
            else:
                x = mu

        return x.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════
#  完整 MultiDiffNILM 模型
# ═══════════════════════════════════════════════════════════

class MultiDiffNILM(nn.Module):
    """
    UNet多通道输出 + DiffNILM扩散骨干

    训练时：调用 compute_loss(mains, target, time_codes)
    推理时：调用 disaggregate(mains, time_codes)
    """

    def __init__(self, n_appliances: int):
        super().__init__()
        self.n_app   = n_appliances
        self.denoise = DenoiseNet(n_appliances)
        self.diff    = DiffusionProcess()

    # ── 训练损失 ──────────────────────────────────────────
    def compute_loss(self,
                     x_aggre: torch.Tensor,   # (B,1,L)
                     x0:      torch.Tensor,   # (B,N,L) 目标功率
                     x_time:  torch.Tensor,   # (B,3,L)
                     lambda_physics: float = 0.1
                     ) -> dict:
        B = x_aggre.shape[0]
        device = x_aggre.device

        # 随机时间步
        t = torch.randint(0, DIFFUSION["T"], (B,), device=device)

        # 连续噪声水平
        noise_lvl = self.diff.sample_noise_level(t)   # (B,)

        # 前向加噪
        eps = torch.randn_like(x0)
        sab   = self.diff.sqrt_ab[t].view(-1, 1, 1)
        s1mab = self.diff.sqrt_1mab[t].view(-1, 1, 1)
        x_t   = sab * x0 + s1mab * eps

        # 预测噪声
        eps_hat = self.denoise(x_t, noise_lvl, x_aggre, x_time)

        # ① 去噪损失（log-norm，来自 DiffNILM）
        diff = eps - eps_hat
        loss_denoise = torch.log(
            diff.abs().mean(dim=[1, 2]) + 1e-8).mean()

        # ② 物理守恒损失（UNet-NILM 多输出约束）
        #    还原 x0_hat，要求各电器之和 ≈ 归一化总功率
        x0_hat = (x_t - s1mab * eps_hat) / (sab + 1e-8)
        x0_hat = x0_hat.clamp(0.0, 1.0)
        #    将聚合功率也归一到 [0,1]（按同样最大值）
        aggre_norm = x_aggre.clamp(0.0, 1.0)
        loss_physics = F.mse_loss(
            x0_hat.sum(dim=1, keepdim=True), aggre_norm)

        loss_total = loss_denoise + lambda_physics * loss_physics

        return {
            "loss":         loss_total,
            "loss_denoise": loss_denoise,
            "loss_physics": loss_physics,
        }

    # ── 推理（采样）──────────────────────────────────────
    @torch.no_grad()
    def disaggregate(self,
                     x_aggre: torch.Tensor,   # (B,1,L)
                     x_time:  torch.Tensor    # (B,3,L)
                     ) -> torch.Tensor:       # (B,N,L)
        device = x_aggre.device
        return self.diff.p_sample_loop(
            self.denoise, x_aggre, x_time,
            self.n_app, device)
