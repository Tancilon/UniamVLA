"""Conditional pixel diffusion: clean-image prediction with JiT's v-space loss.

The condition is the VLM's future-query output, never ground-truth visual
features. Camera views share weights and are folded into the batch dimension.
Parameterization reference: https://github.com/LTH14/JiT/blob/main/denoiser.py
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal(values, dim):
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(
            dim // 2,
            device=values.device,
            dtype=torch.float32,
        )
        / (dim // 2)
    )
    phase = values.float().unsqueeze(-1) * frequencies
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.q_norm = nn.RMSNorm(dim // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // heads, eps=1e-6)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context):
        b, n, d = x.shape
        q = self.q(x).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(context).reshape(b, -1, 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v, dropout_p=0.0)
        return self.out(out.transpose(1, 2).reshape(b, n, d))


class _Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norms = nn.ModuleList([nn.RMSNorm(dim, eps=1e-6) for _ in range(3)])
        self.self_attn = _Attention(dim, heads)
        self.cross_attn = _Attention(dim, heads)
        width = int(4 * dim * 2 / 3)
        self.ff_in = nn.Linear(dim, width * 2)
        self.ff_out = nn.Linear(width, dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, condition, time):
        mods = self.modulation(time).unsqueeze(1).chunk(9, dim=-1)
        h = self.norms[0](x) * (1 + mods[1]) + mods[0]
        x = x + mods[2] * self.self_attn(h, h)
        h = self.norms[1](x) * (1 + mods[4]) + mods[3]
        x = x + mods[5] * self.cross_attn(h, condition)
        h = self.norms[2](x) * (1 + mods[7]) + mods[6]
        a, b = self.ff_in(h).chunk(2, dim=-1)
        return x + mods[8] * self.ff_out(F.silu(a) * b)


class FutureRGBDenoiser(nn.Module):
    def __init__(
        self,
        condition_dim,
        hidden_dim=512,
        depth=4,
        num_heads=8,
        image_size=224,
        patch_size=16,
        bottleneck_dim=128,
        time_mean=-0.8,
        time_std=0.8,
        time_eps=0.05,
        sampling_steps=50,
    ):
        super().__init__()
        if image_size % patch_size or hidden_dim % num_heads or hidden_dim % 4:
            raise ValueError("Invalid image/patch/attention dimensions")
        if not 0 < time_eps < 1 or time_std <= 0 or sampling_steps < 1:
            raise ValueError("Invalid diffusion time configuration")
        self.image_size, self.patch_size = image_size, patch_size
        self.hidden_dim = hidden_dim
        self.time_mean, self.time_std, self.time_eps = time_mean, time_std, time_eps
        self.sampling_steps = sampling_steps
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size**2 * 3, bottleneck_dim, bias=False),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.condition_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.time_embed = nn.Sequential(nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([_Block(hidden_dim, num_heads) for _ in range(depth)])
        self.output_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        self.output_proj = nn.Linear(hidden_dim, patch_size**2 * 3)
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def patchify(self, x):
        b, c, h, w = x.shape
        if (c, h, w) != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected RGB at {self.image_size}px, got {tuple(x.shape)}")
        p = self.patch_size
        return x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p * p * c)

    def unpatchify(self, x):
        p, g = self.patch_size, self.image_size // self.patch_size
        return x.reshape(-1, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, self.image_size, self.image_size)

    def forward(self, noisy_rgb, time, condition):
        """Predict clean [-1,1] RGB; conditioning shape is [B,N,D]."""
        dtype = self.patch_embed[0].weight.dtype
        x = self.patch_embed(self.patchify(noisy_rgb).to(dtype))
        g = self.image_size // self.patch_size
        # Compute fixed positions in FP32, including after serving .to(bfloat16).
        yy, xx = torch.meshgrid(torch.arange(g, device=x.device), torch.arange(g, device=x.device), indexing="ij")
        pos = torch.cat(
            (sinusoidal(yy.flatten(), self.hidden_dim // 2), sinusoidal(xx.flatten(), self.hidden_dim // 2)), dim=-1
        )
        x = x + pos.to(x.dtype)
        time = self.time_embed(sinusoidal(time, 256).to(dtype))
        condition = self.condition_norm(self.condition_proj(condition.to(dtype)))
        for block in self.blocks:
            x = block(x, condition, time)
        shift, scale = self.output_modulation(time).unsqueeze(1).chunk(2, dim=-1)
        x = self.output_norm(x) * (1 + scale) + shift
        return self.unpatchify(self.output_proj(x))

    def compute_loss(self, future_hidden, future_rgb):
        """RGB labels [B,V,3,H,W] in [0,1]; return unweighted FP32 losses."""
        b, v, n, d = future_hidden.shape
        if future_rgb.shape != (b, v, 3, self.image_size, self.image_size):
            raise ValueError("Future RGB shape must match query batch/views and configured size")
        if not torch.isfinite(future_rgb).all() or (future_rgb < 0).any() or (future_rgb > 1).any():
            raise ValueError("Future RGB labels must be finite and in [0,1]")
        with torch.autocast(future_hidden.device.type, enabled=False):
            clean = future_rgb.detach().float().flatten(0, 1) * 2 - 1
            time = (torch.randn(b * v, device=clean.device) * self.time_std + self.time_mean).sigmoid()
            tau = time[:, None, None, None]
            noise = torch.randn_like(clean)
            noisy = tau * clean + (1 - tau) * noise
        prediction = self(noisy, time, future_hidden.reshape(b * v, n, d))
        with torch.autocast(future_hidden.device.type, enabled=False):
            error = prediction.float() - clean
            loss = (error / (1 - tau).clamp_min(self.time_eps)).square().mean()
            rgb_mse = (error / 2).square().mean().detach()
        return loss, rgb_mse

    @torch.no_grad()
    def sample(self, future_hidden, generator=None, steps=None):
        """Offline Euler sampling; returns [B,V,3,H,W] RGB in [0,1]."""
        b, v, n, d = future_hidden.shape
        steps = self.sampling_steps if steps is None else int(steps)
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        x = torch.randn(
            b * v,
            3,
            self.image_size,
            self.image_size,
            device=future_hidden.device,
            dtype=torch.float32,
            generator=generator,
        )
        condition = future_hidden.reshape(b * v, n, d)
        for index in range(steps):
            time = torch.full((b * v,), index / steps, device=x.device, dtype=torch.float32)
            clean = self(x, time, condition).float()
            x = x + (clean - x) / max(1 - index / steps, self.time_eps) / steps
        return ((x + 1) / 2).clamp(0, 1).reshape(b, v, 3, self.image_size, self.image_size)
