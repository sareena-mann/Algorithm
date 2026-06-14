"""
Diffusion Transformer (DiT) for DJ transition inpainting.

Architecture overview:
  - Input: [t_token | cond_token | context_A tokens | noisy_bridge tokens | context_B tokens]
  - Output: predicted noise in the bridge region only
  - Training: DDPM (predict epsilon, MSE loss)
  - Conditioning: BPM, key compatibility, genre one-hots

Shapes throughout (B=batch, T=time, D=latent_dim, M=model_dim):
  context_a    : (B, T_a,      LATENT_DIM)
  noisy_bridge : (B, T_bridge, LATENT_DIM)
  context_b    : (B, T_b,      LATENT_DIM)
  conditioning : (B, COND_DIM)
  timestep     : (B,)          int  in [0, T_STEPS)
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

LATENT_DIM = 1024  # must match codec.py (DAC 44kHz encoder output dim)
COND_DIM   = 28    # 4 float scalars + 2 × 12 genre one-hots
T_STEPS    = 1000  # DDPM noise schedule steps

# ── Noise schedule (cosine) ────────────────────────────────────────────────

def _cosine_alphas(T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns alpha_bar (cumulative product) and alpha (step product)."""
    t = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((t / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    alpha     = alpha_bar[1:] / alpha_bar[:-1]
    return alpha_bar[1:], alpha


ALPHA_BAR, ALPHA = _cosine_alphas(T_STEPS)


# ── Building blocks ────────────────────────────────────────────────────────

class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb   = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLN(nn.Module):
    """Adaptive Layer Norm — injects timestep + conditioning into every block."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm    = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj    = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.proj(cond).chunk(2, dim=-1)  # (B, dim) each
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, dropout: float):
        super().__init__()
        self.attn_norm = AdaLN(dim, cond_dim)
        self.attn      = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff_norm   = AdaLN(dim, cond_dim)
        self.ff        = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.ff_norm(x, cond))
        return x


# ── Main model ─────────────────────────────────────────────────────────────

class TransitionDiT(nn.Module):
    """
    Predicts the noise added to the bridge at diffusion timestep t.
    Call `add_noise()` during training and `sample()` during inference.
    """

    def __init__(
        self,
        model_dim: int  = 512,
        n_heads:   int  = 8,
        n_layers:  int  = 12,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim

        # Segment type: 0=context_a, 1=bridge, 2=context_b
        self.seg_embed = nn.Embedding(3, model_dim)
        self.in_proj   = nn.Linear(LATENT_DIM, model_dim)

        # Timestep + conditioning → one combined conditioning vector
        t_dim = model_dim
        self.t_embed = nn.Sequential(
            SinusoidalEmb(t_dim),
            nn.Linear(t_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.c_embed = nn.Sequential(
            nn.Linear(COND_DIM, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        self.blocks  = nn.ModuleList([
            DiTBlock(model_dim, n_heads, model_dim, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, LATENT_DIM)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        context_a:    torch.Tensor,
        noisy_bridge: torch.Tensor,
        context_b:    torch.Tensor,
        timestep:     torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Returns predicted noise, shape (B, T_bridge, LATENT_DIM)."""
        B   = context_a.shape[0]
        T_a = context_a.shape[1]
        T_br = noisy_bridge.shape[1]
        T_b = context_b.shape[1]

        # Project latents to model dim
        a  = self.in_proj(context_a)
        br = self.in_proj(noisy_bridge)
        b  = self.in_proj(context_b)

        # Add segment ids
        a  = a  + self.seg_embed(torch.zeros(B, T_a,  dtype=torch.long, device=a.device))
        br = br + self.seg_embed(torch.ones (B, T_br, dtype=torch.long, device=br.device))
        b  = b  + self.seg_embed(torch.full((B, T_b), 2, dtype=torch.long, device=b.device))

        seq = torch.cat([a, br, b], dim=1)  # (B, T_total, model_dim)

        # Combined adaptive conditioning
        cond = self.t_embed(timestep) + self.c_embed(conditioning)  # (B, model_dim)

        for block in self.blocks:
            seq = block(seq, cond)

        # Extract bridge slice only
        out = self.out_norm(seq[:, T_a: T_a + T_br, :])
        return self.out_proj(out)  # (B, T_bridge, LATENT_DIM)


# ── DDPM helpers (used by train.py and infer.py) ───────────────────────────

def add_noise(x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: returns (x_t, epsilon)."""
    alpha_bar = ALPHA_BAR.to(x0.device)[t].float()
    while alpha_bar.dim() < x0.dim():
        alpha_bar = alpha_bar.unsqueeze(-1)
    epsilon = torch.randn_like(x0)
    x_t = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * epsilon
    return x_t, epsilon


@torch.no_grad()
def ddpm_sample(
    model:        TransitionDiT,
    context_a:    torch.Tensor,
    context_b:    torch.Tensor,
    conditioning: torch.Tensor,
    bridge_len:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Ancestral sampling — generates bridge latents from pure noise.
    Returns (1, bridge_len, LATENT_DIM).
    """
    model.eval()
    B = context_a.shape[0]
    x = torch.randn(B, bridge_len, LATENT_DIM, device=device)

    alpha_bar = ALPHA_BAR.to(device)
    alpha     = ALPHA.to(device)

    for t in range(T_STEPS - 1, -1, -1):
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        eps_pred = model(context_a, x, context_b, t_batch, conditioning)

        a_t  = alpha[t]
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        x0_pred = x0_pred.clamp(-4, 4)

        # DDPM reverse step
        mean = (a_t.sqrt() * ab_prev) / (1 - ab_t) * x + \
               ((1 - ab_prev).sqrt() * (1 - a_t)) / (1 - ab_t) * x0_pred

        if t > 0:
            var  = (1 - ab_prev) / (1 - ab_t) * (1 - a_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean

    return x
