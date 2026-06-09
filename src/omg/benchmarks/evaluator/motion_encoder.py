from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths: Optional[torch.Tensor], frames: int, device: torch.device) -> torch.Tensor:
    if lengths is None:
        return torch.ones(1, frames, dtype=torch.bool, device=device)
    return torch.arange(frames, device=device)[None, :] < lengths[:, None].to(device)


class MovementEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 512,
        output_dim: int = 512,
        mode: str = "conv",
        dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in {"linear", "conv"}:
            raise ValueError(f"movement mode must be 'linear' or 'conv', got {mode}")
        self.mode = str(mode)
        self.downsample_factor = 4 if self.mode == "conv" else 1
        if self.mode == "linear":
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Conv1d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1),
                nn.Dropout(dropout),
                nn.GELU(),
            )
            self.out_norm = nn.LayerNorm(output_dim)

    def output_lengths(self, lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if lengths is None or self.downsample_factor == 1:
            return lengths
        return torch.div(lengths.clamp_min(self.downsample_factor), self.downsample_factor, rounding_mode="floor")

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            return self.net(motion)
        h = self.net(motion.transpose(1, 2)).transpose(1, 2)
        return self.out_norm(h)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device))
        return freqs.cos().to(dtype)[None, None], freqs.sin().to(dtype)[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class RopeSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = float(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        cos, sin = self.rope(seq_len, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn_mask = None if valid_mask is None else valid_mask[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim))


class RopeTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RopeSelfAttention(hidden_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), valid_mask=valid_mask))
        return x + self.mlp(self.norm2(x))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 4096):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        kind: str = "transformer_rope",
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        if kind not in {"transformer_rope", "transformer_sin", "bigru"}:
            raise ValueError(f"Unsupported temporal encoder kind: {kind}")
        self.kind = str(kind)
        if self.kind == "transformer_rope":
            self.layers = nn.ModuleList([RopeTransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(hidden_dim)
        elif self.kind == "transformer_sin":
            self.pos = SinusoidalPositionalEncoding(hidden_dim, max_len=max_len)
            layer = nn.TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                int(hidden_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.kind == "transformer_rope":
            for layer in self.layers:
                x = layer(x, valid_mask=valid_mask)
            return self.norm(x)
        if self.kind == "transformer_sin":
            return self.norm(self.encoder(self.pos(x), src_key_padding_mask=~valid_mask))
        if lengths is None:
            lengths = valid_mask.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.gru(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        return self.norm(y)


class MotionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 36,
        movement_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        movement_mode: str = "conv",
        temporal_kind: str = "transformer_rope",
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 4096,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.movement_encoder = MovementEncoder(input_dim, hidden_dim, movement_dim, mode=movement_mode, dropout=dropout)
        self.to_hidden = nn.Identity() if movement_dim == hidden_dim else nn.Linear(movement_dim, hidden_dim)
        self.temporal_encoder = TemporalEncoder(hidden_dim, temporal_kind, num_layers, num_heads, mlp_ratio, dropout, max_len)
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if motion.ndim == 4:
            motion = motion.flatten(start_dim=2)
        if motion.ndim != 3:
            raise ValueError(f"MotionEncoder expects motion shape (B,T,D) or (B,T,J,3), got {tuple(motion.shape)}")
        x = self.movement_encoder(motion)
        if lengths is None and valid_mask is not None:
            lengths = valid_mask.long().sum(dim=1)
        lengths = self.movement_encoder.output_lengths(lengths)
        batch_size, frames = x.shape[:2]
        x = self.to_hidden(x)
        if valid_mask is None:
            mask = lengths_to_mask(lengths, frames, x.device)
            if mask.shape[0] == 1 and batch_size != 1:
                mask = mask.expand(batch_size, -1)
        else:
            if self.movement_encoder.downsample_factor > 1:
                mask_lengths = self.movement_encoder.output_lengths(valid_mask.long().sum(dim=1))
                mask = lengths_to_mask(mask_lengths, frames, x.device)
            else:
                mask = valid_mask.to(device=x.device, dtype=torch.bool)
        x = self.temporal_encoder(x, valid_mask=mask, lengths=lengths)
        weights = mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1) if self.normalize else z
