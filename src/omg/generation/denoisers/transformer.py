from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from omg.generation.denoisers.interface import BaseMotionDenoiser


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = timesteps.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = -x_odd
    out[..., 1::2] = x_even
    return out


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device=device))
        cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)
        return cos.to(dtype=dtype).unsqueeze(0).unsqueeze(0), sin.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.forward(q.shape[-2], q.device, q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class RotarySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even attention head dim, got {self.head_dim}")
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope.apply(q, k)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                batch_size,
                1,
                1,
                seq_len,
                device=x.device,
                dtype=x.dtype,
            )
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :].bool(),
                torch.finfo(x.dtype).min,
            )

        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        h = h.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        h = self.out_proj(h)
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)
        return h


class LocalTemporalCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window: int = 8,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even attention head dim, got {self.head_dim}")
        self.window = int(window)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.kv_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)
        self.dropout = float(dropout)

    def _local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.window < 0:
            return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        positions = torch.arange(seq_len, device=device)
        return (positions[:, None] - positions[None, :]).abs() <= self.window

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        query_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.shape != x.shape:
            raise ValueError(f"Expected local attention context shape {tuple(x.shape)}, got {tuple(context.shape)}")
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).view(batch_size, seq_len, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope.apply(q, k)

        allowed = self._local_mask(seq_len, x.device)[None, None, :, :]
        if context_mask is not None:
            allowed = allowed & context_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        has_key = allowed.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(allowed)
        fallback[..., 0] = True
        allowed = torch.where(has_key, allowed, fallback)

        attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(~allowed, torch.finfo(x.dtype).min)
        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        h = h.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        h = self.out_proj(h)
        h = h * has_key.squeeze(1).to(dtype=h.dtype)
        if query_padding_mask is not None:
            h = h.masked_fill(query_padding_mask.unsqueeze(-1).bool(), 0.0)
        return h


def _qk_normalized_cross_attention(
    attention: nn.Module,
    query: torch.Tensor,
    context: torch.Tensor,
    context_key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Cross-attend with per-head query/key directions instead of magnitudes.

    QK normalization removes the otherwise unidentifiable radial degree of
    freedom in query and key projections.  Scaling either projection therefore
    cannot sharpen the logits or amplify the input Jacobian.  Multiplying unit
    vectors by ``sqrt(head_dim)`` preserves the variance of standard scaled
    dot-product attention at initialization.
    """
    if hasattr(attention, "_qkv_same_embed_dim") and not attention._qkv_same_embed_dim:
        raise ValueError("QK-normalized cross-attention requires equal query/key/value dimensions")

    embed_dim = attention.embed_dim
    num_heads = attention.num_heads
    head_dim = embed_dim // num_heads
    if hasattr(attention, "q_proj"):
        q = attention.q_proj(query)
        k = attention.k_proj(context)
        v = attention.v_proj(context)
    else:
        if attention.in_proj_weight is None:
            raise ValueError("QK-normalized cross-attention requires packed projection weights")
        weight_q, weight_k, weight_v = attention.in_proj_weight.chunk(3, dim=0)
        if attention.in_proj_bias is None:
            bias_q = bias_k = bias_v = None
        else:
            bias_q, bias_k, bias_v = attention.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, weight_q, bias_q)
        k = F.linear(context, weight_k, bias_k)
        v = F.linear(context, weight_v, bias_v)
    batch_size, query_len, _ = q.shape
    context_len = k.shape[1]
    q = q.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, context_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, context_len, num_heads, head_dim).transpose(1, 2)

    head_scale = math.sqrt(head_dim)
    q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype) * head_scale
    k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype) * head_scale

    attn_mask = None
    if context_key_padding_mask is not None:
        attn_mask = torch.zeros(
            batch_size,
            1,
            1,
            context_len,
            device=q.device,
            dtype=q.dtype,
        )
        attn_mask = attn_mask.masked_fill(
            context_key_padding_mask[:, None, None, :].bool(),
            torch.finfo(q.dtype).min,
        )
    h = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=float(getattr(attention, "dropout", 0.0)) if attention.training else 0.0,
        is_causal=False,
    )
    h = h.transpose(1, 2).reshape(batch_size, query_len, embed_dim)
    return attention.out_proj(h)


class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RotarySelfAttention(hidden_dim, num_heads, dropout=dropout, rope_base=rope_base)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)
        return x


class RotaryTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RotaryTransformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


def _expand_timesteps(timesteps: torch.Tensor, seq_len: int) -> torch.Tensor:
    if timesteps.ndim == 1:
        return timesteps[:, None].expand(-1, seq_len)
    if timesteps.ndim == 2:
        return timesteps
    raise ValueError(f"Expected timestep shape (B,) or (B,T), got {tuple(timesteps.shape)}")


def _zero_init_last_linear(module: nn.Sequential) -> None:
    for layer in reversed(module):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class MotionTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        use_frame_film: bool = True,
        use_control_local_attn: bool = False,
        audio_local_attn_window: int = 8,
        human_motion_local_attn_window: int = 2,
        use_human_motion_local_attn: bool = False,
    ):
        super().__init__()
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = RotarySelfAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        ff_dim = int(round(hidden_dim * float(mlp_ratio)))
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.audio_film = None
        self.human_motion_film = None
        if bool(use_frame_film):
            self.audio_film = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2))
            self.human_motion_film = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2))
            _zero_init_last_linear(self.audio_film)
            _zero_init_last_linear(self.human_motion_film)
        self.norm_audio_local_attn = None
        self.audio_local_attn = None
        self.audio_local_attn_gate = None
        self.human_motion_control = None
        self.human_motion_control_gate = None
        self.norm_human_motion_local_attn = None
        self.human_motion_local_attn = None
        self.human_motion_local_attn_gate = None
        if bool(use_control_local_attn):
            self.norm_audio_local_attn = nn.LayerNorm(hidden_dim)
            self.audio_local_attn = LocalTemporalCrossAttention(
                hidden_dim,
                num_heads,
                window=audio_local_attn_window,
                dropout=dropout,
                rope_base=rope_base,
            )
            self.audio_local_attn_gate = nn.Parameter(torch.zeros(()))
            self.human_motion_control = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.human_motion_control_gate = nn.Parameter(torch.zeros(()))
            if bool(use_human_motion_local_attn):
                self.norm_human_motion_local_attn = nn.LayerNorm(hidden_dim)
                self.human_motion_local_attn = LocalTemporalCrossAttention(
                    hidden_dim,
                    num_heads,
                    window=human_motion_local_attn_window,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                self.human_motion_local_attn_gate = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def _apply_frame_film(
        self,
        x: torch.Tensor,
        *,
        audio_cond: torch.Tensor | None,
        audio_mask: torch.Tensor | None,
        human_motion_cond: torch.Tensor | None,
        human_motion_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scale = None
        shift = None
        for cond, mask, film in (
            (audio_cond, audio_mask, self.audio_film),
            (human_motion_cond, human_motion_mask, self.human_motion_film),
        ):
            if cond is None or film is None:
                continue
            film_out = film(cond)
            half_dim = film_out.shape[-1] // 2
            curr_scale = film_out[..., :half_dim]
            curr_shift = film_out[..., half_dim:]
            if mask is not None:
                mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
                curr_scale = curr_scale * mask
                curr_shift = curr_shift * mask
            scale = curr_scale if scale is None else scale + curr_scale
            shift = curr_shift if shift is None else shift + curr_shift
        if scale is None or shift is None:
            return x
        return x * (1.0 + torch.tanh(scale)) + shift

    def _apply_control_local_attn(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        audio_cond: torch.Tensor | None,
        audio_mask: torch.Tensor | None,
        human_motion_cond: torch.Tensor | None,
        human_motion_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if audio_cond is not None and self.audio_local_attn is not None and self.audio_local_attn_gate is not None:
            assert self.norm_audio_local_attn is not None
            audio = self.audio_local_attn(
                self.norm_audio_local_attn(x),
                audio_cond,
                context_mask=audio_mask,
                query_padding_mask=key_padding_mask,
            )
            x = x + self.dropout(self.audio_local_attn_gate.to(dtype=x.dtype) * audio)
        if (
            human_motion_cond is not None
            and self.human_motion_control is not None
            and self.human_motion_control_gate is not None
        ):
            human_motion = self.human_motion_control(human_motion_cond)
            if human_motion_mask is not None:
                human_motion = human_motion * human_motion_mask.to(device=x.device, dtype=human_motion.dtype).unsqueeze(-1)
            x = x + self.dropout(self.human_motion_control_gate.to(dtype=x.dtype) * human_motion)
        if (
            human_motion_cond is not None
            and self.human_motion_local_attn is not None
            and self.human_motion_local_attn_gate is not None
        ):
            assert self.norm_human_motion_local_attn is not None
            human_motion = self.human_motion_local_attn(
                self.norm_human_motion_local_attn(x),
                human_motion_cond,
                context_mask=human_motion_mask,
                query_padding_mask=key_padding_mask,
            )
            x = x + self.dropout(self.human_motion_local_attn_gate.to(dtype=x.dtype) * human_motion)
        return x


    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        audio_cond: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        human_motion_cond: torch.Tensor | None = None,
        human_motion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_key_padding = ~context_mask.bool()
        h = self.norm_cross(x)
        cross = _qk_normalized_cross_attention(
            self.cross_attn,
            query=h,
            context=context,
            context_key_padding_mask=context_key_padding,
        )
        x = x + self.dropout(cross)
        h = self.norm_self(x)
        x = x + self.dropout(self.self_attn(h, key_padding_mask=key_padding_mask))
        x = self._apply_control_local_attn(
            x,
            key_padding_mask=key_padding_mask,
            audio_cond=audio_cond,
            audio_mask=audio_mask,
            human_motion_cond=human_motion_cond,
            human_motion_mask=human_motion_mask,
        )
        h = self.norm_ff(x)
        h = self._apply_frame_film(
            h,
            audio_cond=audio_cond,
            audio_mask=audio_mask,
            human_motion_cond=human_motion_cond,
            human_motion_mask=human_motion_mask,
        )
        x = x + self.dropout(self.ff(h))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)
        return x


class MotionTransformerDenoiser(BaseMotionDenoiser):
    _FRAME_COND_INJECTION_MODES = {
        "sum_to_time",
        "separate_to_h",
        "per_layer_film",
        "control_local_attn",
    }

    def __init__(
        self,
        input_dim: int = 123,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        text_dim: int = 768,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        frame_cond_injection: str = "per_layer_film",
        audio_local_attn_window: int = 8,
        human_motion_local_attn_window: int = 2,
        use_human_motion_local_attn: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.dropout = float(dropout)
        self.set_frame_cond_injection(frame_cond_injection)
        self.audio_local_attn_window = int(audio_local_attn_window)
        self.human_motion_local_attn_window = int(human_motion_local_attn_window)
        self.use_human_motion_local_attn = bool(use_human_motion_local_attn)
        self.time_embed = TimestepEmbedding(self.hidden_dim)
        self.input_proj = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.text_proj = nn.Linear(int(text_dim), self.hidden_dim)
        self.extra_proj = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers = nn.ModuleList(
            [
                MotionTransformerBlock(
                    self.hidden_dim,
                    int(num_heads),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    rope_base=rope_base,
                    use_frame_film=self.frame_cond_injection == "per_layer_film",
                    use_control_local_attn=self.frame_cond_injection == "control_local_attn",
                    audio_local_attn_window=self.audio_local_attn_window,
                    human_motion_local_attn_window=self.human_motion_local_attn_window,
                    use_human_motion_local_attn=self.use_human_motion_local_attn,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.input_dim)
        if self.frame_cond_injection == "separate_to_h":
            self.audio_adapter = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.human_motion_adapter = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.audio_gate = nn.Parameter(torch.zeros(()))
            self.human_motion_gate = nn.Parameter(torch.zeros(()))

    def set_frame_cond_injection(self, mode: str) -> None:
        mode = str(mode)
        if mode not in self._FRAME_COND_INJECTION_MODES:
            choices = ", ".join(sorted(self._FRAME_COND_INJECTION_MODES))
            raise ValueError(f"Unsupported frame_cond_injection={mode!r}; expected one of: {choices}")
        self.frame_cond_injection = mode

    def _validate_frame_condition_modules(self) -> None:
        if self.frame_cond_injection == "per_layer_film":
            required = ("audio_film", "human_motion_film")
        elif self.frame_cond_injection == "control_local_attn":
            required = ("audio_local_attn", "human_motion_control")
        elif self.frame_cond_injection == "separate_to_h":
            names = [
                "audio_adapter",
                "human_motion_adapter",
                "audio_gate",
                "human_motion_gate",
            ]
            missing = [name for name in names if not hasattr(self, name)]
            if missing:
                raise RuntimeError(
                    "MotionTransformerDenoiser is missing modules for "
                    f"frame_cond_injection={self.frame_cond_injection!r}: {missing}. "
                    "Instantiate the denoiser with the same frame_cond_injection used by MotionGenerator/config."
                )
            return
        else:
            return
        for layer in self.layers:
            for module_name in required:
                if getattr(layer, module_name, None) is None:
                    raise RuntimeError(
                        "MotionTransformerDenoiser was constructed without modules for "
                        f"frame_cond_injection={self.frame_cond_injection!r}. Instantiate the denoiser with "
                        "the same frame_cond_injection used by MotionGenerator/config."
                    )

    def _context(self, conditions: dict, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        text_context = conditions.get("text_context")
        text_mask = conditions.get("text_mask")
        extra_tokens = conditions.get("extra_tokens")
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        if extra_tokens is not None:
            extra_tokens = self.extra_proj(extra_tokens.to(device=device, dtype=dtype))
            pieces.append(extra_tokens)
            masks.append(torch.ones(extra_tokens.shape[:2], dtype=torch.bool, device=device))

        if text_context is not None and text_mask is not None:
            text_context = self.text_proj(text_context.to(device=device, dtype=dtype))
            text_mask = text_mask.to(device=device, dtype=torch.bool)
            pieces.append(text_context)
            masks.append(text_mask)

        if not pieces:
            token = torch.zeros(batch_size, 1, self.hidden_dim, device=device, dtype=dtype)
            return token, torch.ones(batch_size, 1, device=device, dtype=torch.bool)

        context = torch.cat(pieces, dim=1)
        context_mask = torch.cat(masks, dim=1)
        has_context = context_mask.any(dim=1, keepdim=True)
        first_token_mask = torch.zeros_like(context_mask)
        first_token_mask[:, :1] = True
        context_mask = context_mask | (~has_context & first_token_mask)
        return context, context_mask

    def _frame_condition(self, cond_tokens: dict, key: str, shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        value = cond_tokens.get(key)
        if value is None:
            return None
        value = value.to(device=device, dtype=dtype)
        if value.shape != shape:
            raise ValueError(f"Expected {key} shape {tuple(shape)}, got {tuple(value.shape)}")
        return value

    def _frame_condition_mask(self, cond_tokens: dict, key: str, shape: torch.Size, device: torch.device) -> torch.Tensor | None:
        value = cond_tokens.get(key)
        if value is None:
            return None
        value = value.to(device=device, dtype=torch.bool)
        if value.shape != shape:
            raise ValueError(f"Expected {key} shape {tuple(shape)}, got {tuple(value.shape)}")
        return value


    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond_tokens: dict,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_frame_condition_modules()
        batch_size, seq_len, _ = x.shape
        t = _expand_timesteps(timesteps.to(device=x.device), seq_len)
        time = self.time_embed(t).to(dtype=x.dtype)
        if self.frame_cond_injection == "sum_to_time":
            frame_cond = self._frame_condition(cond_tokens, "frame_cond", time.shape, x.dtype, x.device)
            if frame_cond is not None:
                time = time + frame_cond
            h = self.input_proj(torch.cat([x, time], dim=-1))
        elif self.frame_cond_injection == "separate_to_h":
            h = self.input_proj(torch.cat([x, time], dim=-1))
            audio_cond = self._frame_condition(cond_tokens, "audio_cond", h.shape, h.dtype, x.device)
            if audio_cond is not None:
                h = h + self.audio_gate.to(dtype=h.dtype) * self.audio_adapter(audio_cond)
            human_motion_cond = self._frame_condition(cond_tokens, "human_motion_cond", h.shape, h.dtype, x.device)
            if human_motion_cond is not None:
                h = h + self.human_motion_gate.to(dtype=h.dtype) * self.human_motion_adapter(human_motion_cond)
        elif self.frame_cond_injection in {"per_layer_film", "control_local_attn"}:
            h = self.input_proj(torch.cat([x, time], dim=-1))
        else:
            raise RuntimeError(f"Unhandled frame_cond_injection={self.frame_cond_injection!r}")
        audio_cond = audio_mask = human_motion_cond = human_motion_mask = None
        if self.frame_cond_injection in {"per_layer_film", "control_local_attn"}:
            audio_cond = self._frame_condition(cond_tokens, "audio_cond", h.shape, h.dtype, x.device)
            audio_mask = self._frame_condition_mask(cond_tokens, "audio_mask", h.shape[:2], x.device)
            human_motion_cond = self._frame_condition(cond_tokens, "human_motion_cond", h.shape, h.dtype, x.device)
            human_motion_mask = self._frame_condition_mask(cond_tokens, "human_motion_mask", h.shape[:2], x.device)
            if audio_cond is not None and audio_mask is None:
                raise ValueError(
                    f"audio_mask is required when audio_cond is provided for {self.frame_cond_injection} conditioning"
                )
            if human_motion_cond is not None and human_motion_mask is None:
                raise ValueError(
                    "human_motion_mask is required when human_motion_cond is provided for "
                    f"{self.frame_cond_injection} conditioning"
                )
        context, context_mask = self._context(cond_tokens, batch_size, x.device, h.dtype)
        pad_mask = None if valid_mask is None else ~valid_mask.to(device=x.device, dtype=torch.bool)
        for layer in self.layers:
            h = layer(
                h,
                context=context,
                context_mask=context_mask,
                key_padding_mask=pad_mask,
                audio_cond=audio_cond,
                audio_mask=audio_mask,
                human_motion_cond=human_motion_cond,
                human_motion_mask=human_motion_mask,
            )
        h = self.norm(h)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return self.output(h)
