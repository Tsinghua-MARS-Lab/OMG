from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 512,
        model_name: str = "t5-base",
        max_length: int = 100,
        freeze_encoder: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise ImportError("TextEncoder requires transformers and sentencepiece.") from exc
        requested_model_name = str(model_name)
        models_root = Path(os.environ.get("OMG_MODELS_ROOT", "models"))
        local_t5_3b_path = os.environ.get("OMG_T5_3B_MODEL", str(models_root / "t5-3b-local"))
        if requested_model_name == "t5-3b" and Path(local_t5_3b_path).exists():
            self.model_name = local_t5_3b_path
            local_files_only = True
        else:
            self.model_name = requested_model_name
            local_files_only = Path(self.model_name).exists()

        self.output_dim = int(output_dim)
        self.max_length = int(max_length)
        self.freeze_encoder = bool(freeze_encoder)
        self.normalize = bool(normalize)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, local_files_only=local_files_only)
        self.encoder = T5EncoderModel.from_pretrained(self.model_name, local_files_only=local_files_only)
        if self.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        self.t5_dim = int(self.encoder.config.d_model)
        self.proj = nn.Sequential(nn.Linear(self.t5_dim, self.output_dim), nn.LayerNorm(self.output_dim))

    def tokenize(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value.to(device) for key, value in tokens.items()}

    @torch.no_grad()
    def raw_encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenize(texts, device)
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_raw(self, raw_embeddings: torch.Tensor) -> torch.Tensor:
        z = self.proj(raw_embeddings)
        return F.normalize(z, dim=-1) if self.normalize else z

    def forward(self, captions: list[str], device: torch.device) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze_encoder):
            raw = self.raw_encode([str(caption) for caption in captions], device=device)
        return self.encode_raw(raw)
