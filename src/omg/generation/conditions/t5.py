from __future__ import annotations

import torch
import torch.nn as nn


class FrozenT5TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "t5-base",
        max_length: int = 50,
        output_dim: int = 768,
    ):
        super().__init__()
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise ImportError(
                "FrozenT5TextEncoder requires the `transformers` package. "
                "Install OMG with the training extras."
            ) from exc

        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.encoder = T5EncoderModel.from_pretrained(self.model_name)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        hidden_dim = int(self.encoder.config.d_model)
        self.output_dim = int(output_dim)
        if hidden_dim == self.output_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(hidden_dim, self.output_dim)

    def forward(
        self,
        captions: list[str],
        has_text: torch.Tensor | None = None,
        force_null_text: bool = False,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        if force_null_text:
            captions = ["" for _ in captions]
        if len(captions) == 0:
            raise ValueError("FrozenT5TextEncoder received an empty caption batch")

        tokenized = self.tokenizer(
            captions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if device is None:
            device = next(self.parameters()).device
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            hidden = self.encoder(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            ).last_hidden_state
        hidden = self.proj(hidden)
        mask = tokenized["attention_mask"].bool()
        if has_text is not None:
            has_text = has_text.to(device=device, dtype=torch.bool).view(-1, 1)
            mask = mask & has_text
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return {"context": hidden, "mask": mask}
