from typing import List, Optional

import torch
import torch.nn as nn

from transformers import Qwen3Model
from peft import LoraConfig, get_peft_model

DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class DecoderBase(nn.Module):
    """
    Base for VLM decoder components. Consumes (B, L, embed_dim), returns (B, L, embed_dim).
    Subclasses implement the actual causal transformer (LLaMA, Qwen, minimal, etc.).
    """

    @property
    def embed_dim(self) -> int:
        """Output (and typically input) embedding dimension."""
        raise NotImplementedError

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, seq_len, embed_dim)
        Returns: (B, seq_len, embed_dim) hidden states.
        """
        raise NotImplementedError


class QwenDecoder(DecoderBase):
    """
    Decoder backed by HuggingFace Qwen3. Projects pipeline embed_dim to/from
    Qwen hidden_size. Optional: use_lora=True freezes the base and trains LoRA adapters only.
    Requires: transformers, qwen3; for LoRA: peft.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
        embed_dim: int = 256,
        freeze: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self.model = Qwen3Model.from_pretrained(model_name_or_path)
        hidden_size = self.model.config.hidden_size
        self.input_proj = nn.Linear(embed_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, embed_dim)

        if use_lora:

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or DEFAULT_LORA_TARGET_MODULES,
                lora_dropout=0.05,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()
        elif freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, seq_len, embed_dim)
        Returns: (B, seq_len, embed_dim)
        """
        x = self.input_proj(embeddings)
        out = self.model(inputs_embeds=x).last_hidden_state
        return self.output_proj(out)
