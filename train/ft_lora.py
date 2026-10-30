"""A small, self-contained LoRA for the Qwen3-TTS talker.

Why hand-rolled instead of peft: the official checkpoint format is a single flat
``model.safetensors`` whose keys must match the stock module tree exactly, and the
fine-tuning forward pass reaches into the model by attribute name
(``model.talker.model.text_embedding`` and friends). A wrapper that renames or
re-parents modules breaks both. Ninety lines here keep the tree untouched, keep the
adapter in fp32 over a frozen bf16 base, and make "merge into the official state
dict" a two-line operation.

Why LoRA at all on a 16 GB card: a full fine-tune of the 1.7B talker in pure bf16
needs ~15 GB for weights + grads + AdamW moments alone, and - worse - a bf16 weight
cannot represent an update smaller than ~0.4% of its own magnitude, so Adam steps at
the learning rates this recipe wants are silently rounded away. Freezing the base and
training an fp32 adapter dodges both problems.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Every projection the talker stack owns, including the text->talker resize MLP and
#: the talker->code-predictor projection. Embeddings and output heads are left alone.
DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "linear_fc1", "linear_fc2",
    "small_to_mtp_projection",
)

#: Only inside the talker. `speaker_encoder` is frozen by policy (see issue #351).
DEFAULT_SCOPE = r"^talker\."


class LoRALinear(nn.Module):
    """``base(x) + B @ A @ x * (alpha / r)`` with the adapter held in fp32."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.dropout(x).to(torch.float32)
        h = F.linear(F.linear(h, self.lora_A), self.lora_B) * self.scaling
        return out + h.to(out.dtype)

    @torch.no_grad()
    def delta(self) -> torch.Tensor:
        return (self.lora_B.float() @ self.lora_A.float()) * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:g}"


def inject(model: nn.Module, r: int = 32, alpha: float = 64.0, dropout: float = 0.05,
           targets: tuple[str, ...] = DEFAULT_TARGETS,
           scope: str = DEFAULT_SCOPE) -> dict:
    """Replace matching ``nn.Linear`` leaves in place. Returns a summary dict."""
    scope_re = re.compile(scope)
    hits: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear) or child_name not in targets:
                continue
            if not scope_re.search(full):
                continue
            setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
            hits.append(full)
    if not hits:
        raise RuntimeError(f"LoRA matched nothing for targets={targets} scope={scope!r}")
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"r": r, "alpha": alpha, "dropout": dropout, "targets": list(targets),
            "scope": scope, "modules_wrapped": len(hits), "trainable_params": n_lora,
            "example": hits[:3]}


def freeze_base(model: nn.Module) -> None:
    """Everything except the adapters stops learning."""
    for name, p in model.named_parameters():
        p.requires_grad_(name.endswith(".lora_A") or name.endswith(".lora_B"))


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if k.endswith(".lora_A") or k.endswith(".lora_B")}


def load_adapter(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing = model.load_state_dict(state, strict=False)
    unexpected = [k for k in missing.unexpected_keys]
    if unexpected:
        raise RuntimeError(f"adapter file has keys this model does not: {unexpected[:5]}")


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The official flat state dict, with every adapter folded into its base weight.

    Keys come back exactly as the stock model tree names them, so the result drops
    straight into ``model.safetensors`` next to an unmodified ``config.json``.
    """
    lora = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    out: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        if ".base." in key:
            owner, suffix = key.rsplit(".base.", 1)
            if owner in lora:
                key = f"{owner}.{suffix}"
                if suffix == "weight":
                    value = (value.float() + lora[owner].delta().to(value.device)).to(value.dtype)
        out[key] = value.detach().to("cpu")
    return out
