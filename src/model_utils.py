"""Shared transformer helpers: dependency checks, seeding, PEFT (LoRA / rsLoRA / QLoRA) injection,
optimiser and scheduler construction.

Every heavy import is deferred so the scikit-learn baseline keeps working when the
`deeplearning` extra is not installed.
"""

import importlib.util
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import yaml

PEFT_METHODS = ("full", "lora", "rslora", "qlora")


def deep_learning_available() -> bool:
    """True when torch, transformers and peft can all be imported."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft"))


def require_deep_learning() -> None:
    if not deep_learning_available():
        raise SystemExit(
            "The transformer pipeline needs the deeplearning extra.\n"
            "Install it with:  uv sync --extra deeplearning"
        )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    import torch
    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def bitsandbytes_available() -> bool:
    if importlib.util.find_spec("bitsandbytes") is None:
        return False
    import torch
    return torch.cuda.is_available()


# Task heads stay in full precision: they are trained in full and must not be 4-bit quantised
TASK_HEAD_MODULES = ["classifier", "pre_classifier", "score", "qa_outputs"]


def quantization_config(quant_type: str = "nf4", double_quant: bool = True):
    """4-bit NormalFloat quantisation with double quantisation and bfloat16 compute (QLoRA)."""
    import torch
    from transformers import BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=TASK_HEAD_MODULES,
    )


def lora_scaling(alpha: float, r: int, rank_stabilized: bool) -> float:
    """Adapter scaling factor: alpha / r for standard LoRA, alpha / sqrt(r) for rsLoRA."""
    return alpha / (r ** 0.5) if rank_stabilized else alpha / r


def build_lora_config(lora_cfg: Dict[str, Any], task_type: str, rank_stabilized: bool = False,
                      modules_to_save: Optional[list] = None):
    from peft import LoraConfig
    return LoraConfig(
        task_type=task_type,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["query", "value"])),
        use_rslora=rank_stabilized,
        modules_to_save=modules_to_save,
    )


def load_peft_model(model_cls, base_model: str, method: str, lora_cfg: Dict[str, Any],
                    task_type: str, device, quant_cfg: Optional[Dict[str, Any]] = None,
                    **model_kwargs):
    """Loads `model_cls.from_pretrained(base_model)` and applies the requested fine-tuning method.

    method: full   -> every weight trainable
            lora   -> low-rank adapters, scaling alpha / r
            rslora -> low-rank adapters, scaling alpha / sqrt(r)
            qlora  -> 4-bit NF4 frozen base + LoRA adapters (falls back to LoRA without CUDA)
    """
    if method not in PEFT_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {PEFT_METHODS}")

    if method == "qlora" and not bitsandbytes_available():
        print("   [warn] bitsandbytes/CUDA unavailable - falling back from QLoRA to LoRA")
        method = "lora"

    if method == "qlora":
        from peft import prepare_model_for_kbit_training
        quant_cfg = quant_cfg or {}
        model = model_cls.from_pretrained(
            base_model,
            quantization_config=quantization_config(
                quant_cfg.get("quant_type", "nf4"), quant_cfg.get("use_double_quant", True)
            ),
            device_map={"": 0},
            **model_kwargs,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = model_cls.from_pretrained(base_model, **model_kwargs).to(device)

    if method == "full":
        return model, method

    from peft import get_peft_model
    peft_cfg = build_lora_config(lora_cfg, task_type, rank_stabilized=(method == "rslora"))
    model = get_peft_model(model, peft_cfg)
    return model, method


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_optimizer_and_scheduler(model, learning_rate: float, weight_decay: float,
                                  total_steps: int, warmup_ratio: float = 0.1,
                                  scheduler: str = "cosine"):
    """AdamW (betas 0.9/0.999, eps 1e-8) with decoupled weight decay and a warmup schedule."""
    import torch
    from transformers import get_scheduler

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    params = [p for p in model.named_parameters() if p[1].requires_grad]
    groups = [
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = get_scheduler(
        scheduler, optimizer=optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=max(total_steps, 1),
    )
    return optimizer, lr_scheduler


class EarlyStopping:
    """Stops training when the monitored metric has not improved for `patience` evaluations."""

    def __init__(self, patience: int = 2, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.bad_rounds = 0

    def step(self, value: float) -> bool:
        """Records a new value; returns True if it is the best so far."""
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.bad_rounds = 0
            return True
        self.bad_rounds += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_rounds >= self.patience
