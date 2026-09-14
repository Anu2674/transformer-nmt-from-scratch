"""Training utilities: label smoothing, Noam LR schedule, seeding, logging."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LabelSmoothingLoss(nn.Module):
    """KL-div to a smoothed target distribution (Vaswani et al. sec 5.4).

    confidence on the gold token = 1 - smoothing; the rest is spread over the
    remaining vocab. <pad> positions are ignored.
    """

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, V) log-space after log_softmax ; target: (N,)
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(logp, self.smoothing / (self.vocab - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_id] = 0.0
            mask = target == self.pad_id
            true_dist[mask] = 0.0
        n_tok = (~mask).sum().clamp(min=1)
        return torch.sum(-true_dist * logp) / n_tok


class NoamLR:
    """lr = factor * d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)."""

    def __init__(self, optimizer, d_model: int, warmup: int, factor: float = 1.0):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor
        self.step_num = 0
        self._last = 0.0

    def step(self) -> None:
        self.step_num += 1
        s = self.step_num
        lr = self.factor * self.d_model ** -0.5 * min(s ** -0.5, s * self.warmup ** -1.5)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self._last = lr

    @property
    def lr(self) -> float:
        return self._last

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, sd):
        self.step_num = sd["step_num"]


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.t0 = time.time()

    def log(self, **kw) -> None:
        kw["wall_s"] = round(time.time() - self.t0, 1)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")
