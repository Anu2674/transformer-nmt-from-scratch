"""Attention masks.

Convention used everywhere in this project:
    a mask value of  True  means "this key position is allowed / visible"
    a mask value of  False means "block this position" (filled with -inf pre-softmax)

Shapes are always broadcastable to (batch, n_heads, q_len, k_len).
"""
from __future__ import annotations

import torch


def make_pad_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """(B, T) token ids -> (B, 1, 1, T) bool, True where token is NOT padding."""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """-> (1, 1, size, size) bool lower-triangular (query t may see keys 0..t)."""
    full = torch.ones(size, size, dtype=torch.bool, device=device)
    return torch.tril(full).unsqueeze(0).unsqueeze(0)


def make_decoder_self_mask(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Combine target padding mask with the causal mask -> (B, 1, T, T) bool."""
    pad = make_pad_mask(tgt, pad_id)                       # (B,1,1,T)
    causal = make_causal_mask(tgt.size(1), tgt.device)     # (1,1,T,T)
    return pad & causal
