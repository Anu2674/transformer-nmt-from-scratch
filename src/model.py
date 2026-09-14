"""Phase 1: Transformer encoder-decoder implemented from scratch.

Only torch primitives are used (nn.Linear, nn.LayerNorm, nn.Dropout, nn.Embedding,
tensor ops). No nn.Transformer / nn.MultiheadAttention / nn.TransformerEncoderLayer.

Deviation from Vaswani et al. 2017 that is deliberate and documented:
  * pre-norm residual blocks (norm_first=True) instead of the paper's post-norm.
    Post-norm needs a carefully tuned warmup to stay stable on a single small
    GPU; pre-norm trains reliably at this scale and is what most modern
    implementations use. Everything else (sinusoidal PE, scaled dot-product
    multi-head attention, position-wise FFN, residuals, weight tying) follows
    the paper.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from masking import make_pad_mask, make_decoder_self_mask


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017, sec 3.5)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with h heads.

    mask: broadcastable to (B, h, Tq, Tk), True = keep, False = block.
    Stores the last attention probabilities in ``self.attn`` for visualisation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn: torch.Tensor | None = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)  # (B,h,T,d_k)

    def forward(self, query, key, value, mask=None, store_attn: bool = False):
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        # rows that were fully masked (e.g. a padded query) -> softmax gives nan; zero them
        probs = torch.nan_to_num(probs)
        if store_attn:
            self.attn = probs.detach()
        probs = self.dropout(probs)

        out = torch.matmul(probs, v)                       # (B,h,T,d_k)
        b, _, t, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, store_attn=False):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, src_mask, store_attn))
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask, store_attn=False):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask, store_attn))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, src_mask, store_attn))
        h = self.norm3(x)
        x = x + self.dropout(self.ff(h))
        return x


class Encoder(nn.Module):
    def __init__(self, emb, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        super().__init__()
        self.emb = emb
        self.scale = math.sqrt(d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask, store_attn=False):
        x = self.pos(self.emb(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask, store_attn)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, emb, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        super().__init__()
        self.emb = emb
        self.scale = math.sqrt(d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask, src_mask, store_attn=False):
        x = self.pos(self.emb(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask, store_attn)
        return self.norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.config = dict(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, max_len=max_len, pad_id=pad_id,
            tie_embeddings=tie_embeddings,
        )

        shared_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.encoder = Encoder(shared_emb, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(shared_emb, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.generator.weight = shared_emb.weight

        self._init_params()

    def _init_params(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name and "generator" not in name:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.encoder.emb.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.encoder.emb.weight[self.pad_id].zero_()

    # -- pieces, used by search / visualisation --
    def encode(self, src, store_attn=False):
        src_mask = make_pad_mask(src, self.pad_id)
        return self.encoder(src, src_mask, store_attn), src_mask

    def decode(self, tgt, memory, src_mask, store_attn=False):
        tgt_mask = make_decoder_self_mask(tgt, self.pad_id)
        return self.decoder(tgt, memory, tgt_mask, src_mask, store_attn)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        dec = self.decode(tgt_in, memory, src_mask)
        return self.generator(dec)          # (B, T, vocab)

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
