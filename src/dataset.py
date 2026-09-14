"""Data pipeline: SentencePiece-encoded Multi30k with length-bucketed batches."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def _read(split: str, lang: str) -> list[str]:
    return [ln.rstrip("\r\n") for ln in (RAW / f"{split}.{lang}").open(encoding="utf-8")]


class Multi30k(Dataset):
    """src = English, tgt = German.  Returns (src_ids, tgt_ids) with bos/eos."""

    def __init__(self, split: str, tok: Tokenizer, max_len: int = 100):
        self.tok = tok
        src_lines, tgt_lines = _read(split, "en"), _read(split, "de")
        self.src_text, self.tgt_text = src_lines, tgt_lines
        self.pairs: list[tuple[list[int], list[int]]] = []
        self.kept_idx: list[int] = []
        for i, (s, t) in enumerate(zip(src_lines, tgt_lines)):
            si = tok.encode(s)
            ti = tok.encode(t)
            if len(si) <= max_len and len(ti) <= max_len:
                self.pairs.append((si, ti))
                self.kept_idx.append(i)
        self.dropped = len(src_lines) - len(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        s, t = self.pairs[i]
        return torch.tensor(s), torch.tensor(t)


def collate(batch, pad_id: int = 0):
    srcs, tgts = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return src, tgt


class BucketBatchSampler(torch.utils.data.Sampler):
    """Yield lists of indices whose sources have similar length, to cut padding.

    Sorts by source length inside randomly shuffled mega-batches, slices into
    batches, then shuffles batch order. Fresh randomness every epoch.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True,
                 mega: int = 50, seed: int = 0):
        self.lengths = lengths
        self.bs = batch_size
        self.shuffle = shuffle
        self.mega = mega
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        chunk = self.bs * self.mega
        for c in range(0, len(idx), chunk):
            block = sorted(idx[c:c + chunk], key=lambda i: self.lengths[i])
            for b in range(0, len(block), self.bs):
                batches.append(block[b:b + self.bs])
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


def make_loader(ds: Multi30k, batch_size: int, shuffle: bool, seed: int = 0):
    sampler = BucketBatchSampler(
        [len(s) for s, _ in ds.pairs], batch_size, shuffle=shuffle, seed=seed
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, collate_fn=lambda b: collate(b, ds.tok.pad_id)
    )
    return loader, sampler
