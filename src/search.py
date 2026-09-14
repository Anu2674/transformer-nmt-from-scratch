"""Inference-time decoding for the Transformer: greedy and beam search."""
from __future__ import annotations

import torch

from masking import make_pad_mask


@torch.no_grad()
def rnn_greedy_decode(model, src, max_len: int = 100, bos_id: int = 2, eos_id: int = 3,
                      pad_id: int = 0) -> list[list[int]]:
    """Greedy decoding for the GRU+attention baseline (step-by-step GRU, no
    encode/decode split like the Transformer -- the recurrence is inherent)."""
    model.eval()
    device = src.device
    b = src.size(0)
    src_len = (src != pad_id).sum(1)
    mask = src != pad_id
    enc_outputs, hidden = model.encoder(src, src_len)
    tok = torch.full((b,), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    cols = []
    for _ in range(max_len - 1):
        pred, hidden, _ = model.decoder(tok, hidden, enc_outputs, mask)
        nxt = pred.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        cols.append(nxt)
        done = done | (nxt == eos_id)
        tok = nxt
        if done.all():
            break
    if not cols:
        return [[] for _ in range(b)]
    mat = torch.stack(cols, dim=1).tolist()
    out = []
    for row in mat:
        if eos_id in row:
            row = row[: row.index(eos_id)]
        out.append(row)
    return out


@torch.no_grad()
def greedy_decode(model, src, max_len: int = 100, bos_id: int = 2, eos_id: int = 3,
                  pad_id: int = 0) -> list[list[int]]:
    model.eval()
    device = src.device
    memory, src_mask = model.encode(src)
    b = src.size(0)
    ys = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        dec = model.decode(ys, memory, src_mask)
        logits = model.generator(dec[:, -1])          # (b, V)
        nxt = logits.argmax(-1)
        nxt[done] = pad_id
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done |= nxt == eos_id
        if done.all():
            break
    out = []
    for row in ys.tolist():
        row = row[1:]  # drop bos
        if eos_id in row:
            row = row[: row.index(eos_id)]
        out.append(row)
    return out


@torch.no_grad()
def beam_decode(model, src_1, beam: int = 5, max_len: int = 100, bos_id: int = 2,
                eos_id: int = 3, pad_id: int = 0, length_penalty: float = 0.6) -> list[int]:
    """Single-sentence beam search. src_1: (1, S)."""
    model.eval()
    device = src_1.device
    memory, src_mask = model.encode(src_1)
    memory = memory.expand(beam, -1, -1)
    src_mask = src_mask.expand(beam, -1, -1, -1)

    ys = torch.full((beam, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((beam,), -1e9, device=device)
    scores[0] = 0.0
    finished: list[tuple[float, list[int]]] = []

    for _ in range(max_len - 1):
        dec = model.decode(ys, memory, src_mask)
        logp = torch.log_softmax(model.generator(dec[:, -1]), dim=-1)   # (beam, V)
        vocab = logp.size(-1)
        cand = scores.unsqueeze(1) + logp                              # (beam, V)
        flat = cand.view(-1)
        top_s, top_i = flat.topk(beam)
        beam_idx = top_i // vocab
        tok_idx = top_i % vocab
        ys = torch.cat([ys[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        scores = top_s
        keep = []
        for i in range(beam):
            if tok_idx[i].item() == eos_id:
                seq = ys[i, 1:-1].tolist()
                lp = ((len(seq) + 1) ** length_penalty)
                finished.append((scores[i].item() / lp, seq))
                scores[i] = -1e9
            else:
                keep.append(i)
        if len(finished) >= beam:
            break
    if not finished:
        best = scores.argmax().item()
        return ys[best, 1:].tolist()
    finished.sort(key=lambda x: x[0], reverse=True)
    return finished[0][1]
