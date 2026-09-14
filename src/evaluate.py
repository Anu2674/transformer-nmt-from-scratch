"""Evaluation with sacrebleu (standard BLEU + chrF), plus length-bucket breakdown.

BLEU here is sacrebleu's default: case-sensitive, `13a` tokenisation, exp
smoothing -- the same configuration reported in most MT papers, so the numbers
are comparable. Signature is printed alongside every score.
"""
from __future__ import annotations

import time

import torch
from sacrebleu.metrics import BLEU, CHRF

_BLEU = BLEU()      # defaults: case-sensitive, 13a tokeniser, exp smoothing
_CHRF = CHRF()      # defaults: chrF (beta=2), not chrF++

from baseline import RNNSeq2Seq
from dataset import Multi30k, make_loader
from search import greedy_decode, rnn_greedy_decode


@torch.no_grad()
def translate_split(model, ds: Multi30k, device, batch_size: int = 64,
                    max_len: int = 100, decoder="greedy", beam: int = 5) -> list[str]:
    """Translate the whole split and return hypotheses in ORIGINAL order.

    The loader reorders sentences into length buckets, so each batch carries its
    dataset indices and results are scattered back into place afterwards.
    """
    model.eval()
    tok = ds.tok
    is_rnn = isinstance(model, RNNSeq2Seq)
    loader, sampler = make_loader(ds, batch_size, shuffle=False)
    batch_index_lists = list(iter(sampler))

    hyps: list[str | None] = [None] * len(ds)
    for (src, _), idxs in zip(loader, batch_index_lists):
        src = src.to(device)
        if is_rnn:
            outs = rnn_greedy_decode(model, src, max_len=max_len, bos_id=tok.bos_id,
                                     eos_id=tok.eos_id, pad_id=tok.pad_id)
        elif decoder == "greedy":
            outs = greedy_decode(model, src, max_len=max_len, bos_id=tok.bos_id,
                                 eos_id=tok.eos_id, pad_id=tok.pad_id)
        else:
            from search import beam_decode
            outs = [beam_decode(model, src[i:i + 1], beam=beam, max_len=max_len,
                                bos_id=tok.bos_id, eos_id=tok.eos_id, pad_id=tok.pad_id)
                    for i in range(src.size(0))]
        for j, row in zip(idxs, outs):
            hyps[j] = tok.decode(row)
    assert all(h is not None for h in hyps)
    # map from kept-subset order back to raw file order
    full = [""] * len(ds.src_text)
    for k, raw_i in enumerate(ds.kept_idx):
        full[raw_i] = hyps[k]
    return full


def score(hyps: list[str], refs: list[str]) -> dict:
    bleu = _BLEU.corpus_score(hyps, [refs])
    chrf = _CHRF.corpus_score(hyps, [refs])
    return {
        "bleu": round(bleu.score, 2),
        "bleu_sig": _BLEU.get_signature().format(short=True),
        "chrf": round(chrf.score, 2),
        "chrf_sig": _CHRF.get_signature().format(short=True),
    }


def length_buckets(hyps, refs, src_texts, edges=(0, 10, 20, 1000)) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, s in enumerate(src_texts) if lo < len(s.split()) <= hi]
        if not idx:
            continue
        h = [hyps[i] for i in idx]
        r = [refs[i] for i in idx]
        out.append({
            "range": f"{lo+1}-{hi if hi < 1000 else 'inf'} src words",
            "n": len(idx),
            **score(h, r),
        })
    return out


@torch.no_grad()
def evaluate_model(model, ds: Multi30k, device, decoder="greedy", beam=5) -> dict:
    t0 = time.time()
    hyps = translate_split(model, ds, device, decoder=decoder, beam=beam)
    refs = ds.tgt_text
    res = score(hyps, refs)
    res["decode_s"] = round(time.time() - t0, 1)
    res["n_sentences"] = len(hyps)
    res["by_length"] = length_buckets(hyps, refs, ds.src_text)
    res["sample"] = [
        {"src": ds.src_text[i], "ref": refs[i], "hyp": hyps[i]}
        for i in (0, 1, 2, 10, 50)
    ]
    res["_hyps"] = hyps
    return res
