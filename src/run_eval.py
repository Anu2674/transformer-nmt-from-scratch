"""Phase 4/6: evaluate a checkpoint on the FULL test set with sacrebleu.

Saves outputs/eval_<tag>.json with overall BLEU/chrF (+ signatures), the
length-bucket breakdown, per-sentence hypotheses, and best/worst examples by
sentence-level BLEU for error analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU

from baseline import RNNSeq2Seq
from dataset import Multi30k
from evaluate import evaluate_model
from model import Seq2SeqTransformer
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
_SBLEU = BLEU(effective_order=True)


def load_any(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ck["kind"] == "transformer":
        model = Seq2SeqTransformer(**ck["config"])
    else:
        model = RNNSeq2Seq(**ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def sentence_bleu(hyp, ref):
    return _SBLEU.sentence_score(hyp, [ref]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--decoder", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer()
    test = Multi30k("test", tok)
    model, ck = load_any(args.ckpt, device)
    res = evaluate_model(model, test, device, decoder=args.decoder, beam=args.beam)
    hyps = res.pop("_hyps")

    per = []
    for i, (s, r, h) in enumerate(zip(test.src_text, test.tgt_text, hyps)):
        per.append({"i": i, "src": s, "ref": r, "hyp": h,
                    "sent_bleu": round(sentence_bleu(h, r), 2),
                    "src_words": len(s.split())})
    per_sorted = sorted(per, key=lambda d: d["sent_bleu"])

    payload = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "kind": ck["kind"],
        "train_val_bleu": ck.get("val_bleu"),
        "best_epoch": ck.get("epoch"),
        "decoder": args.decoder + (f"{args.beam}" if args.decoder == "beam" else ""),
        "test": {k: res[k] for k in ("bleu", "bleu_sig", "chrf", "chrf_sig",
                                     "n_sentences", "decode_s")},
        "by_length": res["by_length"],
        "worst_10": per_sorted[:10],
        "best_10": per_sorted[-10:][::-1],
        "all_hyps": hyps,
    }
    out = OUT / f"eval_{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.tag}] test BLEU {res['bleu']}  chrF {res['chrf']}  "
          f"({res['bleu_sig']}) -> {out.name}")


if __name__ == "__main__":
    main()
