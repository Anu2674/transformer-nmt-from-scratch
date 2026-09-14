"""Phase 6: error analysis for one model (default: main transformer, seed 1).

Consumes outputs/eval_<tag>.json (produced by run_eval.py) and writes
outputs/error_analysis_<tag>.txt: length-bucket BLEU/chrF, a length-vs-quality
trend, and real good / bad qualitative examples with short automatic diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def diagnose(ex: dict) -> str:
    hyp, ref = ex["hyp"], ex["ref"]
    h_w, r_w = hyp.split(), ref.split()
    tags = []
    if len(h_w) < 0.6 * len(r_w):
        tags.append("under-translation (hyp much shorter than ref)")
    if len(h_w) > 1.6 * len(r_w):
        tags.append("over-generation / repetition")
    if len(set(h_w)) < 0.6 * len(h_w) and len(h_w) > 4:
        tags.append("token repetition")
    if ex["src_words"] > 20:
        tags.append("long source sentence")
    if not tags:
        tags.append("lexical / word-choice errors, meaning mostly wrong")
    return "; ".join(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main_s1")
    args = ap.parse_args()
    ev = json.loads((OUT / f"eval_{args.tag}.json").read_text(encoding="utf-8"))

    L = []
    L.append("=" * 78)
    L.append(f"PHASE 6  -  ERROR ANALYSIS  ({args.tag}, {ev['kind']}, decode={ev['decoder']})")
    L.append("=" * 78)
    L.append(f"Overall test BLEU {ev['test']['bleu']}   chrF {ev['test']['chrf']}")
    L.append(f"Signature: {ev['test']['bleu_sig']}")
    L.append("")
    L.append("PERFORMANCE BY SOURCE SENTENCE LENGTH")
    L.append(f"  {'bucket':<22}{'n':>6}{'BLEU':>9}{'chrF':>9}")
    for b in ev["by_length"]:
        L.append(f"  {b['range']:<22}{b['n']:>6}{b['bleu']:>9.2f}{b['chrf']:>9.2f}")
    buckets = ev["by_length"]
    if len(buckets) >= 2:
        delta = buckets[0]["bleu"] - buckets[-1]["bleu"]
        L.append("")
        L.append(f"  -> BLEU drop from shortest to longest bucket: {delta:+.2f}")
    L.append("")
    L.append("GOOD TRANSLATIONS (highest sentence-BLEU)")
    for ex in ev["best_10"][:4]:
        L.append(f"  [sentBLEU {ex['sent_bleu']:.1f}]  ({ex['src_words']} src words)")
        L.append(f"    SRC: {ex['src']}")
        L.append(f"    REF: {ex['ref']}")
        L.append(f"    HYP: {ex['hyp']}")
        L.append("")
    L.append("FAILURE CASES (lowest sentence-BLEU)")
    for ex in ev["worst_10"][:5]:
        L.append(f"  [sentBLEU {ex['sent_bleu']:.1f}]  ({ex['src_words']} src words)")
        L.append(f"    SRC: {ex['src']}")
        L.append(f"    REF: {ex['ref']}")
        L.append(f"    HYP: {ex['hyp']}")
        L.append(f"    likely cause: {diagnose(ex)}")
        L.append("")
    txt = "\n".join(L)
    (OUT / f"error_analysis_{args.tag}.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
