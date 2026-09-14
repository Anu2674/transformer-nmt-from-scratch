"""Phase 0 deliverable: real dataset statistics + example pairs.

Writes outputs/data_report.txt (human) and outputs/data_report.json (machine).
All numbers are computed from the downloaded files; nothing is hard-coded.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def lines(p: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in p.open(encoding="utf-8")]


def split_stats(tok: Tokenizer, split: str) -> dict:
    en, de = lines(RAW / f"{split}.en"), lines(RAW / f"{split}.de")
    assert len(en) == len(de)
    en_wtok = [len(s.split()) for s in en]
    de_wtok = [len(s.split()) for s in de]
    en_sub = [len(tok.encode(s, add_bos=False, add_eos=False)) for s in en]
    de_sub = [len(tok.encode(s, add_bos=False, add_eos=False)) for s in de]

    def summ(xs: list[int]) -> dict:
        xs_sorted = sorted(xs)
        return {
            "min": xs_sorted[0],
            "max": xs_sorted[-1],
            "mean": round(statistics.mean(xs), 2),
            "median": statistics.median(xs),
            "p95": xs_sorted[int(0.95 * (len(xs) - 1))],
        }

    return {
        "pairs": len(en),
        "en_words": summ(en_wtok),
        "de_words": summ(de_wtok),
        "en_subwords": summ(en_sub),
        "de_subwords": summ(de_sub),
    }


def main() -> None:
    tok = Tokenizer()
    report = {
        "dataset": "Multi30k (task1) English-German",
        "source": "https://github.com/multi30k/dataset",
        "tokenizer": {
            "type": "SentencePiece BPE, joint En+De",
            "vocab_size": len(tok),
            "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        },
        "splits": {s: split_stats(tok, s) for s in ("train", "val", "test")},
    }

    # OOV / <unk> rate on each split (how lossy the 8k vocab is)
    for s in ("train", "val", "test"):
        unk = tot = 0
        for lang in ("en", "de"):
            for line in lines(RAW / f"{s}.{lang}"):
                ids = tok.encode(line, add_bos=False, add_eos=False)
                unk += sum(1 for i in ids if i == tok.unk_id)
                tot += len(ids)
        report["splits"][s]["unk_rate_pct"] = round(100 * unk / tot, 4)

    examples = []
    for i in (0, 100, 500):
        examples.append(
            {
                "en": lines(RAW / "train.en")[i],
                "de": lines(RAW / "train.de")[i],
            }
        )
    report["example_pairs"] = examples

    (OUT / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("=" * 70)
    L.append("PHASE 0  -  MULTI30K (EN-DE) DATA REPORT")
    L.append("=" * 70)
    L.append(f"Dataset : {report['dataset']}")
    L.append(f"Source  : {report['source']}")
    L.append(f"Tokenizer: {report['tokenizer']['type']}  |  vocab = {report['tokenizer']['vocab_size']}")
    L.append("")
    for s in ("train", "val", "test"):
        st = report["splits"][s]
        L.append(f"[{s.upper()}]  {st['pairs']} sentence pairs   (<unk> rate {st['unk_rate_pct']}%)")
        L.append(
            f"    EN words   : mean {st['en_words']['mean']:>5}  median {st['en_words']['median']:>3}  "
            f"max {st['en_words']['max']:>3}  p95 {st['en_words']['p95']:>3}"
        )
        L.append(
            f"    DE words   : mean {st['de_words']['mean']:>5}  median {st['de_words']['median']:>3}  "
            f"max {st['de_words']['max']:>3}  p95 {st['de_words']['p95']:>3}"
        )
        L.append(
            f"    EN subword : mean {st['en_subwords']['mean']:>5}  max {st['en_subwords']['max']:>3}  "
            f"p95 {st['en_subwords']['p95']:>3}"
        )
        L.append(
            f"    DE subword : mean {st['de_subwords']['mean']:>5}  max {st['de_subwords']['max']:>3}  "
            f"p95 {st['de_subwords']['p95']:>3}"
        )
        L.append("")
    L.append("EXAMPLE TRAINING PAIRS")
    for e in examples:
        L.append(f"    EN: {e['en']}")
        L.append(f"    DE: {e['de']}")
        L.append("")
    txt = "\n".join(L)
    (OUT / "data_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
