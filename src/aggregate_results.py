"""Phases 4/7/8: assemble the final results table from all eval + training runs.

Reads every outputs/eval_*.json and logs/*_summary.json, groups the main
Transformer runs by seed to report mean / std, and writes:
  outputs/results_table.txt   (human, plain text -- NOT markdown by request)
  outputs/results_table.json  (machine)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
LOGS = ROOT / "logs"


def load_evals() -> dict[str, dict]:
    d = {}
    for p in sorted(OUT.glob("eval_*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        d[j["tag"]] = j
    return d


def load_summaries() -> dict[str, dict]:
    d = {}
    for p in sorted(LOGS.glob("*_summary.json")):
        j = json.loads(p.read_text())
        d[j["tag"]] = j
    return d


def main():
    evals = load_evals()
    summ = load_summaries()

    rows = []
    for tag, ev in evals.items():
        s = summ.get(tag, {})
        rows.append({
            "tag": tag,
            "model": ev["kind"],
            "seed": s.get("seed"),
            "params": s.get("n_params"),
            "decoder": ev["decoder"],
            "test_bleu": ev["test"]["bleu"],
            "test_chrf": ev["test"]["chrf"],
            "val_bleu": ev.get("train_val_bleu"),
            "best_epoch": ev.get("best_epoch"),
            "train_min": s.get("train_time_min"),
            "bleu_sig": ev["test"]["bleu_sig"],
        })

    # seed aggregation for the main transformer (tags like main_s1, main_s2)
    main_rows = [r for r in rows if r["tag"].startswith("main_s")]
    agg = None
    if len(main_rows) >= 2:
        bl = [r["test_bleu"] for r in main_rows]
        cf = [r["test_chrf"] for r in main_rows]
        agg = {
            "n_seeds": len(main_rows),
            "seeds": [r["seed"] for r in main_rows],
            "bleu_mean": round(statistics.mean(bl), 2),
            "bleu_std": round(statistics.pstdev(bl), 2) if len(bl) > 1 else 0.0,
            "bleu_min": min(bl), "bleu_max": max(bl),
            "chrf_mean": round(statistics.mean(cf), 2),
            "chrf_std": round(statistics.pstdev(cf), 2) if len(cf) > 1 else 0.0,
        }

    result = {"rows": rows, "main_transformer_across_seeds": agg}
    (OUT / "results_table.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    L = []
    L.append("=" * 92)
    L.append("MULTI30K EN->DE  |  TEST SET RESULTS (sacrebleu)")
    L.append("=" * 92)
    hdr = f"{'run':<16}{'model':<13}{'seed':>5}{'params':>11}{'decode':>9}{'BLEU':>8}{'chrF':>8}{'val_BLEU':>10}{'ep':>4}{'min':>7}"
    L.append(hdr)
    L.append("-" * 92)
    for r in rows:
        L.append(
            f"{r['tag']:<16}{r['model']:<13}{str(r['seed'] or '-'):>5}"
            f"{(r['params'] or 0):>11,}{r['decoder']:>9}"
            f"{r['test_bleu']:>8.2f}{r['test_chrf']:>8.2f}"
            f"{(r['val_bleu'] if r['val_bleu'] is not None else 0):>10.2f}"
            f"{str(r['best_epoch'] or '-'):>4}{(r['train_min'] or 0):>7.1f}"
        )
    L.append("-" * 92)
    if agg:
        L.append(
            f"MAIN TRANSFORMER across {agg['n_seeds']} seeds {tuple(agg['seeds'])}: "
            f"BLEU {agg['bleu_mean']} +/- {agg['bleu_std']} "
            f"(min {agg['bleu_min']}, max {agg['bleu_max']}) | "
            f"chrF {agg['chrf_mean']} +/- {agg['chrf_std']}"
        )
    if rows:
        L.append("")
        L.append(f"BLEU signature: {rows[0]['bleu_sig']}")
    txt = "\n".join(L)
    (OUT / "results_table.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
