"""Small CLI demo: translate English -> German with a trained checkpoint.

  echo "A man is riding a bike." | python src/translate.py --ckpt checkpoints/main_s1_best.pt --beam 5
  python src/translate.py --ckpt checkpoints/main_s1_best.pt --text "Two dogs are playing in the snow."
"""
from __future__ import annotations

import argparse
import sys

import torch

from run_eval import load_any
from search import beam_decode, greedy_decode
from tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--beam", type=int, default=0, help="0 = greedy")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer()
    model, ck = load_any(args.ckpt, device)

    lines = [args.text] if args.text else [ln.strip() for ln in sys.stdin if ln.strip()]
    for line in lines:
        src = torch.tensor([tok.encode(line)], device=device)
        if args.beam and ck["kind"] == "transformer":
            ids = beam_decode(model, src, beam=args.beam, bos_id=tok.bos_id,
                              eos_id=tok.eos_id, pad_id=tok.pad_id)
        else:
            ids = greedy_decode(model, src, bos_id=tok.bos_id, eos_id=tok.eos_id,
                                pad_id=tok.pad_id)[0]
        print(f"EN: {line}")
        print(f"DE: {tok.decode(ids)}\n")


if __name__ == "__main__":
    main()
