"""Phase 5: real cross-attention heatmaps from the trained Transformer.

For a few test sentences we run greedy decoding and, at every generation step,
record the decoder->encoder (cross) attention of the LAST decoder layer for the
newest target token. The result is a (target_len x source_len) matrix of real
attention weights, plotted with matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import Multi30k
from model import Seq2SeqTransformer
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "attn"
OUT.mkdir(parents=True, exist_ok=True)


def load_transformer(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = Seq2SeqTransformer(**cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def greedy_with_attention(model, src, tok, max_len=100, head="mean"):
    device = src.device
    memory, src_mask = model.encode(src)
    ys = torch.full((1, 1), tok.bos_id, dtype=torch.long, device=device)
    attn_rows = []
    for _ in range(max_len - 1):
        dec = model.decode(ys, memory, src_mask, store_attn=True)
        logits = model.generator(dec[:, -1])
        cross = model.decoder.layers[-1].cross_attn.attn  # (1, h, T, S)
        vec = cross[0, :, -1, :]                          # (h, S)
        vec = vec.mean(0) if head == "mean" else vec[int(head)]
        attn_rows.append(vec.float().cpu())
        nxt = logits.argmax(-1)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        if nxt.item() == tok.eos_id:
            break
    gen_ids = ys[0, 1:].tolist()
    if tok.eos_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(tok.eos_id)]
    A = torch.stack(attn_rows[: len(gen_ids)], dim=0)     # (T, S)
    return gen_ids, A


# SentencePiece marks a subword that starts a new word with the code point
# U+2581 (a low one-eighth block). Referenced by code point, not as a literal
# character, so this source file stays plain ASCII.
_WORD_BOUNDARY_MARKER = chr(0x2581)


def pieces(tok, ids):
    # Replace the SentencePiece word-boundary marker with a plain underscore
    # for a readable axis label.
    return [tok.sp.id_to_piece(i).replace(_WORD_BOUNDARY_MARKER, "_") for i in ids]


def plot(src_pieces, tgt_pieces, A, title, path):
    fig, ax = plt.subplots(figsize=(max(4, len(src_pieces) * 0.5),
                                    max(3, len(tgt_pieces) * 0.45)))
    im = ax.imshow(A.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_pieces)))
    ax.set_xticklabels(src_pieces, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tgt_pieces)))
    ax.set_yticklabels(tgt_pieces, fontsize=8)
    ax.set_xlabel("source (English) subwords")
    ax.set_ylabel("generated (German) subwords")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "main_s1_best.pt"))
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 4, 11, 23, 77])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer()
    test = Multi30k("test", tok)
    model, ck = load_transformer(args.ckpt, device)
    print(f"loaded {args.ckpt}  (val BLEU {ck.get('val_bleu')}, epoch {ck.get('epoch')})")

    for idx in args.indices:
        src_ids = test.pairs[idx][0]
        src = torch.tensor([src_ids], device=device)
        gen_ids, A = greedy_with_attention(model, src, tok)
        sp = pieces(tok, [i for i in src_ids if i not in
                          (tok.bos_id, tok.eos_id, tok.pad_id)])
        # A columns include bos/eos of source; trim to match sp
        A = A[:, 1:1 + len(sp)]
        tp = pieces(tok, gen_ids)
        out = OUT / f"attn_test{idx:03d}.png"
        plot(sp, tp, A,
             f"test #{idx}\nEN: {test.src_text[test.kept_idx[idx]]}\n"
             f"hyp: {tok.decode(gen_ids)}", out)
        print(f"  #{idx}: {tok.decode(gen_ids)}  -> {out.name}")


if __name__ == "__main__":
    main()
