"""Phase 2 / 3 trainer for both the from-scratch Transformer and the RNN baseline.

Features required by the brief:
  * LR warmup-then-decay (Noam) for the Transformer
  * label smoothing 0.1
  * gradient clipping
  * mixed precision (torch.amp) with optional gradient accumulation for the 4GB GPU
  * checkpoint on best validation BLEU (sacrebleu, greedy)

Usage:
  python src/train.py --model transformer --seed 1 --tag main_s1
  python src/train.py --model rnn         --seed 1 --tag baseline_s1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from baseline import RNNSeq2Seq
from dataset import Multi30k, make_loader
from evaluate import evaluate_model
from model import Seq2SeqTransformer
from tokenizer import Tokenizer
from utils import JsonlLogger, LabelSmoothingLoss, NoamLR, set_seed

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
LOGS = ROOT / "logs"


def build_model(kind: str, vocab: int, pad_id: int, args) -> nn.Module:
    if kind == "transformer":
        return Seq2SeqTransformer(
            vocab_size=vocab, d_model=args.d_model, n_layers=args.layers,
            n_heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
            max_len=128, pad_id=pad_id, tie_embeddings=True,
        )
    return RNNSeq2Seq(vocab=vocab, emb_dim=256, hid_dim=512, dropout=0.3, pad_id=pad_id)


def run_epoch(model, kind, loader, crit, optim, sched, scaler, device, accum, clip):
    model.train()
    total, ntok = 0.0, 0
    optim.zero_grad(set_to_none=True)
    for step, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            if kind == "transformer":
                logits = model(src, tgt_in)
            else:
                logits = model(src, tgt, teacher_forcing=0.5)[:, 1:]
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss = loss / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()
        n = (tgt_out != crit.pad_id).sum().item()
        total += loss.item() * accum * n
        ntok += n
    return total / max(ntok, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["transformer", "rnn"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--lr_factor", type=float, default=1.0)
    ap.add_argument("--rnn_lr", type=float, default=7e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--resume", default=None, help="path to a *_last.pt checkpoint to resume from")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    tok = Tokenizer()
    vocab = len(tok)

    train_ds = Multi30k("train", tok)
    val_ds = Multi30k("val", tok)
    train_loader, train_sampler = make_loader(train_ds, args.batch_size, shuffle=True,
                                              seed=args.seed)

    model = build_model(args.model, vocab, tok.pad_id, args).to(device)
    crit = LabelSmoothingLoss(vocab, tok.pad_id, args.label_smoothing).to(device)

    if args.model == "transformer":
        optim = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamLR(optim, args.d_model, args.warmup, args.lr_factor)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.rnn_lr)
        sched = None
    plateau = (None if args.model == "transformer" else
               torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max",
                                                          factor=0.5, patience=3))

    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp and device == "cuda"))

    n_params = model.num_params()
    print(f"[{args.tag}] model={args.model} params={n_params:,} device={device} "
          f"amp={scaler.is_enabled()} train_pairs={len(train_ds)} (dropped {train_ds.dropped})")

    logger = JsonlLogger(LOGS / f"{args.tag}.jsonl")
    logger.log(event="config", **vars(args), n_params=n_params,
               train_pairs=len(train_ds), val_pairs=len(val_ds), device=device,
               amp=scaler.is_enabled())

    best_bleu, best_epoch, bad = -1.0, -1, 0
    start_epoch = 1
    CKPT.mkdir(exist_ok=True)
    t_start = time.time()

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        if sched is not None:
            sched.load_state_dict(ck["sched"])
        if plateau is not None and ck.get("plateau") is not None:
            plateau.load_state_dict(ck["plateau"])
        best_bleu, best_epoch, bad = ck["best_bleu"], ck["best_epoch"], ck["bad"]
        start_epoch = ck["epoch"] + 1
        t_start -= ck.get("elapsed_s", 0.0)
        print(f"[{args.tag}] resumed from {args.resume}: epoch {ck['epoch']} done, "
              f"best_bleu={best_bleu:.2f}@{best_epoch}, bad={bad} -> continuing at "
              f"epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        te0 = time.time()
        tr_loss = run_epoch(model, args.model, train_loader, crit, optim, sched,
                            scaler, device, args.accum, args.clip)
        val = evaluate_model(model, val_ds, device, decoder="greedy")
        if plateau is not None:
            plateau.step(val["bleu"])
        lr_now = (sched.lr if sched is not None else optim.param_groups[0]["lr"])
        dt = time.time() - te0
        print(f"  epoch {epoch:2d}  train_loss {tr_loss:.3f}  val_BLEU {val['bleu']:.2f}  "
              f"val_chrF {val['chrf']:.2f}  lr {lr_now:.2e}  {dt:.1f}s")
        logger.log(event="epoch", epoch=epoch, train_loss=round(tr_loss, 4),
                   val_bleu=val["bleu"], val_chrf=val["chrf"], lr=lr_now,
                   epoch_s=round(dt, 1))

        if val["bleu"] > best_bleu:
            best_bleu, best_epoch, bad = val["bleu"], epoch, 0
            torch.save({
                "model": model.state_dict(),
                "kind": args.model,
                "config": model.config,
                "args": vars(args),
                "epoch": epoch,
                "val_bleu": val["bleu"],
                "val_chrf": val["chrf"],
            }, CKPT / f"{args.tag}_best.pt")
        else:
            bad += 1

        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "sched": (sched.state_dict() if sched is not None else None),
            "plateau": (plateau.state_dict() if plateau is not None else None),
            "kind": args.model,
            "config": model.config,
            "args": vars(args),
            "epoch": epoch,
            "best_bleu": best_bleu,
            "best_epoch": best_epoch,
            "bad": bad,
            "elapsed_s": time.time() - t_start,
        }, CKPT / f"{args.tag}_last.pt")

        if bad >= args.patience:
            print(f"  early stop at epoch {epoch} (no val BLEU gain for {bad})")
            break

    total_min = (time.time() - t_start) / 60
    summary = {
        "tag": args.tag, "model": args.model, "seed": args.seed,
        "n_params": n_params, "best_val_bleu": best_bleu, "best_epoch": best_epoch,
        "epochs_run": epoch, "train_time_min": round(total_min, 1),
        "device": device, "amp": scaler.is_enabled(),
        "hparams": {k: getattr(args, k) for k in
                    ("batch_size", "accum", "d_model", "layers", "heads", "d_ff",
                     "dropout", "warmup", "lr_factor", "rnn_lr", "label_smoothing",
                     "clip", "patience", "epochs")},
    }
    (LOGS / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    logger.log(event="done", **summary)
    print(f"[{args.tag}] DONE  best val BLEU {best_bleu:.2f} @ epoch {best_epoch}  "
          f"({total_min:.1f} min)")


if __name__ == "__main__":
    main()
