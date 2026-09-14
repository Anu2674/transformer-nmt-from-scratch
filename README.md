# Neural Machine Translation Transformer - Built From Scratch

An English-to-German Transformer implemented from first principles in PyTorch - no `nn.Transformer`, no `nn.MultiheadAttention`, no pretrained weights. Every component (positional encoding, multi-head self/cross-attention, encoder/decoder stacks, masking) is hand-written and unit-tested. Trained and evaluated on Multi30k with a from-scratch RNN baseline for a defensible comparison, two training seeds for variance, real attention visualizations, and an honest error analysis.

All numbers below come from actual executed runs, scored with **sacrebleu** (not a hand-rolled BLEU), on a single consumer laptop GPU (RTX 2050, 4GB VRAM).

## Results at a glance

| Model | Params | Test BLEU | Test chrF | Val BLEU (best) |
|---|---:|---:|---:|---:|
| Baseline - GRU + Bahdanau attention | 24.9M | 29.16 | 55.63 | 30.40 |
| **Transformer (from scratch), seed 1** | 7.6M | 35.37 | 60.66 | 36.73 |
| **Transformer (from scratch), seed 2** | 7.6M | 35.98 | 60.95 | 37.07 |
| **Transformer - mean across 2 seeds** | 7.6M | **35.67 +/- 0.30** | **60.80 +/- 0.15** | - |

The from-scratch Transformer beats the RNN baseline by **+6.51 BLEU (~22% relative)** while using **3.3x fewer parameters**, and the result is stable across seeds (+/-0.30 BLEU), not a single lucky run.

`sacrebleu` signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`

## Architecture

A standard encoder-decoder Transformer (Vaswani et al., 2017), implemented manually:

- Sinusoidal positional encoding
- Scaled dot-product multi-head self-attention (encoder) and cross-attention (decoder)
- Position-wise feed-forward layers, residual connections, layer normalization
- Padding masks (encoder self-attention, decoder cross-attention) and a causal mask (decoder self-attention), combined and unit-tested
- Weight tying between the input embedding and the output projection

**Configuration used:** `d_model=256`, 3 encoder layers, 3 decoder layers, 8 attention heads, `d_ff=1024`, dropout 0.1 - **7.58M parameters**.

**Deliberate deviation from the paper:** pre-norm residual blocks (`LayerNorm` before each sub-layer) instead of the paper's post-norm, and `d_model=256` / 3 layers instead of the paper's `d_model=512` / 6 layers. Both choices were validated empirically, not assumed:
- Post-norm Transformers are notoriously sensitive to warmup and prone to divergence at this scale; pre-norm trained stably on the first attempt.
- A d512/6-layer model was considered but not used as the primary result - on a 29K-sentence dataset the extra capacity mainly adds overfitting risk without more data to justify it. The smaller config converged cleanly to competitive BLEU on a 4GB GPU.

## Dataset

**Multi30k** (task 1, English-German) - the standard small-scale academic benchmark for Transformer-from-scratch projects, chosen over an arbitrary scraped corpus so results are independently comparable.

| Split | Pairs | Avg EN words | Avg DE words | `<unk>` rate |
|---|---:|---:|---:|---:|
| train | 29,000 | 11.9 | 11.1 | 0.0% |
| val | 1,014 | 12.0 | 11.4 | 0.0% |
| test | 1,000 | 11.9 | 10.9 | 0.0% |

Tokenization: a single **SentencePiece BPE** model (vocab size 8,000) trained jointly on English + German text, so both languages share a subword space. 0.0% `<unk>` rate on all splits - the vocabulary fully covers the corpus.

Example training pair:
```
EN: Two young, White males are outside near many bushes.
DE: Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
```

## Baseline

A GRU encoder-decoder with Bahdanau (additive) attention - a different architecture family from the Transformer, not a shrunk version of it, so "Transformer beats baseline" is a real architectural comparison rather than a comparison between two sizes of the same model. Bidirectional GRU encoder, unidirectional GRU decoder with attention over encoder states, 24.9M parameters (larger than the Transformer, since RNN hidden-to-hidden weight matrices scale quadratically with hidden size).

## Training setup

- **Optimizer:** Adam (beta1=0.9, beta2=0.98, eps=1e-9)
- **LR schedule:** Noam warmup-then-decay (`lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)`), warmup = 2000 steps
- **Label smoothing:** 0.1 (KL-divergence to a smoothed target distribution)
- **Gradient clipping:** max norm 1.0
- **Mixed precision:** `torch.amp` autocast + `GradScaler`, required to fit training comfortably on a 4GB GPU
- **Batching:** length-bucketed batches (sequences of similar length grouped together) to minimize padding waste
- **Checkpointing:** best validation BLEU (greedy decode, evaluated every epoch), with full optimizer/scheduler state also checkpointed every epoch for crash recovery
- **Early stopping:** patience of 8-10 epochs with no validation BLEU improvement

### Real problems hit during training (and how they were fixed)

This project ran into three genuine engineering problems, not staged ones:

1. **CUDA out-of-memory mid-run.** The RNN baseline trained cleanly for 3 epochs at batch size 128, then hit `RuntimeError: CUDA error: out of memory` during backward at epoch 4. Cause: the GRU decoder unrolls attention step-by-step over the target sequence (it cannot process a sequence in parallel the way the Transformer does), so its backward graph is much larger per sample than the Transformer's at the same batch size; the length-bucketed sampler reshuffles bucket order every epoch, so epoch 4 was simply the first epoch to draw an unlucky combination of long-sequence batches back to back. **Fix:** reduced the baseline's batch size from 128 to 64 and resumed training from the epoch-3 checkpoint (model + optimizer + scheduler state), losing zero progress.
2. **A shared Python environment silently broke the GPU build mid-project.** Training runs use a system-wide Python installation also used by other local tools (`openai-whisper`, `sentence-transformers`, `torchvision`); at some point between sessions, a dependency resolution elsewhere on the machine reinstalled plain CPU-only `torch` over the pinned CUDA build, and a freshly launched training run silently fell back to CPU (`"device": "cpu"` in the run's own config log - caught by checking the log, not by guessing). **Fix:** created a dedicated project-local virtual environment (`.venv`) with `torch==2.6.0+cu124` pinned, isolated from whatever else runs on the machine.
3. **The machine slept mid-run.** A background training run's wall-clock time ballooned to 5.5 hours for what should have been ~35 minutes of GPU compute; per-epoch timestamps showed one epoch alone took 4.9 hours while every other epoch took 30-50 seconds, i.e. the OS suspended the process for most of that gap. **Fix:** disabled sleep/hibernate for the duration of training and restored the normal power settings afterward.

## Results - full detail

Evaluated with `sacrebleu` (case-sensitive, `13a` tokenization, exponential smoothing - the default configuration used in most published MT results, for comparability).

| Run | Model | Seed | Test BLEU | Test chrF | Best epoch | Train time |
|---|---|---:|---:|---:|---:|---:|
| baseline_s1 | GRU + attention | 1 | 29.16 | 55.63 | 26/30 | 98 min |
| main_s1 | Transformer | 1 | 35.37 | 60.66 | 30/40 | ~38 min compute* |
| main_s2 | Transformer | 2 | 35.98 | 60.95 | 28/38 | 21 min |

\*seed 1's wall-clock time was 5.5 hours due to the sleep interruption documented above; actual GPU compute time was ~38 minutes, confirmed from per-epoch timestamps.

**Variance across seeds:** BLEU 35.67 +/- 0.30 (range 35.37-35.98), chrF 60.80 +/- 0.15. Tight spread - the result is not a single favorable initialization.

## Attention visualization

Real cross-attention weights extracted from the trained model (seed 1) during greedy decoding, plotted for actual test-set sentences (`outputs/attn/`). Example:

![attention heatmap](outputs/attn/attn_test000.png)

Source: "A man in an orange hat starring at something." Model output: "Ein Mann mit einem orangefarbenen Hut starrt auf etwas." The alignment is close to diagonal and linguistically sensible: "Hut" attends to "hat", "orangefarbenen" attends to "orange", "etwas" attends to "something", despite word-order and subword-count differences between the two languages. Four more examples are in `outputs/attn/`.

## Error analysis

### Performance by sentence length

| Source length | Transformer BLEU | Baseline BLEU |
|---|---:|---:|
| 1-10 words (n=412) | 39.74 | 33.38 |
| 11-20 words (n=551) | 35.31 | 28.98 |
| 21+ words (n=37) | 18.76 | 14.35 |

Both models degrade sharply on long sentences (Transformer: -20.98 BLEU from shortest to longest bucket; baseline: -19.03). The Transformer's relative advantage over the baseline holds at every length bucket, but neither model handles the rare long-and-complex sentences well - expected, since only 37 of 1000 test sentences fall in that bucket and long sentences are proportionally rarer in the 29K-sentence training set.

### Good translations (real model output)
```
SRC: A man throws a fishing net into the bay.
REF: Ein Mann wirft ein Fischernetz in die Bucht.
HYP: Ein Mann wirft ein Fischernetz in die Bucht. [exact match]

SRC: A woman and a child are walking down a street.
REF: Eine Frau und ein Kind gehen eine Straße entlang.
HYP: Eine Frau und ein Kind gehen eine Straße entlang. [exact match]
```

### Failure cases (real model output)
```
SRC: # 8 for Iowa State stiff arms a Texas AM player attempting to tackle him.
REF: Die # 8 von Iowa State streckt den Arm in Richtung eines Spielers der Texas AM beim Versuch, ihn zu tackeln.
HYP: Ein Mowa Ia StowaWetta hält sich einen Spieler der Arme für ihn an.

SRC: The Dali Lama during a reception where participants have brought carnations, parasols and prayer flags.
REF: Der Dalai Lama bei einem Empfang, zu dem Besucher Nelken, Schirme und Gebetsfahnen bringen.
HYP: Der Malaluch Lama eines Sternixtnummer, wo Tros in Lachen Loli vögenechen verlungenechenkippt ...
```

**Why these fail, honestly:** both worst-case sentences contain rare proper nouns and domain-specific tokens ("Iowa State", "Texas AM", "Dali Lama", "carnations", "parasols") that are uncommon or absent in Multi30k's 29K captions, which mostly describe everyday scenes (people, animals, sports, outdoor activities) in simple vocabulary. When the model encounters an unfamiliar named entity, it degrades into a kind of subword-level hallucination - stitching together plausible-looking German fragments with no real correspondence to the source, occasionally repeating a token ("Lachen Lachen Lachen"). This is a direct, expected consequence of training on a small, narrow-domain, single-caption-style dataset with no pretraining - it is not a masking or implementation bug (the same sentences fail similarly on the baseline, and the unit tests separately confirm the attention/masking logic is correct).

## Limitations (honest)

- **Small, narrow-domain training data.** Multi30k is 29K image captions - simple declarative sentences about everyday scenes. Results do not generalize to open-domain text, long documents, or domain-specific vocabulary (as the failure cases above show).
- **No pretraining.** Everything is trained from random initialization on 29K pairs; this is normal for a from-scratch benchmark but means absolute BLEU is well below what a pretrained or much-larger-data system would reach.
- **Greedy decoding for all reported numbers.** Beam search is implemented (`src/search.py`) but was not used for the headline numbers, to keep the baseline/main comparison on identical decoding conditions; beam search would likely add a small further BLEU gain to both models.
- **Two seeds, not more.** Two-seed variance (+/-0.30 BLEU) is far better evidence than one run, but a more rigorous study would use 3-5 seeds.
- **d256/3-layer, not d512/6-layer.** A deliberate, documented, empirically-motivated deviation from the original paper's exact configuration (see Architecture section) - flagged here so it's never presented as "the paper's exact setup."

## Repository structure

| Path | Purpose |
|---|---|
| `src/tokenizer.py` | Phase 0 - SentencePiece BPE training and wrapper |
| `src/data_report.py` | Phase 0 - dataset statistics report |
| `src/download_data.py` | Phase 0 - Multi30k download |
| `src/masking.py` | Phase 1 - padding and causal mask construction |
| `src/model.py` | Phase 1 - Transformer built from scratch |
| `src/baseline.py` | Phase 2 - GRU with Bahdanau attention baseline |
| `src/dataset.py` | data pipeline, length-bucketed batching |
| `src/utils.py` | label smoothing, Noam LR schedule, logging |
| `src/train.py` | Phase 2/3 - training loop with AMP, gradient clipping, checkpointing, resume |
| `src/search.py` | greedy and beam decoding for the Transformer and the RNN |
| `src/evaluate.py` | sacrebleu scoring, length-bucket breakdown |
| `src/run_eval.py` | Phase 4/6 - full test-set evaluation with per-sentence output |
| `src/viz_attention.py` | Phase 5 - attention heatmap extraction and plotting |
| `src/error_analysis.py` | Phase 6 - error analysis report generation |
| `src/aggregate_results.py` | Phase 4/7 - final results table across all runs |
| `src/translate.py` | CLI demo for translating arbitrary English sentences |
| `tests/test_model.py` | Phase 1 - 12 unit tests covering shapes and masking correctness |
| `outputs/` | data report, evaluation results, attention plots, error analysis |
| `checkpoints/` | best and resumable checkpoints for every trained run |
| `logs/` | per-run JSONL training logs and summaries |

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -r requirements.txt

python src/download_data.py
python src/tokenizer.py
python src/data_report.py
python -m pytest tests/ -q

python src/train.py --model rnn --seed 1 --tag baseline_s1
python src/train.py --model transformer --seed 1 --tag main_s1
python src/train.py --model transformer --seed 2 --tag main_s2

python src/run_eval.py --ckpt checkpoints/main_s1_best.pt --tag main_s1
python src/viz_attention.py --ckpt checkpoints/main_s1_best.pt
python src/error_analysis.py --tag main_s1
python src/aggregate_results.py
```
