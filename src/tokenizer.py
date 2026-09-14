"""Phase 0: joint subword tokenizer (SentencePiece BPE) for En-De Multi30k.

A single shared vocabulary is trained on the concatenation of the English and
German training text. Shared BPE is standard for related European language
pairs and lets the model tie source/target embeddings if desired.

Special ids are fixed so masking / loss code can rely on them:
    0 = <pad>   1 = <unk>   2 = <bos>   3 = <eos>
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TOK = ROOT / "data" / "tok"
TOK.mkdir(parents=True, exist_ok=True)

MODEL_PREFIX = TOK / "spm_joint_bpe8000"
VOCAB_SIZE = 8000

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_spm() -> None:
    if MODEL_PREFIX.with_suffix(".model").exists():
        print(f"skip spm train ({MODEL_PREFIX.with_suffix('.model').name} exists)")
        return
    combined = TOK / "train_combined.txt"
    with combined.open("w", encoding="utf-8") as out:
        for lang in ("en", "de"):
            for line in (RAW / f"train.{lang}").open(encoding="utf-8"):
                out.write(line)
    spm.SentencePieceTrainer.train(
        input=str(combined),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        num_threads=4,
    )
    combined.unlink()
    print(f"trained {MODEL_PREFIX.with_suffix('.model')}")


class Tokenizer:
    """Thin wrapper over the trained SentencePiece model."""

    def __init__(self, model_path: str | Path | None = None):
        model_path = model_path or MODEL_PREFIX.with_suffix(".model")
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = PAD_ID, UNK_ID, BOS_ID, EOS_ID

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(ids)


if __name__ == "__main__":
    train_spm()
    tok = Tokenizer()
    print("vocab size:", len(tok))
    s = "A man in a blue shirt is standing on a ladder cleaning a window."
    enc = tok.encode(s)
    print("example ids:", enc)
    print("round-trip :", tok.decode(enc))
