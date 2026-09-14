"""Phase 0: download the Multi30k (En-De) task-1 dataset.

Source: https://github.com/multi30k/dataset  (raw text, task1/raw/*.gz)
Splits used: train (29000), val (1014), test = test_2016_flickr (1000).

NOTE ON TLS: this machine runs Avast HTTPS interception with a CA cert that
Python 3.13's OpenSSL 3.x rejects ("Basic Constraints not marked critical").
pip is handled with --trusted-host; for these public, well-known dataset files
we fall back to an unverified TLS context. This is a documented, deliberate
workaround for a local AV setup, not a security choice in the code.
"""
import gzip
import io
import ssl
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"

FILES = {
    "train.en": "train.en.gz",
    "train.de": "train.de.gz",
    "val.en": "val.en.gz",
    "val.de": "val.de.gz",
    "test.en": "test_2016_flickr.en.gz",
    "test.de": "test_2016_flickr.de.gz",
}

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
        return r.read()


def main() -> None:
    for out_name, remote in FILES.items():
        out_path = RAW / out_name
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"skip  {out_name} (exists, {out_path.stat().st_size} bytes)")
            continue
        data = fetch(BASE + remote)
        text = gzip.decompress(data).decode("utf-8")
        # normalise: strip trailing spaces, ensure one sentence per line
        lines = [ln.rstrip("\n").rstrip() for ln in text.split("\n") if ln.strip() != ""]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out_name}: {len(lines)} lines, {out_path.stat().st_size} bytes")

    # integrity check: parallel line counts must match
    for split in ("train", "val", "test"):
        n_en = sum(1 for _ in (RAW / f"{split}.en").open(encoding="utf-8"))
        n_de = sum(1 for _ in (RAW / f"{split}.de").open(encoding="utf-8"))
        assert n_en == n_de, f"{split}: {n_en} en vs {n_de} de"
        print(f"check {split}: {n_en} parallel pairs")


if __name__ == "__main__":
    sys.exit(main())
