"""Phase 1 deliverable: unit tests for shapes + masking correctness.

Run:  python -m pytest tests/ -q     (or)     python tests/test_model.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from masking import make_pad_mask, make_causal_mask, make_decoder_self_mask  # noqa: E402
from model import (  # noqa: E402
    PositionalEncoding,
    MultiHeadAttention,
    Seq2SeqTransformer,
)

torch.manual_seed(0)
V, B, D, H = 200, 4, 32, 4


def test_causal_mask_is_lower_triangular():
    m = make_causal_mask(5)[0, 0]
    assert m.shape == (5, 5)
    assert torch.equal(m, torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    assert m[0].sum() == 1 and m[4].sum() == 5


def test_pad_mask_shape_and_values():
    seq = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
    m = make_pad_mask(seq, pad_id=0)
    assert m.shape == (2, 1, 1, 4)
    assert m[0, 0, 0].tolist() == [True, True, False, False]
    assert m[1, 0, 0].tolist() == [True, True, True, False]


def test_decoder_self_mask_combines_pad_and_causal():
    tgt = torch.tensor([[2, 5, 6, 0]])
    m = make_decoder_self_mask(tgt, pad_id=0)[0, 0]
    assert m.shape == (4, 4)
    # row 2 (third query) may see cols 0,1,2 but not the padded col 3
    assert m[2].tolist() == [True, True, True, False]
    # causal: query 0 sees only key 0
    assert m[0].tolist() == [True, False, False, False]


def test_positional_encoding_preserves_shape():
    pe = PositionalEncoding(D, dropout=0.0, max_len=50)
    x = torch.randn(B, 10, D)
    assert pe(x).shape == x.shape


def test_mha_output_shape():
    mha = MultiHeadAttention(D, H, dropout=0.0)
    q = torch.randn(B, 7, D)
    out = mha(q, q, q)
    assert out.shape == (B, 7, D)
    assert mha.n_heads * mha.d_k == D


def test_mha_causal_mask_blocks_future():
    """With a causal mask, output at position t must not depend on inputs > t."""
    torch.manual_seed(1)
    mha = MultiHeadAttention(D, H, dropout=0.0).eval()
    x = torch.randn(1, 6, D)
    mask = make_causal_mask(6)
    out1 = mha(x, x, x, mask)
    x2 = x.clone()
    x2[0, 5] += 10.0  # perturb the LAST position only
    out2 = mha(x2, x2, x2, mask)
    # positions 0..4 must be unchanged; position 5 may change
    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-5)
    assert not torch.allclose(out1[0, 5], out2[0, 5], atol=1e-5)


def test_mha_padding_mask_ignores_pad_keys():
    torch.manual_seed(2)
    mha = MultiHeadAttention(D, H, dropout=0.0).eval()
    x = torch.randn(1, 5, D)
    keep = torch.tensor([[[[True, True, True, False, False]]]])  # last 2 are pad
    out1 = mha(x, x, x, keep)
    x2 = x.clone()
    x2[0, 3:] += 5.0  # change only the padded key positions
    out2 = mha(x2, x2, x2, keep)
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5)


def _tiny_model():
    return Seq2SeqTransformer(
        vocab_size=V, d_model=D, n_layers=2, n_heads=H, d_ff=64, dropout=0.0, max_len=64
    ).eval()


def test_transformer_forward_shape():
    m = _tiny_model()
    src = torch.randint(4, V, (B, 9))
    tgt_in = torch.randint(4, V, (B, 7))
    logits = m(src, tgt_in)
    assert logits.shape == (B, 7, V)


def test_transformer_decoder_is_causal_end_to_end():
    m = _tiny_model()
    src = torch.randint(4, V, (1, 9))
    tgt = torch.randint(4, V, (1, 8))
    logits1 = m(src, tgt)
    tgt2 = tgt.clone()
    tgt2[0, -1] = (tgt2[0, -1] + 1) % V  # change the final target token
    logits2 = m(src, tgt2)
    assert torch.allclose(logits1[0, :-1], logits2[0, :-1], atol=1e-5)


def test_transformer_src_padding_is_ignored():
    """Appending trailing <pad> to the source must not change outputs for the
    real tokens (this is what the encoder pad-mask + cross-attn pad-mask buy us)."""
    m = _tiny_model()
    core = torch.randint(4, V, (1, 7))
    tgt = torch.randint(4, V, (1, 6))
    out_short = m(core, tgt)
    padded = torch.cat([core, torch.zeros(1, 3, dtype=torch.long)], dim=1)  # +3 pad
    out_padded = m(padded, tgt)
    assert torch.allclose(out_short, out_padded, atol=1e-5)


def test_transformer_tgt_padding_is_ignored():
    """Trailing <pad> on the target side must not change logits at real positions."""
    m = _tiny_model()
    src = torch.randint(4, V, (1, 9))
    core = torch.randint(4, V, (1, 5))
    out_short = m(src, core)
    padded = torch.cat([core, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    out_padded = m(src, padded)
    assert torch.allclose(out_short, out_padded[:, :5], atol=1e-5)


def test_weight_tying_and_param_count():
    m = _tiny_model()
    assert m.generator.weight.data_ptr() == m.encoder.emb.weight.data_ptr()
    assert m.decoder.emb.weight.data_ptr() == m.encoder.emb.weight.data_ptr()
    assert m.num_params() > 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
