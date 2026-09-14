"""Phase 2: a pre-Transformer baseline -- GRU seq2seq with Bahdanau attention.

This is a different architecture family (recurrent, not attention-only), which
makes "the Transformer beats this by X BLEU" a clean, defensible claim rather
than a comparison between two versions of the same model. Kept deliberately
small so it trains fast and is not a straw man.

Reference: Bahdanau, Cho, Bengio 2015, "Neural MT by Jointly Learning to Align
and Translate". Implemented from scratch (nn.GRU is a standard primitive, not a
seq2seq black box).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, dropout, pad_id):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        emb = self.dropout(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # combine the two final directions -> initial decoder hidden state
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return outputs, hidden          # outputs: (B, S, 2H) ; hidden: (B, H)


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs, mask):
        s = hidden.size(1) if hidden.dim() == 3 else enc_outputs.size(1)
        h = hidden.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((h, enc_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)              # (B, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=1)                 # (B, S)


class Decoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, dropout, pad_id):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.attn = Attention(hid_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim * 2, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim * 3 + emb_dim, vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tok, hidden, enc_outputs, mask):
        tok = tok.unsqueeze(1)                          # (B,1)
        emb = self.dropout(self.emb(tok))              # (B,1,E)
        a = self.attn(hidden, enc_outputs, mask).unsqueeze(1)   # (B,1,S)
        ctx = torch.bmm(a, enc_outputs)               # (B,1,2H)
        rnn_in = torch.cat((emb, ctx), dim=2)
        out, hidden = self.rnn(rnn_in, hidden.unsqueeze(0))
        out = out.squeeze(1)
        pred = self.fc(torch.cat((out, ctx.squeeze(1), emb.squeeze(1)), dim=1))
        return pred, hidden.squeeze(0), a.squeeze(1)


class RNNSeq2Seq(nn.Module):
    def __init__(self, vocab, emb_dim=256, hid_dim=512, dropout=0.3, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(vocab, emb_dim, hid_dim, dropout, pad_id)
        self.decoder = Decoder(vocab, emb_dim, hid_dim, dropout, pad_id)
        self.config = dict(vocab=vocab, emb_dim=emb_dim, hid_dim=hid_dim,
                           dropout=dropout, pad_id=pad_id)

    def forward(self, src, tgt, teacher_forcing=0.5):
        src_len = (src != self.pad_id).sum(1)
        mask = src != self.pad_id
        enc_outputs, hidden = self.encoder(src, src_len)
        b, t = tgt.shape
        vocab = self.decoder.fc.out_features
        logits = torch.zeros(b, t, vocab, device=src.device)
        tok = tgt[:, 0]
        for i in range(1, t):
            pred, hidden, _ = self.decoder(tok, hidden, enc_outputs, mask)
            logits[:, i] = pred
            tok = tgt[:, i] if random.random() < teacher_forcing else pred.argmax(1)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
