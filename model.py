"""
Same idea as before (predict the next character), with two big changes:
  1. Attention lets every character look at ALL earlier characters in its window (128, not 8).
  2. Validation uses one whole held-out module (tarfile), so val loss is comparable between runs.
This split differs from model.py, so treat this run as a NEW baseline, not a comparison to it.
"""
import argparse
import ast
import dataclasses
import difflib
import inspect
import json
import math
import os
import shutil
import socket
import ssl
import subprocess
import tarfile
import textwrap
import threading
import time
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--prompt", type=str, default="def ")
parser.add_argument("--temperature", type=float, default=0.8)
args = parser.parse_args()

# Settings (change ONE at a time, then log what happened)
BLOCK = 128      # how many previous characters the model can look at
N_EMBD = 128     # size of each character's vector
N_HEAD = 4       # attention heads (each one looks for a different kind of pattern)
N_LAYER = 4      # how many attention + MLP blocks are stacked
DROPOUT = 0.1    # randomly switches off units while training to reduce memorizing
BATCH = 64
LR = 1e-3

# Checkpoints live in a checkpoints/ folder next to this script, not in the project root
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "rowboat_m1_1.pt")
os.makedirs(CKPT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print("device:", device)

# 1. Data
VAL_MODULE = tarfile  # never trained on: gives an honest score
train_modules = [os, json, textwrap, argparse, inspect, ast, dataclasses, difflib,
                 shutil, socket, ssl, subprocess, threading, typing]


def read(m):
    return open(m.__file__, encoding="utf-8").read()


train_text = "\n".join(read(m) for m in train_modules)
val_text = read(VAL_MODULE)
chars = sorted(set(train_text + "\n" + val_text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
train_data = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
val_data = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
print(f"train: {len(train_data):,} chars | val: {len(val_data):,} chars | vocab: {len(chars)}")


def get_batch(which):
    d = train_data if which == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])  # the same window shifted by one
    return x.to(device), y.to(device)


# 2. Model
class CausalSelfAttention(nn.Module):
    """Each position asks: which earlier characters matter for predicting what comes next?"""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)   # query, key, value for all heads at once
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        # lower-triangle mask: a character may look backwards, never at the future
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK, BLOCK)).view(1, 1, BLOCK, BLOCK))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hs = C // N_HEAD
        q = q.view(B, T, N_HEAD, hs).transpose(1, 2)   # (B, heads, T, head_size)
        k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)          # how well each query matches each key
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))                   # turn match scores into weights
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)  # weighted mix of the values
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection: add, don't replace
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, N_EMBD)   # what character is this?
        self.pos_emb = nn.Embedding(BLOCK, N_EMBD)   # where in the window is it?
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))   # (B, T, vocab): a prediction at EVERY position
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature):
        for _ in range(n_new):
            logits, _ = self(idx[:, -BLOCK:])                      # only the last BLOCK characters fit
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


model = GPT(len(chars)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"loss of pure guessing would be about {math.log(len(chars)):.2f}")


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for which in ("train", "val"):
        total = 0.0
        for _ in range(30):
            x, y = get_batch(which)
            total += model(x, y)[1].item()
        out[which] = total / 30
    model.train()
    return out


# 3. Training loop
best_val = float("inf")
t0 = time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        losses = estimate_loss()
        marker = ""
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "chars": chars}, CKPT_PATH)
            marker = "  <- best so far, saved"
        print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
              f"| {time.time() - t0:5.0f}s{marker}")
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

# 4. Generate from the BEST checkpoint (not necessarily the last step)
ckpt = torch.load("hanny-m1_1.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
start = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
out = model.generate(start, 600, args.temperature)[0].tolist()
print(f"\nbest val loss: {best_val:.3f}")
print("\n- output -")
print("".join(itos[i] for i in out))