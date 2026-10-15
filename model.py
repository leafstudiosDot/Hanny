"""
It reads a few standard-library modules as training text (no downloads), trains a small
neural network to predict the next character, then generates Python-looking text.
Every concept from the lectures shows up here in about 80 lines.
"""
import argparse
import inspect
import json
import math
import os
import textwrap

import ast, dataclasses, difflib, shutil, socket, ssl, subprocess, tarfile, threading, typing

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=4000)
args = parser.parse_args()

BLOCK = 8      # how many previous characters the model looks at
EMB = 32       # size of each character's vector
HIDDEN = 512   # width of the hidden layers
BATCH = 128
LR = 3e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print("device:", device)

# Data: turn text into numbers
modules = [os, json, textwrap, argparse, inspect,
           ast, dataclasses, difflib, shutil, socket, ssl, subprocess, tarfile, threading, typing]
text = "\n".join(open(m.__file__, encoding="utf-8").read() for m in modules)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"{len(text):,} characters, vocabulary of {len(chars)}")


def get_batch(which):
    d = train_data if which == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])   # BLOCK characters of context
    y = torch.stack([d[i + BLOCK] for i in ix])     # the character that comes next
    return x.to(device), y.to(device)


# 2. Model: embedding, then a small MLP
class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB)
        self.net = nn.Sequential(
            nn.Linear(BLOCK * EMB, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, vocab),
        )

    def forward(self, x):
        return self.net(self.emb(x).flatten(1))   # scores for every possible next character


model = TinyLM(len(chars)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"loss of pure guessing would be about {math.log(len(chars)):.2f}")


@torch.no_grad()
def eval_loss(which):
    model.eval()
    total = 0.0
    for _ in range(20):
        x, y = get_batch(which)
        total += F.cross_entropy(model(x), y).item()
    model.train()
    return total / 20


# 3. Training loop: predict, measure error, nudge the weights
for step in range(args.steps + 1):
    if step % 500 == 0:
        print(f"step {step:5d} | train loss {eval_loss('train'):.3f} | val loss {eval_loss('val'):.3f}")
    x, y = get_batch("train")
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()


# 4. Generate: sample one character at a time
@torch.no_grad()
def generate(n_chars=400):
    model.eval()
    ctx = torch.zeros(1, BLOCK, dtype=torch.long, device=device)  # start with newlines
    out = []
    for _ in range(n_chars):
        probs = F.softmax(model(ctx), dim=-1)
        nxt = torch.multinomial(probs, 1)
        out.append(itos[nxt.item()])
        ctx = torch.cat([ctx[:, 1:], nxt], dim=1)
    return "".join(out)


print("\n- generated")
print(generate())