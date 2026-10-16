"""
Setup:   list your code folders in data_dirs.txt (one per line; start a line with ! to leave a folder out).
Check:   python data.py          (shows what will be used and skipped, no training)
Train:   python model.py     (add --run my-name to name the run)
The prepared data is cached in cache/, so later runs start fast. After adding code, it rebuilds by itself.

Files that look like they contain API keys, tokens, private keys or hardcoded passwords are
skipped entirely (see data.py). Whole files are held out for validation.
Each run gets its own folder under checkpoints/ with best.pt, last.pt, sample.txt, data_report.txt.
"""
import argparse
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--dirs-file", type=str, default=os.path.join(HERE, "data_dirs.txt"))
parser.add_argument("--run", type=str, default=time.strftime("own-code-%Y%m%d-%H%M"))
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--max-chars", type=int, default=30_000_000, help="cap on training text size")
parser.add_argument("--refresh-data", action="store_true", help="ignore the cached corpus and rebuild it")
parser.add_argument("--prompt", type=str, default="def ")
parser.add_argument("--temperature", type=float, default=0.8)
args = parser.parse_args()

# Settings (change ONE at a time, then log what happened)
BLOCK = 128      # how many previous characters the model can look at
N_EMBD = 128     # size of each character's vector
N_HEAD = 4       # attention heads
N_LAYER = 4      # stacked attention + MLP blocks
DROPOUT = 0.1    # randomly switches off units while training to reduce memorizing
BATCH = 64
LR = 1e-3
MIN_CHAR_COUNT = 50   # rarer characters (emoji, odd symbols) all become one "unknown" symbol
UNK = "�"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
np.random.seed(1337)
print("device:", device)

RUN_DIR = os.path.join(HERE, "checkpoints", args.run)
os.makedirs(RUN_DIR, exist_ok=True)

# 1. Data: your code, minus junk and secrets
include, exclude = data.load_or_explain(args.dirs_file)
train_text, val_text, report = data.get_corpus(
    include, exclude, max_chars=args.max_chars,
    cache_dir=os.path.join(HERE, "cache"), refresh=args.refresh_data)
print(data.format_report(report))
with open(os.path.join(RUN_DIR, "data_report.txt"), "w", encoding="utf-8") as f:
    f.write(data.full_report_text(report) + "\n")
if report["train_chars"] < 200_000 or report["val_chars"] < 10_000:
    raise SystemExit("Too little code found. Add more folders to data_dirs.txt (or check the warnings above).")

counts = Counter(train_text)
chars = sorted(c for c, n in counts.items() if n >= MIN_CHAR_COUNT)
if UNK not in chars:
    chars.append(UNK)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
unk_id = stoi[UNK]


# Fast vectorized encoder: look up every character's code point in a table
lut = np.full(0x110000, unk_id, dtype=np.uint16)
for c, i in stoi.items():
    lut[ord(c)] = i


def encode(s):
    return lut[np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)]


train_data, val_data = encode(train_text), encode(val_text)
del train_text, val_text
print(f"vocab: {len(chars)} characters | encoded train {len(train_data):,}, val {len(val_data):,}")


def get_batch(which):
    d = train_data if which == "train" else val_data
    ix = np.random.randint(0, len(d) - BLOCK - 1, size=BATCH)
    x = np.stack([d[i:i + BLOCK] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1:i + BLOCK + 1] for i in ix]).astype(np.int64)   # the same window shifted by one
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# 2. Model (identical to the first version)
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK, BLOCK)).view(1, 1, BLOCK, BLOCK))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hs = C // N_HEAD
        q = q.view(B, T, N_HEAD, hs).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature):
        for _ in range(n_new):
            logits, _ = self(idx[:, -BLOCK:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


model = GPT(len(chars)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"loss of pure guessing would be about {math.log(len(chars)):.2f}")
config = {"BLOCK": BLOCK, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD, "N_LAYER": N_LAYER,
          "DROPOUT": DROPOUT, "BATCH": BATCH, "LR": LR, "steps": args.steps}


def save(name):
    torch.save({"model": model.state_dict(), "chars": chars, "config": config},
               os.path.join(RUN_DIR, name))


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
            save("best.pt")
            marker = "  <- best so far, saved"
        print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
              f"| {time.time() - t0:5.0f}s{marker}")
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
save("last.pt")

# 4. Generate from the BEST checkpoint
model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best.pt"), map_location=device)["model"])
model.eval()
start = torch.tensor([[stoi.get(c, unk_id) for c in args.prompt]], dtype=torch.long, device=device)
out = model.generate(start, 600, args.temperature)[0].tolist()
sample = "".join(itos[i] for i in out)
with open(os.path.join(RUN_DIR, "sample.txt"), "w", encoding="utf-8") as f:
    f.write(sample + "\n")
print(f"\nbest val loss: {best_val:.3f}   (files saved in {RUN_DIR})")
print("\n- output -")
print(sample)