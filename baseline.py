r"""
M3 - the baseline. Score an untouched Qwen3.5-0.8B on the same held-out code
Hanny trains on, using the same fixed evaluation windows M2 introduced.

Why this file exists
--------------------
So far "is the model any good?" meant "is the loss below ln(vocab)". That is the
score of a model that has learned *nothing*. It is the wrong reference point
here, because fine-tuning does not teach a model to code -- the model already
knows how to code. Fine-tuning teaches it *your* code, your idioms, your
naming. So the question is not "is this smart?" but "what did it already know
before I touched it?"

That number is the baseline. M4's whole job is to move it, and the only
honest way to show you moved it is to have written it down first.

Method (deliberately identical to M2's, so the two are comparable)
  - the same held-out files, from data.py's cache, so this costs no rebuild
  - evenly spaced fixed windows, never randomly resampled
  - mean +- standard error over the windows, so you can see the noise floor

Run:
    .venv\Scripts\python.exe baseline.py
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import data
import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--windows", type=int, default=64,
                    help="fixed eval windows (M2 used 256; fewer because a 248k vocab "
                         "makes each forward pass expensive)")
parser.add_argument("--seq-len", type=int, default=512,
                    help="tokens per window. 512 is the safe ceiling on a 4GB card: "
                         "1024 peaked at ~4.1GB and 4x1024 ran out of memory")
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--out", type=str, default=os.path.join(HERE, "baseline.json"))
parser.add_argument("--refresh-data", action="store_true")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# fp16, not bf16: bf16 is an Ampere (compute 8.0+) feature. This card is 7.5.
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"device: {device} | dtype: {dtype}")

# 1. The same held-out split Hanny uses. From cache, so this is instant.
include, exclude = data.load_or_explain(os.path.join(HERE, "data_dirs.txt"))
_, val_text, report = data.get_corpus(
    include, exclude, max_chars=30_000_000,
    cache_dir=os.path.join(HERE, "cache"), refresh=args.refresh_data)
print(f"held-out validation text: {len(val_text):,} chars "
      f"({report.get('val_files', '?')} files, never trained on)")

# 2. Tokenize. This is the whole point of moving off characters: the model now
#    reads ~3 characters per token, so a fixed context window covers ~3x more code.
tok = AutoTokenizer.from_pretrained(args.model)
# The tokenizer warns because we hand it 900k tokens at once, while the model's
# advertised limit is 262144. We slice into windows of --seq-len immediately
# below and never feed more than that, so the check is noise here.
tok.model_max_length = int(1e12)
t0 = time.time()
ids = tok(val_text, add_special_tokens=False)["input_ids"]
ids = np.asarray(ids, dtype=np.int64)
chars_per_token = len(val_text) / len(ids)
print(f"tokenized to {len(ids):,} tokens in {time.time()-t0:.1f}s "
      f"({chars_per_token:.2f} chars per token)")

# 3. Fixed, evenly spaced windows -- identical method to M2's eval set.
if len(ids) < args.seq_len + 1:
    raise SystemExit(f"validation text too small: {len(ids)} tokens for "
                     f"seq-len {args.seq_len}")
if args.windows < 2:
    raise SystemExit("--windows must be at least 2 (one window has no spread)")
starts = np.linspace(0, len(ids) - args.seq_len - 1, args.windows).astype(np.int64)
windows = np.stack([ids[s:s + args.seq_len] for s in starts])
print(f"{args.windows} fixed windows of {args.seq_len} tokens "
      f"(~{args.seq_len * chars_per_token:.0f} chars of code each)")

# 4. Load the untouched model and score it.
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"{args.model}: {n_params/1e6:.0f}M parameters, nothing fine-tuned yet")

x_all = torch.from_numpy(windows)

per_window = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(x_all), args.batch):
        xb = x_all[i:i + args.batch].to(device)
        loss = model(xb, labels=xb).loss.item()
        # every row has the same token count, so the batch mean IS the mean of
        # its window means -- same trick M2 used, and exact for the same reason
        per_window.extend([loss] * len(xb))
elapsed = time.time() - t0

a = np.asarray(per_window, dtype=np.float64)
mean = float(a.mean())
# A standard error needs at least two samples; numpy's ddof=1 returns NaN for n=1,
# which would silently poison baseline.json with a null noise floor.
if len(a) < 2:
    raise SystemExit(f"--windows {len(a)} cannot produce a standard error; use 2 or more")
se = float(a.std(ddof=1) / np.sqrt(len(a)))

print("\n" + "=" * 68)
print(f"BASELINE  {args.model}  (no fine-tuning at all)")
print(f"  loss         {mean:.4f} +- {se:.4f}")
print(f"  perplexity   {math.exp(mean):.3f}")
print(f"  vs M2 scale  a random guess over this vocab would be ~{math.log(len(tok)):.2f}")
print(f"  scored       {args.windows} windows x {args.seq_len} tokens "
      f"in {elapsed:.1f}s")
print("=" * 68)
print("\nPerplexity is the average number of choices the model weighs at each\n"
      "token. ~1.5 means it is nearly certain; 10 means genuinely guessing.\n"
      "Write this number down. M4's only job is to beat it.")

# 5. Save it, so M4 compares against a file rather than a memory.
result = {
    "model": args.model,
    "params": int(n_params),
    "dtype": str(dtype),
    "device": device,
    "windows": args.windows,
    "seq_len": args.seq_len,
    "chars_per_token": round(chars_per_token, 3),
    "val_chars": len(val_text),
    "val_tokens": int(len(ids)),
    "loss": round(mean, 6),
    "loss_se": round(se, 6),
    "perplexity": round(math.exp(mean), 4),
    "seconds": round(elapsed, 1),
}
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nwrote {args.out}")