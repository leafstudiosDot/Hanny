# Hanny

Two lines of work in one repo, for two different reasons.

**Hanny** — a character-level language model written from scratch in PyTorch.
No `transformers`, no external model code, no downloaded weights. Causal
self-attention, MLP blocks, embeddings and sampling are all written out in
plain `torch`. It predicts the next *character* on your own code. **This is
the teaching track:** everything here is small enough to read in one sitting.

**The token-level line** — the same held-out code, scored by a real
pretrained model (`Qwen3.5-0.8B`) that has never seen it. This is the setup
for fine-tuning: it establishes *the number fine-tuning has to beat*. **This is
the shipping track.**

This is a **training sandbox, not a library**. No package directory, no
`pyproject.toml`, no tests, no CI, no lint config. You run top-level scripts.

---

## Requirements

- Python **3.13**
- An NVIDIA GPU with a CUDA build of torch (your GTX 1650 4GB works — with
  limits, see [Hardware limits](#hardware-limits))
- [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Do not `pip install torch` from the default index.** That pulls the CPU
> build and silently replaces the CUDA one. `requirements.txt` points at the
> cu128 extra index and pins `torch==2.11.0+cu128`. After any install, confirm
> with `gpu_check.py`.

### Check your GPU first

```bash
.venv\Scripts\python.exe gpu_check.py
```

Prints the torch version, whether CUDA is visible, the GPU name, VRAM, compute
capability, and a CPU-vs-GPU matmul benchmark. **Exits 1 if no GPU is
available** — worth running before any long run, because falling back to CPU
is otherwise silent.

---

## What to do right now

Three steps, in order. Total about six minutes.

### Step 0 — confirm the GPU (2 seconds)

```bash
.venv\Scripts\python.exe gpu_check.py
```

Expect `CUDA: True`, your GPU name, and `M0.1 passed.`

### Step 1 — see what would be trained (5 seconds)

```bash
.venv\Scripts\python.exe data.py
```

Expect:

```
loaded the corpus from cache (add --refresh-data to rebuild it)
train: 2,040 files, 27,412,052 characters
val:   215 files, 2,603,687 characters (whole files held out)
skipped: not read (character cap reached) 5438, duplicate content 101, ...
biggest sources (to leave one out, add a line to data_dirs.txt starting with !):
  63.5%  <some vendored library>
```

**Read this before training** — it is how you spot a folder you did not mean
to include, and it is where you will discover whether your corpus is mostly
code you didn't write (see [Gotchas](#gotchas) item 1). Nothing is trained
here.

### Step 2 — smoke test Hanny (about 20 seconds)

```bash
.venv\Scripts\python.exe model.py --run smoke --steps 200 --eval-every 50
```

Expect roughly:

```
device: cuda
loaded the corpus from cache (add --refresh-data to rebuild it)
vocab: 100 characters | encoded train 27,412,052, val 2,603,687
parameters: 835,428
loss of pure guessing would be about 4.61
step     0 | train 4.772+-0.000 | val 4.773+-0.001 |     0s  <- best so far, saved
step    50 | train 2.757+-0.004 | val 2.815+-0.004 |     4s  <- best so far, saved
step   100 | train 2.602+-0.005 | val 2.643+-0.005 |     8s  <- best so far, saved
step   150 | train 2.524+-0.005 | val 2.573+-0.005 |    12s  <- best so far, saved
step   200 | train 2.434+-0.005 | val 2.485+-0.006 |    15s  <- best so far, saved

best val loss: 2.485
```

**It worked if** the loss drops well below `4.61` within 50 steps, and
`checkpoints/smoke/` contains `curve.csv`, `sample.txt` and `best.pt`.

Two things worth understanding immediately:

- **`4.61` is `ln(100)`** — the score of a model that has learned *nothing*.
  It is the floor, not a target.
- **The `+-` is a standard error.** If two numbers overlap by more than their
  `±`, they are the same number. This is the M2 guarantee that makes every
  later comparison meaningful.

### Step 3 — the token-level baseline (about 2.5 minutes)

```bash
.venv\Scripts\python.exe baseline.py
```

Expect:

```
tokenized to 899,874 tokens in 2.7s (2.89 chars per token)
64 fixed windows of 512 tokens (~1481 chars of code each)
Qwen/Qwen3.5-0.8B: 752M parameters, nothing fine-tuned yet
====================================================================
BASELINE  Qwen/Qwen3.5-0.8B  (no fine-tuning at all)
  loss         1.3240 +- 0.0608
  perplexity   3.758
====================================================================
```

**This is the number M4 has to beat.** It is written to `baseline.json` so
later runs compare against a file rather than a memory.

It is *not* comparable to Hanny's `4.61`. Different model, different units
(characters vs tokens), different vocabulary. They are two separate tracks.

### Step 4 — train Hanny for real (about 5 minutes)

```bash
.venv\Scripts\python.exe model.py --run hanny-v1 --steps 4000 --eval-every 250
```

Then read `checkpoints/hanny-v1/curve.csv` and `sample.txt`.

Change **one** hyperparameter at a time and log what happened. With a noise
floor of about `±0.01`, a change smaller than `~0.03` is not evidence of
anything.

---

## Options

### `model.py` — character-level Hanny

| Flag | Default | Meaning |
| --- | --- | --- |
| `--run` | timestamp `own-code-YYYYMMDD-HHMM` | Folder name under `checkpoints/` |
| `--dirs-file` | `data_dirs.txt` | Which list of folders to train on |
| `--steps` | `4000` | Number of training steps |
| `--eval-every` | `250` | Steps between validation passes |
| `--max-chars` | `30_000_000` | Cap on training text size |
| `--refresh-data` | off | Ignore the cached corpus and rebuild it |
| `--prompt` | `"def "` | Seed text the generated sample starts from |
| `--temperature` | `0.8` | Sampling temperature for generation |

### `baseline.py` — token-level baseline

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3.5-0.8B` | Any causal LM on the Hub |
| `--windows` | `64` | Fixed eval windows (minimum 2) |
| `--seq-len` | `512` | Tokens per window — **512 is the 4GB ceiling** |
| `--batch` | `1` | Windows per forward pass |
| `--out` | `baseline.json` | Where to write the result |
| `--refresh-data` | off | Rebuild the corpus cache |

---

## The model

A small decoder-only transformer, **835,428 parameters** at a 100-character
vocabulary (the embedding scales with vocabulary, so this number moves if your
corpus changes):

```
tokens  ->  [vocab=100]  ->  token embedding (128)
         ->  positional embedding (128 positions)
         ->  4 x Transformer block
                 LayerNorm
                 causal self-attention (4 heads, 128 dim)
                 residual add
                 LayerNorm
                 MLP: 128 -> 512 -> 128, GELU
                 residual add
         ->  final LayerNorm
         ->  linear head -> vocab logits
```

Hyperparameters live as constants near the top of `model.py`:

```python
BLOCK = 128      # how many previous characters the model can look at
N_EMBD = 128     # size of each character's vector
N_HEAD = 4       # attention heads
N_LAYER = 4      # attention + MLP blocks stacked
DROPOUT = 0.1
BATCH = 64
LR = 1e-3
```

**Change one at a time and log what happened.** Loss numbers are only
comparable between runs sharing the same architecture and data split.

---

## Data

Training text is **your own code**, listed one folder per line in
`data_dirs.txt` (copy [`data_dirs.tmp.txt`](data_dirs.tmp.txt);
`data_dirs.txt` is gitignored). Subfolders are walked automatically.

```
# data_dirs.txt
D:\path\to\my-project
D:\another\project
# !D:\vendored\library      <- leading ! excludes it and everything inside
```

A line starting with `!` **leaves out** that folder. This matters more than it
sounds: see [Gotchas](#gotchas) item 1.

It skips, on purpose:

- dependency and build folders (`node_modules`, `.git`, `.venv`, `dist`,
  `build`, `target`, `site-packages`, `__pycache__`, …)
- lockfiles and minified/generated output (`*.min.js`, `*.map`, `*.d.ts`, …)
- binaries, non-UTF-8 files, duplicates, files over 200 KB, or lines over
  1000 characters
- **anything that looks like it contains a secret** — private key blocks,
  AWS / Google / GitHub / Slack / Stripe / OpenAI-style keys, JWTs, credentials
  inside URLs, hardcoded password assignments, unquoted secrets in YAML, and
  secrets in Dockerfile `ENV`/`ARG`. A match skips the **entire file**; its
  content never enters the corpus.
- files whose *name* suggests secrets (`secret`, `credential`, `keystore`,
  `google-services`, `appsettings.prod`, …)

Validation is **whole files held out** (about 1 in 10, by a hash of the path),
so val loss measures generalization to code genuinely never seen.

The prepared corpus is cached in `cache/`, so later runs start in seconds. Add
code and it rebuilds automatically when the fingerprint changes; `--refresh-data`
forces it.

### A word of caution

The secret check is best-effort pattern matching, not a guarantee. Before
publishing anything trained on private code, also scan with **gitleaks** or
**trufflehog**, and rotate any key that was ever committed. `data.py` reports
the *paths and reasons* for skipped files, never their contents.

**Specific folder names are deliberately kept out of `LOG.md`.** They live in
each run's `data_report.txt`, which is local and gitignored.

---

## Checkpoints and artifacts

Each run gets **its own folder**, so runs never overwrite each other:

```
checkpoints/<run>/
  best.pt           lowest val loss seen so far
  last.pt           the final step
  sample.txt        600 characters generated from --prompt
  data_report.txt   every file used and skipped, and why
  curve.csv         step, train, train_se, val, val_se, seconds
```

Each `.pt` stores `{"model": state_dict, "chars": chars, "config": {...}}`, so
an old file explains itself when you come back in six months.

Generation loads **`best.pt`**, not `last.pt`. Evaluation runs at step 0 before
any gradient step, so a checkpoint exists even if training never improves —
meaning if you see pure noise, check the `best val loss` line before blaming
the architecture.

`checkpoints/` is gitignored and local only. Prune it yourself.

---

## Hardware limits

Measured on the GTX 1650 (4GB), for `Qwen/Qwen3.5-0.8B`:

| batch × seq | peak VRAM | |
| --- | --- | --- |
| 1 × 512 | 2.84 GB | safe |
| 1 × 1024 | 4.12 GB | over budget |
| 4 × 1024 | — | out of memory |

The cause is the **248,320-token vocabulary**. Cross-entropy over that many
classes makes the logits, not the weights, the memory bottleneck — 752M
parameters is only 1.50 GB in fp16.

**So this card can evaluate the model but cannot train it.** Optimizer states
do not fit. M4 runs on Kaggle T4.

Two more reasons Kaggle rather than local:

- Qwen3.5's layers want `flash-linear-attention`. Without it, transformers
  prints a warning and falls back to a correct but **much slower** reference
  implementation.
- `bitsandbytes` (needed for 4-bit QLoRA) is a CUDA extension that generally
  fails to build on Windows and generally works on Linux.

---

## Gotchas

The things that will waste an afternoon if you don't know them.

1. **Your corpus is probably mostly code you didn't write.** Check
   `data_report.txt`. Vendored libraries, SDKs and cloned repos are usually the
   largest single contributor, and the model will happily learn to imitate
   *them*. Use `!` prefixes in `data_dirs.txt` to exclude them. If the cap in
   `--max-chars` is being hit, files are being dropped in walk order — check
   the "not read (character cap reached)" line in the report.

2. **Val loss is now a fixed number.** Evaluation scores the same 256 windows,
   spread evenly across the split, every run. Two runs are directly comparable,
   and a change smaller than the printed `±` is noise. If you ever see a
   `NaN` standard error, more than one window was requested.

3. **Old checkpoints don't load after an architecture change.** `BLOCK`,
   `N_EMBD`, `N_HEAD` and `N_LAYER` are module-level constants, so changing
   them makes existing `.pt` files unloadable (shape mismatch). The `config`
   dict is there so you can tell what a file was trained with — it does not
   make it compatible.

4. **The vocabulary is corpus-dependent.** It is rebuilt from your files on
   every run, so changing your folders changes the vocabulary and old
   checkpoints stop loading.

5. **`transformers` 5.x renamed `torch_dtype=` to `dtype=`.** The old keyword
   is gone. Versions are pinned in `requirements.txt`.

6. **Check compiled packages against your Python's ABI.** A compiled wheel is
   named for the interpreter that built it — `_imaging.cp315-...pyd` means
   Python 3.15. If one lands in a 3.13 venv, the failure message is actively
   misleading: matplotlib reports a *circular import* inside itself, which is
   not the real problem at all. It was Pillow's `_imaging` that was unloadable.

   ```bash
   .venv\Scripts\python.exe -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot"
   ```

   Fix by rebuilding that package's stack for your interpreter — reinstall
   `matplotlib`, `pillow`, and `kiwisolver`, never `--force-reinstall` on
   everything, or you will replace the working CUDA torch build.

7. **A script can finish successfully and still exit 1.** A legacy Windows
   console codepage cannot encode some characters, and Python raises
   `UnicodeEncodeError` at *flush* time — after the run saved everything. Every
   entry point imports `utf8console.py` to prevent this. If you add a script,
   do the same.

8. **Don't import `model.py`.** It calls `argparse.parse_args()` at module
   level, so importing rather than running it parses your `sys.argv`.

9. **There is no test suite.** Verification is running the scripts. `baseline.py`
   run twice must produce identical numbers; `model.py --steps 40` run twice
   must produce an identical `curve.csv`.

---

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Character-level training / eval / generate, one script |
| `data.py` | Corpus collector with secret-scanning and caching |
| `baseline.py` | Scores a pretrained model on the same held-out code |
| `utf8console.py` | Makes non-ASCII printing safe on Windows consoles |
| `gpu_check.py` | CUDA availability check and CPU-vs-GPU benchmark |
| `baseline.json` | The M3 baseline M4 must beat |
| `LOG.md` | Build log, one section per milestone |
| `data_dirs.tmp.txt` | Template for `data_dirs.txt` |
| `AGENTS.md` | Notes for coding agents working in this repo |
| `checkpoints/` | Per-run artifacts — gitignored, local only |

---

## Where this is going

The target is a coding agent in the GLM-5 class. GLM-5 is 744B parameters
(40B active) trained on 28.5T tokens — thousands of GPUs, months, tens of
millions of dollars. **That is not reachable, and nobody reaches it from
scratch.** What *is* reachable is the part that actually makes an agent work:
the data pipeline, the evaluation, the tool schemas, and the loop.

| | Milestone | |
| --- | --- | --- |
| done | M1 | Character-level GPT from scratch |
| done | M2 | Fixed evaluation set, standard errors, `curve.csv` |
| done | M3 | Token-level baseline — the number to beat |
| next | M4 | LoRA fine-tune against that baseline (Kaggle T4) |
| | M5 | Tool-calling SFT on this project's own tools |
| | M6 | Supervised CLI agent |
| | M7 | Guardrails, autonomous loop, eval |
| | M8 | Publish — model card, dataset card, license |

Success in M4 is **not** "it trained." It is `baseline.json`'s `1.3240` going
down, measured with the identical window method, by more than the noise floor.

---

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE). Copyright 2026
leafstudiosDot.

Apache-2.0 over MIT mainly for the explicit patent grant in §3, which MIT does
not provide. The tradeoff is §4(b): anyone who modifies these files must mark
them as changed.

The license covers **the code only**. Model weights are a separate question:
`data.py` is built to train on your own private source code, so you can only
grant rights in a checkpoint to the extent you hold rights in the training
data. Keep `checkpoints/` gitignored (as it already is), and do not publish
weights trained from private code without a provenance review.

## Git

The only remote is a LAN Gitea instance. **Version-control decisions are the
user's.** `.gitattributes` declares Git LFS for `*.pt` and archives, and
checkpoints are multi-megabyte, so an accidental push is expensive and awkward
to undo. Agents working in this repo do not stage, commit, or push.