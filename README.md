# Hanny

A character-level language model written from scratch in PyTorch — no
`transformers`, no external model code, no downloaded weights. Every layer
(causal self-attention, MLP blocks, embeddings, sampling) is written out in
plain `torch` in [`model.py`](model.py).

Hanny learns to predict the **next character**. It trains on Python source code,
validates against a module it never trains on, and prints a 600-character sample
at the end of every run.

This is a **training sandbox, not a library**. There is no package directory, no
`pyproject.toml`, no tests, no CI, and no lint configuration. You run a
top-level script; that's it.

---

## Requirements

- Python **3.13** (the corpus is built from this interpreter's own stdlib source,
  so the exact version changes the data)
- An NVIDIA GPU with the CUDA 12.8 build of torch
- [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Do not `pip install torch` from the default index.** That pulls the CPU
> build and silently replaces the CUDA one. `requirements.txt` already points at
> the cu128 extra index and pins `torch==2.11.0+cu128`.

### Check your GPU first

```bash
.venv\Scripts\python.exe gpu_check.py
```

Prints the torch version, whether CUDA is visible, the GPU name, VRAM, compute
capability, and a CPU-vs-GPU matmul benchmark. **Exits 1 if no GPU is
available** — worth running before every long training run, because the
fallback to CPU is silent (see [Gotchas](#gotchas)).

---

## Quick start

```bash
.venv\Scripts\python.exe model.py --steps 3000 --eval-every 250 --prompt "def " --temperature 0.8
```

A full run prints roughly:

```
device: cuda
saving this run to: ...\checkpoints\hanny_run001.pt
train: 950,699 chars | val: 115,782 chars | vocab: 96
parameters: 834,400
loss of pure guessing would be about 4.56
step     0 | train 4.560 | val 4.564 |    1s  <- best so far, saved
step   250 | train 2.912 | val 3.104 |   19s
...
step  3000 | train 1.487 | val 2.031 |  226s

best val loss: 2.031
checkpoint: ...\checkpoints\hanny_run001.pt

- output -
def _check_validity(self, obj, name=None): ...
```

The "loss of pure guessing" line is `ln(vocab)`. It is the score a uniform
random model would get, so any val loss below it means the model learned
something.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--run-name` | `hanny` | Base name for this run's checkpoint file |
| `--steps` | `3000` | Number of training steps |
| `--eval-every` | `250` | Steps between validation passes |
| `--prompt` | `"def "` | Seed text the generated sample starts from |
| `--temperature` | `0.8` | Sampling temperature for generation |

Use `--run-name` to group and label related runs — the checkpoint file name is
derived from it.

---

## The model

A small decoder-only transformer, about **834,400 parameters**:

```
tokens  ->  [vocab=96]  ->  token embedding (128)
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

Hyperparameters live as constants at the top of `model.py`:

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
comparable between runs that share the same architecture and data split.

---

## Data

The training corpus is deliberately tiny and completely self-contained. There
are no downloads and no dataset files — the text is the source code of 14
modules from the running interpreter's stdlib, read via `m.__file__`:

```python
train_modules = [os, json, textwrap, argparse, inspect, ast, dataclasses,
                 difflib, shutil, socket, ssl, subprocess, threading, typing]
VAL_MODULE = tarfile    # never trained on
```

That is ~950k training characters and ~116k validation characters over a 96
character vocabulary (the union of the characters appearing in both splits).

`tarfile` is held out entirely, so validation loss measures generalization to
code the model has genuinely never seen. **Changing the module lists or the
split invalidates all loss comparisons** — treat it as a new baseline.

The seed is fixed at `1337`, so an identical configuration reproduces
identically.

---

## Checkpoints

Each run writes its **own** file. `free_run_path()` scans `checkpoints/` and
picks the first free `checkpoints/<run-name>_runNNN.pt`:

```
checkpoints/hanny_run001.pt
checkpoints/hanny_run002.pt
checkpoints/first-own-code_run001.pt
```

Runs never overwrite each other, so older models stay available for comparison.
Each file stores:

```python
{"model": state_dict, "chars": chars, "step": step, "val_loss": best, "config": {...}}
```

The extra keys exist so an old `.pt` explains itself when you come back to it in
six months.

**Generation loads the best-val checkpoint of the current run**, not the last
step. Evaluation also runs at `step 0`, before any gradient step, so a
checkpoint is saved even if training never improves — and that saved model is
then the random init. If the output is pure noise, check the `best val loss`
line before blaming the architecture.

`checkpoints/` is gitignored and local only. Prune it yourself; every file is
about 3.5 MB.

---

## Training on your own code

[`data.py`](data.py) collects source files from your own folders for a
local-corpus run. It currently **previews only — it does not train yet**.

```bash
.venv\Scripts\python.exe data.py
```

List one folder per line in `data_dirs.txt` (copy
[`data_dirs.tmp.txt`](data_dirs.tmp.txt); `data_dirs.txt` is gitignored).
Subfolders are walked automatically.

```
# data_dirs.txt
C:\Users\YourName\projects\my-app
D:\old-projects
```

You can also pass a different list file as the argument:

```bash
.venv\Scripts\python.exe data.py some_other_dirs.txt
```

It skips a lot on purpose:

- dependency and build folders (`node_modules`, `.git`, `.venv`, `dist`,
  `build`, `target`, `site-packages`, `__pycache__`, …)
- lockfiles and minified/generated output (`*.min.js`, `*.map`, `.d.ts`, …)
- binaries, non-UTF-8 files, duplicates, and files over 200 KB or with lines
  over 1000 characters
- **anything that looks like it contains a secret** — private key blocks, AWS /
  Google / GitHub / Slack / Stripe / OpenAI-style keys, JWTs, credentials inside
  URLs, hardcoded password assignments, unquoted secrets in YAML, and secrets in
  Dockerfile `ENV`/`ARG`. A match skips the **entire file**; its content is never
  read into the corpus.
- files whose *name* suggests secrets (`secret`, `credential`, `keystore`,
  `google-services`, `appsettings.prod`, …)

Validation is **whole files held out** (about 1 in 10, chosen by a CRC of the
path), so it matches the "hold out a module" approach of the stdlib run.

### A word of caution

The secret check is best-effort pattern matching, not a guarantee. Before
publishing any model trained on private code, also scan the source with a
dedicated tool such as **gitleaks** or **trufflehog**, and rotate any key that
was ever committed. `data.py` reports the paths and reasons for skipped files,
never their contents.

---

## Gotchas

These are the things that will waste an afternoon if you don't know them.

1. **CPU fallback is silent.** If CUDA is unavailable the script trains on CPU
   and is very slow, with no warning beyond a `device: cpu` line in the output.
   Run `gpu_check.py` first.

2. **Val loss is now a fixed number.** Evaluation scores the same 256 windows,
   spread evenly across the split, on every run — so two runs are directly
   comparable and a change smaller than the `±` printed next to it is noise.
   Each eval row is also appended to `curve.csv` in the run folder. (Before
   this, `estimate_loss()` drew 30 fresh random windows every call, so the same
   model scored differently each time.)

3. **Old checkpoints don't load after an architecture change.** `BLOCK`,
   `N_EMBD`, `N_HEAD`, and `N_LAYER` are module-level constants, so changing
   them makes existing `.pt` files unloadable (`load_state_dict` shape
   mismatch). The `config` dict inside a checkpoint is there so you can tell
   what a file was trained with — it does not make it compatible.

4. **The vocabulary is version-dependent.** It is rebuilt from stdlib source on
   every run and is currently 96 characters. A different Python version means a
   different vocab, so old checkpoints stop loading.

5. **Don't import `model.py`.** It calls `argparse.parse_args()` at module
   level, so importing it (rather than running it) parses your `sys.argv`.

6. **There is no test suite.** The only verification is running the script, or
   `gpu_check.py` before a long run.

---

## Files

| File | Purpose |
| --- | --- |
| `model.py` | The whole training / eval / generate pipeline in one script |
| `data.py` | Corpus collector for the local-code run (preview only) |
| `gpu_check.py` | CUDA availability check and CPU-vs-GPU benchmark |
| `data_dirs.tmp.txt` | Template for `data_dirs.txt` |
| `AGENTS.md` | Notes for coding agents working in this repo |
| `LICENSE` | Apache License 2.0 |
| `checkpoints/` | Model checkpoints — gitignored, local only |

`requirements.txt` also lists `numpy`, `matplotlib`, and `tensorboard`; nothing
in the current scripts imports them yet.

---

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE). Copyright 2026
leafstudiosDot.

Apache-2.0 over MIT mainly for the explicit patent grant in §3, which MIT does
not provide. The tradeoff is §4(b): anyone who modifies these files must mark
them as changed.

The license covers **the code only**. Model weights are a separate question, and
for this project they should stay local: `data.py` is built to train on your own
private source code, so you can only grant rights in a checkpoint to the extent
you hold rights in the training data. Keep `checkpoints/` gitignored (as it
already is) and do not publish weights trained from private code without a
provenance review.

There is no `NOTICE` file, and none is required: Apache-2.0 only imposes §4(d)
handling on a work that already ships one, and this repo has no third-party
attribution to record.

## Git

The only remote is a LAN Gitea instance. **Version-control decisions are the
user's** — do not stage, commit, or push from an agent session. `.gitattributes`
declares Git LFS for `*.pt` and archives, and checkpoints are multi-megabyte, so
an accidental push is expensive and awkward to undo.