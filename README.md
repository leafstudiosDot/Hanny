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
train: 549 files, 2,877,167 characters
val:   44 files, 215,664 characters (whole files held out)
skipped: minified or generated 51, possible secret 15, duplicate content 13, ...
biggest sources (to leave one out, add a line to data_dirs.txt starting with !):
  25.5%  <your own application code>
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
vocab: 99 characters | encoded train 2,877,167, val 215,664
parameters: 835,171
loss of pure guessing would be about 4.60
step     0 | train 4.765+-0.000 | val 4.773+-0.000 |     0s  <- best so far, saved
step    50 | train 2.723+-0.004 | val 2.910+-0.005 |     4s  <- best so far, saved
step   100 | train 2.570+-0.004 | val 2.748+-0.005 |     8s  <- best so far, saved
step   150 | train 2.462+-0.004 | val 2.645+-0.005 |    11s  <- best so far, saved
step   200 | train 2.362+-0.004 | val 2.534+-0.005 |    15s  <- best so far, saved

best val loss: 2.534
```

**It worked if** the loss drops well below `4.60` within 50 steps, and
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
tokenized to 59,549 tokens in 0.2s (3.62 chars per token)
64 fixed windows of 512 tokens (~1854 chars of code each)
Qwen/Qwen3.5-0.8B: 752M parameters, nothing fine-tuned yet
====================================================================
BASELINE  Qwen/Qwen3.5-0.8B  (no fine-tuning at all)
  loss         1.5098 +- 0.0889
  perplexity   4.526
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

### Step 5 — fine-tune on Kaggle (about 80 minutes)

M4 does not fit the 4GB card for a real run. See
[Training on Kaggle](#training-on-kaggle). A 20-step smoke test works locally
first:

```bash
.venv\Scripts\python.exe finetune.py --run m4-smoke --max-steps 20 --batch 1 --eval-every 10
```

It should print a `provenance check passed` line, a loss falling from `~1.60`,
and write `checkpoints/m4-smoke/result.json`.

---

## Options

### `finetune.py` — LoRA fine-tune (M4)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3.5-0.8B` | Base model to adapt |
| `--run` | timestamp | Output folder under `checkpoints/` |
| `--corpus-cache` | `cache/` | Folder holding `corpus-*.json.gz` |
| `--rebuild-corpus` | off | Walk `data_dirs.txt` instead of loading the cache |
| `--seq-len` | `512` | Tokens per window — **must match `baseline.py`** |
| `--batch` | `8` | Windows per training step |
| `--eval-windows` | `64` | **Must match `baseline.py`** or the run is rejected |
| `--eval-batch` | `1` | Raise to `8` on a 16GB GPU; eval dominates wall clock |
| `--epochs` | `3` | Passes over the training corpus |
| `--max-steps` | `0` | Stop early; `0` runs all epochs |
| `--eval-every` | `100` | Steps between validation passes |
| `--lr` | `2e-4` | LoRA learning rate |
| `--lora-r` / `--lora-alpha` | `16` / `32` | LoRA rank and scaling |
| `--seed` | `1337` | Fixed, so runs are reproducible |

---

## Training on Kaggle

M4 runs on Kaggle. Not Colab — Kaggle gives two T4s and a dataset store you can
attach directly, and you need somewhere to put the corpus.

### What to upload

The corpus, prepared locally. It is **0.80 MB** and lives in
`cache/corpus-<fingerprint>.json.gz`.

```powershell
.venv\Scripts\python.exe data.py --export-plain upload
tar -a -cf hanny-kaggle.zip upload\corpus-*.json finetune.py data.py evalkit.py utf8console.py baseline.json model.py requirements.txt
```

> **Why the corpus is uploaded uncompressed.** A `.json.gz` nested inside the
> zip is a triple compression stack, and Kaggle's dataset pipeline mangles it:
> `cache/corpus-<fp>.json.gz` came back as a *directory* called
> `corpus-<fp>.json` holding a `.json.gz.tmp` file, so `find_cached()` matched
> nothing and `finetune.py` died on a `None` path. Plain JSON skips that
> pipeline entirely — 0.8 MB becomes 3.1 MB, against 20 GB of working dir.
> Locally the cache stays gzipped; `data.py` reads both forms.

> **Keep the dataset private.** The cache contains your source code *and* 77
> folder paths plus 15 secret-file paths in its `report` block. It is not
> indexed while private. Do not make it public without stripping `report`.

Upload `hanny-kaggle.zip` as a **private Kaggle Dataset**. Name it after the
fingerprint, e.g. `hanny-corpus-c7189ca5`, so the name says which corpus a run
used — that hash changes automatically whenever the corpus does.

### Notebook cells

```python
# Cell 1 - dependencies
# Re-run this in EVERY new session. Kaggle ships its own torch, and the pip
# installs below do not persist across a session restart -- skip this cell and
# Cell 3 dies on a bare ImportError.
#
# Do NOT use `-r requirements.txt` here: it pins torch==2.11.0+cu128 from the
# cu128 index, which would replace Kaggle's CUDA-matched torch. Install the M4
# subset instead, pinned to what requirements.txt already verified.
!pip install -q "transformers==5.20.0" "accelerate==1.15.0" "huggingface-hub==2.2.0" "peft==0.21.2" "trl==1.16.0" "bitsandbytes==0.50.2"
# Optional, big speedup. Qwen3.5's gated-delta-rule layers fall back to a slow
# reference implementation without it (~7.6s/step on a GTX 1650).
!pip install -q flash-linear-attention

# Verify, because a silent -q install failure resurfaces 2 minutes into Cell 3.
from importlib.metadata import version
def _v(pkg): return tuple(int(x) for x in version(pkg).split(".")[:2])
print({p: version(p) for p in ("bitsandbytes", "transformers", "peft", "trl", "accelerate")})
assert _v("bitsandbytes") >= (0, 46), "bitsandbytes < 0.46.1: 4-bit will not load"
```

```python
# Cell 2 - unpack and check the GPU
# Kaggle auto-extracts dataset archives, so hanny-kaggle.zip is already expanded
# under /kaggle/input. The -r matters: upload/ is a directory, and a plain `cp`
# skips it with a warning while still exiting 0 -- the corpus then goes missing.
!cp -r /kaggle/input/datasets/<your-user>/hanny-corpus-c7189ca5/* /kaggle/working
import os, torch
# All three must pass before Cell 3. A missing script or corpus in
# /kaggle/working otherwise reads like a bad command rather than a bad copy.
print(sorted(os.listdir("/kaggle/working")))        # 'finetune.py' must be listed
print(sorted(os.listdir("/kaggle/working/upload")))  # ['corpus-<fp>.json']
print(os.path.getsize("/kaggle/working/upload/corpus-c7189ca503eb68f1.json"))
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))   # (7, 5) = T4 = Turing = no bf16
```

```python
# Cell 3 - train
# --corpus-cache upload: the archive ships the corpus uncompressed under upload/,
# not the local cache/ path that the flag defaults to.
!cd /kaggle/working && python finetune.py --run m4 --epochs 3 --batch 8 --eval-batch 8 --corpus-cache upload
```

```python
# Cell 4 - results
import json
r = json.load(open("/kaggle/working/checkpoints/m4/result.json"))
print(json.dumps(r, indent=2))
```

### Six things that will bite you

**Cell 1 does not survive a session restart.** Kaggle's pip installs are
per-session. Start a fresh session for a new Dataset — as you must when the
archive changes — and Cell 3 dies on `ImportError: ... requires
bitsandbytes>=0.46.1` until you re-run Cell 1. That is not a broken script; it
is a missing step, and it looks identical to a dependency conflict.

**Never `pip install -r requirements.txt` on Kaggle.** It pins
`torch==2.11.0+cu128` from the cu128 index and would overwrite the
CUDA-matched torch Kaggle already ships. Cell 1 installs the M4 subset instead.

**GPU 2 sits at 0% and that is expected.** This is a single-process run; nothing
here uses DDP or device maps, so the second T4 is idle. Two GPUs would roughly
halve the wall clock, but that is a different recipe, not a switch.

**Expect ~80 minutes, not 20.** 555 steps at 4096 tokens each on one T4. If you
see ~7-9 s/step, check the `flash-linear-attention` line Cell 1 prints: without
it the gated-delta-rule layers run a reference implementation and the run takes
several times longer.

**Kaggle mangles a `.gz` nested inside your dataset archive.** It returned a
*directory* named `corpus-<fp>.json` where the cache file should be, so
`find_cached()` matched nothing and `finetune.py` died on `None`. Upload the
corpus as plain JSON (`--export-plain`) and it never enters that code path.
Worse, because the mangled entry shares the plain file's name, you cannot
create the correct file alongside it — you need a **new Dataset**, not a
new version of the old one.

**`cp` without `-r` skips `upload/` and still exits 0.** It prints
`cp: -r not specified; omitting directory` and carries on, so the corpus is
silently absent and Cell 3 dies with `no corpus cache at None`. Use `cp -r`.

**Kaggle snapshots the input dataset at session start.** After you rebuild
`hanny-kaggle.zip`, an already-open notebook keeps reading the *old* version.
Publish the fix as a new Dataset version and start a fresh session, or Cell 2
will keep unpacking the broken archive.

**T4 is Turing, compute capability 7.5 — it has no bf16.** That is the same
generation as your GTX 1650. Any recipe that hardcodes `bf16=True` fails or
silently downcasts. `finetune.py` detects this and prints a note.

**Save the adapter or lose it.** Kaggle resets idle sessions. After Cell 3,
add the output as a new Kaggle Dataset (or commit it to your LAN Gitea, which
is only ~25 MB). The adapter is the entire fine-tune — you do not need to save
the 1.5 GB base model.

---

## Publishing to Hugging Face

For M8. Three separate things can be published, and only the third is safe with
your current corpus.

**1. The adapter (recommended first).** ~25 MB, Apache-2.0 base, LoRA weights.
Private to start — it was trained on your private code.

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("yourname/hanny-adapter-m4", repo_type="model", private=True)
api.upload_folder("checkpoints/m4/adapter-best",
                  "yourname/hanny-adapter-m4", repo_type="model")
```

**2. A merged full model.** ~1.5 GB. Merge the adapter into the base weights
and upload that if you want a single file anyone can load without `peft`.

**3. The training corpus — do not, yet.** `data.py` is built to train on your
own private source code. You can only grant rights to data you hold rights to,
so publishing the corpus means either stripping it to code you wrote entirely
from scratch, or replacing it with a public dataset such as
`bigcode/the-stack`. This is a provenance decision, not a technical one.

A model card needs at minimum: base model and licence, your LoRA config, the
corpus fingerprint, the before/after loss, and an explicit statement about what
the training data was. `finetune.py` writes most of that into
`result.json` for you.

Note that Kaggle Models now largely mirrors Hugging Face — pushing to the Hub
is the primary move, and Kaggle will surface the model automatically if a
notebook uses it.

---

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

8. **Don't import the scripts.** `model.py`, `baseline.py` and `finetune.py`
   all call `argparse.parse_args()` at module level *and then run the whole
   pipeline*, so importing one rather than executing it parses your `sys.argv`
   and starts training. `data.py` and `evalkit.py` are the only safe imports.

9. **There is no test suite.** Verification is running the scripts. `baseline.py`
   run twice must produce identical numbers; `model.py --steps 40` run twice
   must produce an identical `curve.csv`; `finetune.py` must print
   `provenance check passed` before you believe any number it produces.

10. **4-bit quantisation costs accuracy.** The same model scores 1.5098 in fp16
    and 1.5955 as 4-bit NF4 — about 0.086 nats. `finetune.py` prints its own
    `BEFORE` number from the 4-bit base; compare `AFTER` against **that**, not
    against `baseline.json`. Comparing across precisions would credit the
    fine-tune with a loss that came from quantisation.

---

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Character-level training / eval / generate, one script |
| `finetune.py` | M4 LoRA fine-tune of `Qwen3.5-0.8B` with a provenance check |
| `data.py` | Corpus collector with secret-scanning and caching |
| `evalkit.py` | The one fixed-evaluation implementation, shared by all scripts |
| `baseline.py` | Scores a pretrained model on the same held-out code |
| `utf8console.py` | Makes non-ASCII printing safe on Windows consoles |
| `gpu_check.py` | CUDA availability check and CPU-vs-GPU benchmark |
| `baseline.json` | The M3 baseline, with corpus fingerprint |
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

Success in M4 is **not** "it trained." It is `baseline.json`'s `1.5098` going
down, measured with the identical window method, by more than the noise floor
of about `0.18` (twice the `±0.089`).

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