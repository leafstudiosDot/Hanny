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

`pack.py` builds the archives. Do not hand-write the file list — an earlier
version of this section was a `Compress-Archive` line in the README that
silently omitted `finetune.py`, and the resulting Kaggle failure
(`No such file or directory`) looked like a broken upload rather than a broken
doc.

```powershell
.venv\Scripts\python.exe pack.py            # both archives
.venv\Scripts\python.exe pack.py --check    # verify only, builds nothing
```

Re-run this **after any change to a script, to `lore/`, or to `baseline.json`**.
The notebooks read the archive, not your working tree: an already-open Kaggle
notebook keeps running the version that was uploaded, so editing `lorewrite.py`
locally changes nothing until you rebuild and upload a new Dataset version.

Confirm what went where — this is the check that matters, because a lore file in
the publishable archive is a leak the post-build scan cannot catch (it only
reads the files it is about to ship):

```powershell
.venv\Scripts\python.exe -c "import zipfile; print('PRIVATE:'); [print(' ',n) for n in sorted(zipfile.ZipFile('hanny-data.zip').namelist())]"
.venv\Scripts\python.exe -c "import zipfile; h=[n for n in zipfile.ZipFile('hanny-code.zip').namelist() if 'lore' in n.lower()]; print('PUBLISHABLE clean' if not h else f'LEAK: {h}')"
```

It writes two archives, split by who is allowed to see them:

| Archive | Contains | Publishable? |
| --- | --- | --- |
| `hanny-code.zip` | all scripts, `baseline.json`, the public probe sets | **yes** |
| `hanny-data.zip` | the code corpus cache, the real safety probes, the **lore corpus and its retrieval probes** | **no** |

The lore corpus is the reason the split exists in the shape it does. It is other
people's copyrighted writing, so it goes in the **private** archive and never in
`hanny-code.zip`. `pack.py` treats it as an *optional* input: a clone with no
`lore/` folder still builds both archives, because "I have no lore yet" must not
mean "the packager crashes".

> **The corpus is uploaded uncompressed.** A `.json.gz` nested inside a zip is a
> triple compression stack, and Kaggle's dataset pipeline mangles it:
> `cache/corpus-<fp>.json.gz` came back as a *directory* called
> `corpus-<fp>.json` holding a `.json.gz.tmp`, so `find_cached()` matched
> nothing and `finetune.py` died on a `None` path. Plain JSON skips that
> pipeline entirely — 0.8 MB becomes 3.1 MB, against 20 GB of working dir.
> Locally the cache stays gzipped; `data.py` reads both forms.

> **`pack.py` refuses to build `hanny-code.zip` if it leaks.** It scans every
> shareable file for every folder path and source filename recorded in the
> private corpus `report`, plus the gitignored probe phrasings, then re-scans the
> finished archive's bytes — because the pre-build pass cannot see content the
> packager generates itself. It caught a real one on its first run: a company
> name in the License section below.
>
> If it flags something deliberate, such as the copyright line naming the
> holder, record it in `pack.local.json` (gitignored) rather than deleting the
> check. That file carries the reasoning too, and every entry is reprinted on
> each run, so an allowance stays visible instead of becoming something nobody
> remembers adding. Do not use it for folder paths or secret filenames — that is
> the one case where being wrong publishes someone's private source tree.

### Uploading

Upload **both** zips as a **private Kaggle Dataset**, named after the
fingerprint — `hanny-corpus-c7189ca5` — so the name says which corpus a run used.
That hash changes automatically whenever the corpus does.

Keep it private. `hanny-data.zip` contains your source code, 77 folder paths and
the names of 15 secret-bearing files. Do not make it public without stripping
the `report` block.

Attach it as **one** dataset and unpack both archives in Cell 2 — that is what
the cells below do. Splitting them into two separate datasets is also valid, and
worth doing if you want to rotate the private half without republishing code.

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
# Worth having, but measured at 1.09x, not the order of magnitude the fallback
# implies: 555 steps on a T4 went 4305s -> 3960s.
!pip install -q flash-linear-attention
# This used to be documented as "the one that actually matters", with the run
# launch-bound at ~7s/step without it. Tested: it is not. With fla installed and
# causal-conv1d ABSENT (which is the normal state here -- it builds against nvcc
# and the wheel build fails on Kaggle's image) a run finished in 3977s, 7.17
# s/step, against 3960s / 7.14 s/step for the reference run. transformers prints
# "causal_conv1d_fn is falling back to its reference PyTorch implementation";
# that warning is noise, and ~7 s/step is what this model costs on a T4.
!pip install -q causal-conv1d || true

# Verify, because a silent -q install failure resurfaces 2 minutes into Cell 3.
# The two kernel packages are checked too, because above all they are the ones
# that fail silently: Cell 1 already shipped once with causal-conv1d failing to
# build and this block reporting nothing, since it only asserted the core five.
from importlib.metadata import version
def _v(pkg): return tuple(int(x) for x in version(pkg).split(".")[:2])
print({p: version(p) for p in ("bitsandbytes", "transformers", "peft", "trl", "accelerate")})
assert _v("bitsandbytes") >= (0, 46), "bitsandbytes < 0.46.1: 4-bit will not load"
for _p in ("flash-linear-attention", "causal-conv1d"):
    try:
        __import__("fla" if _p.startswith("flash") else "causal_conv1d")
        print(f"{_p}: importable")
    except Exception:
        # causal-conv1d routinely fails to build on Kaggle and costs ~0.4% of
        # wall clock; only fla is worth chasing, and only after a kernel restart.
        print(f"{_p}: NOT importable -- {'expected here' if 'causal' in _p else 'fix: pip install, then RESTART the kernel'}")
# The gradio/diffusers "huggingface-hub 2.2.0 is incompatible" resolver warnings
# that the install printed are benign and expected: the M4 subset needs hf-hub
# 2.x, Kaggle's image ships gradio pinned to 1.x, and nothing here imports
# gradio. Do not "fix" them by downgrading huggingface-hub -- that breaks
# trl 1.16.
```

```python
# Cell 2 - unpack both archives and check the GPU
# Kaggle auto-extracts the dataset zip, so both hanny-*.zip are already
# expanded under /kaggle/input. Unpack each into /kaggle/working.
# The -r matters: upload/ and evalsets/ are directories, and a plain `cp`
# skips them with a warning while still exiting 0 -- the corpus then goes
# missing and Cell 3 dies on a None path.
!cp -r /kaggle/input/datasets/<your-user>/hanny-corpus-c7189ca5/hanny-code.zip /kaggle/working/
!cp -r /kaggle/input/datasets/<your-user>/hanny-corpus-c7189ca5/hanny-data.zip /kaggle/working/
!cd /kaggle/working && unzip -q hanny-code.zip -d . && unzip -q hanny-data.zip -d . && rm -f hanny-*.zip
import os, torch
# All four must pass before Cell 3. A missing script or corpus in
# /kaggle/working otherwise reads like a bad command rather than a bad copy.
print(sorted(os.listdir("/kaggle/working")))        # 'finetune.py' must be listed
print(sorted(os.listdir("/kaggle/working/evalsets")))
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

### The chat track — writing a cited document

Cells 3 and 4 train on code. These write a document that *cites* sources, which
is a different pipeline and a different measurement. They need **no adapter**:
the lore corpus is retrieval material, not training material, and the two
tracks share only the base model.

The lore corpus ships inside `hanny-data.zip`, so Cell 2 already put it in
`/kaggle/working/lore/`. Nothing extra to upload.

```python
# Cell 0 - restore last session's adapter.  Run this FIRST.
#
# /kaggle/working is wiped when the session ends, so checkpoints/ is gone every
# time. Only two things survive it: a Dataset you uploaded by hand, or a
# *previous notebook's saved output* attached via Input > Add Input (Kaggle:
# "you can then reuse that data in any future Notebook"). This cell copies the
# attached adapter back into /kaggle/working so that
# `--adapter checkpoints/m4/adapter-best` works unchanged, and prints exactly
# which run it is -- finetune.py is not bit-reproducible, so two adapters both
# called 'm4' exist and only loss_after tells them apart.
import glob, json, os, shutil

WORK = "/kaggle/working"
found = sorted(glob.glob("/kaggle/input/**/adapter_model.safetensors",
                         recursive=True))
print(f"adapters attached: {len(found)}")

for a in found:
    # .../<run>/adapter-best/adapter_model.safetensors
    run = os.path.basename(os.path.dirname(os.path.dirname(a)))
    src = os.path.dirname(os.path.dirname(a))          # the run directory
    dst = os.path.join(WORK, "checkpoints", run)
    if os.path.isdir(dst):
        print(f"  {run}: already present in /kaggle/working, leaving it alone")
        continue
    shutil.copytree(src, dst)
    meta = os.path.join(dst, "result.json")
    if os.path.exists(meta):
        j = json.load(open(meta))
        print(f"  {run}: restored  loss_after={j.get('loss_after')}  "
              f"kernels={j.get('kernels')}")
    else:
        print(f"  {run}: restored, but no result.json came with it -- so the "
              f"provenance of these weights is unknown. Treat as suspect.")

if found:
    run = os.path.basename(os.path.dirname(os.path.dirname(found[0])))
    print(f"\n>>> SKIP Cell 3. Use:  --adapter checkpoints/{run}/adapter-best")
else:
    print("\n>>> No adapter attached. Run Cell 3 to produce one (67 min).")
    print(">>> Before you leave this session: Save output ON, then Save Version.")
    print(">>> Next time: Input > Add Input > search for this notebook.")
```

Two things that catch people here:

- **Uploading a new `hanny-data.zip` does not wipe anything.** The session ending
  is what wipes `/kaggle/working`. Kaggle snapshots the input dataset *at session
  start*, so a new Dataset version does not affect a running session — you need a
  new session to pick it up.
- **Save output must be ticked before Save Version.** Without it the notebook has
  no output to attach, and the next session finds nothing.

```python
# Cell 5 - the retriever's instrument check
# Run this first. It is 1 second and it tells you whether the lore corpus
# actually unpacked and whether the probes shipped with it. If this is red,
# Cell 6's output means nothing -- do not read a FAIL there as a model result.
!cd /kaggle/working && python loreeval.py --retrieval
```

```python
# Cell 6 - write a document and score it
#
# --repetition-penalty 1.15 is not cosmetic. Greedy decoding produced 336
# characters of a repeated heading on the same question; 1.15 produced usable
# prose. Do not "clean this up" to greedy.
#
# --model is the ceiling test. Change ONLY this flag to compare models on the
# same question, same corpus, same k, same gate. Everything else identical.
#
# --out-name is optional but recommended: without it the document is named from
# a hash of the model and question, which is fine for the machine and
# unreadable for you. Set it and you know what Cell 6 produced.
#
# Reads /kaggle/working/lore/corpus automatically, writes the document plus a
# .json of provenance (model, question, retrieved chunks, prompt hash) to
# /kaggle/working/lore/generated/.
!cd /kaggle/working && python lorewrite.py \
    --ask "Who is Gurigura's master and why?" \
    --k 6 \
    --max-new-tokens 900 \
    --repetition-penalty 1.15 \
    --model Qwen/Qwen3.5-0.8B \
    --out-name m08b
```

```python
# Cell 7 - score every generated document
#
# There is no filename to fill in. lorewrite.py derives a document's name from a
# hash of the model and the question unless you pass --out-name, so a name typed
# by hand here is a guess -- and a guess fails with "no such document", which
# looks like a broken script rather than a typo. This cell lists what exists and
# says which question produced each one, then scores it.
#
# Exit code 1 per document = the gate failed, which is a result, not a crash.
import glob, json, os, subprocess, sys

os.chdir("/kaggle/working")
files = sorted(glob.glob("lore/generated/*.md"))
if not files:
    print("no documents yet -- run Cell 6 first")
else:
    print(f"{len(files)} document(s) in lore/generated/\n")
    for f in files:
        meta = f[:-len(".md")] + ".json"
        label = ""
        if os.path.exists(meta):
            j = json.load(open(meta))
            label = f"{j.get('model', '?')}  |  {j.get('question', '?')}"
        print("=" * 70)
        print(f)
        if label:
            print(f"  {label}")
        print("=" * 70)
        subprocess.run([sys.executable, "loreeval.py", "--doc", f])
        print()
```

To re-score **one** document, name it explicitly — the name Cell 7 just printed:

```python
!cd /kaggle/working && python loreeval.py --doc lore/generated/a72de5b7.md
```

```python
# Cell 8 - collect everything worth keeping into ONE file
#
# Run this last, before you end the session. It bundles the two things that
# cannot be recomputed cheaply -- the adapter (67 minutes to rebuild) and the
# generated documents with their provenance (the ceiling-test record) -- into a
# single zip you can download in one click.
#
# It prints exactly what went in and how big it is, because "I downloaded the
# output folder and it was missing the adapter" is a much worse afternoon than
# reading a list before you close the browser.
import os, zipfile, glob

WORK = "/kaggle/working"
OUT = os.path.join(WORK, "artifacts.zip")

# Patterns are relative to WORK and include the folder once. An earlier version
# stored (folder, pattern) pairs where the pattern ALSO began with the folder,
# and joined both -- producing /kaggle/working/checkpoints/checkpoints/... and
# an empty zip that reported itself as a success. Keep one form, not both.
PATTERNS = [
    "checkpoints/*/adapter-best/*",
    "checkpoints/*/*.json",
    "checkpoints/*/*.csv",
    "lore/generated/*",
]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for pat in PATTERNS:
        for p in sorted(glob.glob(os.path.join(WORK, pat))):
            arc = os.path.relpath(p, WORK).replace("\\", "/")
            z.write(p, arc)
            print(f"  + {arc}")

with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
size = os.path.getsize(OUT)
print(f"\n{len(names)} file(s), {size:,} bytes -> artifacts.zip")

if not any("adapter_model.safetensors" in n for n in names):
    print("\n*** NO ADAPTER IN THE BUNDLE.")
    print("*** You skipped Cell 3, or it failed. Everything else here is still")
    print("*** worth keeping, but the next session will have to retrain.")
else:
    print("\nAdapter included. Unzip locally into your repo root and Cell 0's")
    print("path (checkpoints/<run>/adapter-best) works unchanged.")

try:
    from IPython.display import FileLink
    print("\nDownload:", FileLink(OUT))
except Exception:
    print(f"\nDownload from the Output pane after saving: {OUT}")
```

**Cell 8 must be the last cell you run.** It bundles whatever is in
`/kaggle/working` at that moment, so running Cell 3 afterwards would produce a
bundle with no adapter in it.

### If the saved-version route is not working for you

Save & Run All re-runs the whole notebook from a clean session, which means it
retrains for 67 minutes — and Cell 0 still reports `adapters attached: 0`,
because Cell 0 runs *before* Cell 3 and there was nothing attached to that fresh
session in the first place.

**Do not use `Expand-Archive -Force` for this.** It will silently swap one set
of weights for another. This repo can hold two adapters both called `m4` —
`finetune.py` is not bit-reproducible, and re-running overwrites the folder
without asking — so `-Force` destroys the record of which run produced which
weights, and `chateval.py` will later hash whichever survived.

```powershell
.venv\Scripts\python.exe pack.py --restore artifacts.zip
```

That refuses to overwrite anything that already exists, prints the provenance of
whatever it restores, and gives you the ready-to-paste `--adapter` path:

```
provenance of what was restored:
  kaggle-m4    loss_after=1.325564  seconds=3936.6
               kernels={'flash_linear_attention': '0.5.2', 'causal_conv1d': False}
               use: --adapter checkpoints/kaggle-m4/adapter-best
```

It also rejects a zip carrying anything outside `checkpoints/` and
`lore/generated/`. Your lore corpus is hand-edited and must never be
overwritten from an import.

If a collision is what you want — you know the new weights are the ones to keep:

```powershell
.venv\Scripts\python.exe pack.py --restore artifacts.zip --force
```

An adapter that arrives without its `result.json` is restored but flagged:
usable, but its provenance is unknown, and an unknown-provenance adapter is not
evidence of anything.

### Not re-running 67 minutes

The fine-tune is 67 minutes; everything else in this notebook is seconds or two
minutes. That asymmetry decides the whole workflow: **the adapter is the only
artifact expensive enough to be worth persisting.**

`/kaggle/working` is wiped when the session ends — 20 GB of it, and only what
you asked to save. Two things survive it:

1. **A previous notebook's output**, attached with *Input → Add Input → search
   for a Notebook*. This is the one to use: no download, no re-upload, and
   Kaggle indexes it alongside Datasets and Models.
2. **A Dataset** you upload and version yourself.

Enable *Save output* on **Save Version**, then attach that notebook to the next
session. Cell 0 reports what arrived and whether to skip Cell 3.

What is **not** worth persisting, because it costs nothing to recompute:

| artifact | why not |
| --- | --- |
| retrieved chunks | BM25 over 140 claims takes **under a second** and is deterministic — re-running gains nothing |
| the corpus | already ships inside `hanny-data.zip` |
| the prompt | it lives in `lorewrite.py` and ships with the code |
| `loreeval` verdicts | seconds to recompute, and they must be recomputed to be trusted |

What *does* accumulate value across sessions is the **documents themselves** —
each one with the `result.json` recording its model, question and prompt hash.
That pair is a row in the ceiling-test comparison table, and the failing ones
are the interesting ones: they are the cases where a bigger model can be shown
to have helped. Keep them.

One caution about persisting the adapter: `finetune.py` is not bit-reproducible,
so a re-trained "m4" is different weights that share its name. That is why
`chateval.py` hashes the adapter it scores, and why you should record the
`loss_after` next to it — Cell 0 prints it for exactly this reason.

**A FAIL here is the normal outcome, and the reasons are the finding.** The
three checks are string-level and cannot tell a correct reading of a source from
a fluent misreading of it. A model can invent a plausible mechanic, use only
words that genuinely appear in your corpus, and pass every check — that has
already happened. Read the document; the gate is a floor, not a verdict.

To collect results, enable **Save output → Save and run all**, then download
`/kaggle/working/lore/generated/`. Each document sits next to the `.json`
recording exactly which chunks produced it.

### Using both T4s

`finetune.py` is single-process: it binds one process to one GPU and never
looks at the second. That is a property of the script, not a Kaggle limit.
`CUDA_VISIBLE_DEVICES` remaps a physical card to `cuda:0`, so no code change is
needed to use the second one.

**Do the `flash-linear-attention` fix first.** Both runs are bottlenecked on
the CPU, not on GPU throughput — a run using ~1 GB of a 15 GB card at 7 s/step
is launch-bound, and that is exactly why `fla` buys 1.09x rather than 5x: it
replaces one dispatch path, it does not make the run compute-bound. Two slow
runs are still two slow runs. At ~66 minutes per run, running two at a time is
obviously worth it.

Because the repo's rule is *one hyperparameter at a time*, two GPUs are worth
more as **two concurrent experiments** than as one run at half the time:

```python
# Two runs, one per T4. The --run names MUST differ: both processes otherwise
# write the same checkpoints/<run>/ directory and corrupt each other's adapter.
!cd /kaggle/working && CUDA_VISIBLE_DEVICES=0 nohup python finetune.py \
    --run m4-2ep  --epochs 2 --batch 8 --eval-batch 8 --corpus-cache upload \
    > /kaggle/working/log-2ep.txt 2>&1 &
!cd /kaggle/working && CUDA_VISIBLE_DEVICES=1 nohup python finetune.py \
    --run m4-lr3e4 --epochs 3 --batch 8 --eval-batch 8 --lr 3e-4 --corpus-cache upload \
    > /kaggle/working/log-lr3e4.txt 2>&1 &
```

`nohup` and the log redirect are not decoration: without them the cell returns
with no way to see progress, and a failure in run two is indistinguishable from
a hang. Watch them with:

```python
!sleep 180; tail -n 6 /kaggle/working/log-2ep.txt /kaggle/working/log-lr3e4.txt
```

Then read both results together:

```python
import json
for run in ("m4-2ep", "m4-lr3e4"):
    r = json.load(open(f"/kaggle/working/checkpoints/{run}/result.json"))
    print(f"{run:9} before {r['loss_before']:.4f} -> after {r['loss_after']:.4f} "
          f"(change {r['loss_change']:+.4f}, {r['seconds']:.0f}s)")
```

**Comparing two runs is legitimate here, and only because of `evalkit.py`.**
`loss_before` is re-measured per run, but from the same fixed windows on the
same corpus, so it should come out identical across runs — if it does not,
stop and find out why before trusting either number. Vary `--lr`, `--epochs`,
`--lora-r` freely. Do **not** vary `--corpus-cache`, `--eval-windows` or
`--seq-len` between concurrent runs: different windows make the two `AFTER`
values incomparable in exactly the way `evalkit.py` exists to prevent.

Two 4-bit models at ~1 GB each fit on separate cards with room to spare, so
there is no VRAM contention to worry about.

**`lorewrite.py` also works two-at-a-time**, and the same rule applies with one
extra wrinkle: `--model` is the *point* of the comparison, so vary nothing else
between the two cards.

```python
# The ceiling test, both T4s, same question. One variable: --model.
!cd /kaggle/working && CUDA_VISIBLE_DEVICES=0 nohup python lorewrite.py \
    --ask "Who is Gurigura's master and why?" --k 6 --repetition-penalty 1.15 \
    --model Qwen/Qwen3.5-0.8B  --out-name m08b \
    > /kaggle/working/log-m08b.txt 2>&1 &
!cd /kaggle/working && CUDA_VISIBLE_DEVICES=1 nohup python lorewrite.py \
    --ask "Who is Gurigura's master and why?" --k 6 --repetition-penalty 1.15 \
    --model Qwen/Qwen3.5-9B    --out-name m9b \
    > /kaggle/working/log-m9b.txt 2>&1 &
```

`--out-name` is required here. Without it both processes derive the same
filename from the same model-and-question pair and overwrite each other's
document — the lore equivalent of two `finetune.py` runs sharing a
`checkpoints/<run>/` directory.

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
halve the wall clock, but that is a different recipe, not a switch — see
[Using both T4s](#using-both-t4s) for the cheap version, which is two
concurrent experiments rather than one faster one.

**Expect ~65 minutes, and do not go hunting for the cause if you see ~7
s/step.** 555 steps at 4096 tokens each on one T4 is a ~66 minute run on this
model, and that is what it costs with the fast kernels in place. `fla` is worth
about 1.09x; `causal_conv1d` is worth about 0.4% and usually cannot be built on
Kaggle at all, so the `causal_conv1d_fn` fallback warning from transformers is
expected noise rather than a diagnosis. `finetune.py` records which of the two
were actually importable in `result.json` under `kernels`, so a slow run can be
compared against a fast one honestly.

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

**Kaggle snapshots the input dataset at session start.** After you rebuild the
archives with `pack.py`, an already-open notebook keeps reading the *old* version.
Publish the fix as a new Dataset version and start a fresh session, or Cell 2
will keep unpacking the stale archive.

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
  falls back to a correct reference implementation — measured at **1.09x**
  slower, not the order of magnitude the fallback suggests, because the run is
  launch-bound on the CPU either way.
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
| `chateval.py` | Scores base-vs-adapter on chat ability and refusal behaviour |
| `pack.py` | Builds the upload archives and refuses to publish a leak |
| `data.py` | Corpus collector with secret-scanning and caching |
| `evalkit.py` | The one fixed-evaluation implementation, shared by all scripts |
| `baseline.py` | Scores a pretrained model on the same held-out code |
| `utf8console.py` | Makes non-ASCII printing safe on Windows consoles |
| `gpu_check.py` | CUDA availability check and CPU-vs-GPU benchmark |
| `evalsets/` | The probe sets `chateval.py` scores — see its [README](evalsets/README.md) |
| `baseline.json` | The M3 baseline, with corpus fingerprint |
| `lorekit.py` | Loads the lore corpus into claim-sized chunks and retrieves with BM25 |
| `loreeval.py` | Scores a generated document: citations, proper nouns, numbers |
| `lorewrite.py` | Retrieve → prompt → generate → score, in one command |
| `lore/` | The lore corpus and its format contract. **Gitignored** — see [lore/README.md](lore/README.md) |
| `LOG.md` | Build log, one section per milestone |
| `data_dirs.tmp.txt` | Template for `data_dirs.txt` |
| `AGENTS.md` | Notes for coding agents working in this repo |
| `checkpoints/` | Per-run artifacts — gitignored, local only |

---

## Checking a fine-tune did not break the chatbot

M4 improved code perplexity and said nothing at all about whether the model can
still hold a conversation or refuse anything. `chateval.py` is the gate that
answers that, and it runs on the 1650 — no Kaggle needed.

```bash
# once, for the reference every later run is compared against
.venv\Scripts\python.exe chateval.py

# base vs a saved adapter, in one process at one precision
.venv\Scripts\python.exe chateval.py --adapter checkpoints/m4/adapter-best --show-probes 3
```

Three things it will not let you do:

- **Compare across precision.** Base and adapter are loaded and scored in the
  same process at the same precision. A 4-bit adapter against an fp16 baseline
  is the M4 mistake all over again.
- **Read one refusal number.** It prints *refused when it should* and *answered
  when it should* separately, because a model that refuses everything scores a
  perfect refusal rate and is useless.
- **Believe a difference smaller than `2xse`.** Both comparisons are **paired**:
  the same items, the same rendered inputs, one process. The chat verdict uses
  the standard error of the *difference* (`evalkit.paired_se`), and the refusal
  verdicts use McNemar's exact test on the probes where the two models actually
  disagreed. Both are far tighter than treating 15 pairs and 26 probes as two
  independent samples would be — the shared part of the two models' errors
  cancels out in a difference, and concordant refusal probes say nothing about
  whether the models differ.

An earlier version of this script got both wrong in the same direction: it
printed the adapter's own standard error next to the base-minus-adapter
difference, and sized the refusal rates' error bars with 26 probes when one rate
was over 10 harmful prompts. Both mistakes made the gate unable to resolve a real
effect. `LOG.md` records the arithmetic.

And it identifies the adapter by its **weights**, not its path: `result.json`
carries `adapter_fingerprint` with `sha256[:16]` of `adapter_model.safetensors`
plus the producing run's `loss_after` and `kernels`. `finetune.py` is not
bit-reproducible — a repeat of the same config moved `loss_after` by 0.00023
nats and produced different weights — so re-running M4 overwrites
`adapter-best/` with a different model that still calls itself `m4`. A chat-eval
number that recorded only the path would quietly describe an adapter that no
longer exists.

**The probe sets are smoke tests, not measurements.** 15 chat pairs and 26 safety
probes is far too few to support a claim about general ability, and the script
says so on every run. Treat it as a tripwire that catches gross regressions, and
grow `evalsets/` before trusting anything finer.

For refusal specifically, "too few" has a precise meaning worth knowing before
you trust a clean result: the paired exact test needs **6 discordant probes
pointing the same way** before it reaches p ≤ 0.05, and 5 gives 0.0625. With 10
harmful probes that means **6 of them have to flip**. Detecting a 20-point
refusal collapse therefore needs roughly 30 harmful probes, not 10 — so today the
gate will miss a fine-tune that erodes refusal on a third of its harmful probes.
It catches total collapse; it does not measure erosion.

Refusal detection is substring matching, which is a proxy in both directions: it
misses refusals phrased unusually, and it will call a helpful answer a refusal if
it opens with an apology. It was wrong once already — the first marker list
reported a safety regression that was entirely the matcher's fault. Use
`--show-probes` and read the replies before believing any refusal number.

### The safety probes are not in git, on purpose

`evalsets/safety_probe.jsonl` holds real red-team phrasings and is **gitignored**.
`evalsets/safety_probe_public.jsonl` keeps the same harm categories with the
wording abstracted, and is committed.

The reason is that a short functional string is harmless on its own, but a
repository full of them reads as what the Hugging Face Content Policy lists
under Restricted Content → Harmful or Abusive Content — *"content that attempts to generate malicious code"* — with
account suspension among the listed sanctions. Intent is not something an
automated flag or a casual reader of a diff gets to assess.

Practically: `chateval.py` prefers the real file, falls back to the redacted one,
prints which it used, and records `safety_probe_kind` in `result.json`. **The
two are not comparable** — `[probe: *]` is not a refusal probe, the
model just answers the literal string — so a public-set run's refusal numbers
should be ignored. It still ships to Kaggle: `pack.py` builds `hanny-data.zip`
from an explicit file list, not a directory sweep, so the gitignore rule does
not apply. Full reasoning in [`evalsets/README.md`](evalsets/README.md).

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
| done | M4 | LoRA fine-tune against that baseline (Kaggle T4) |
| gate | M5-gate | General-chat ability + refusal eval (`chateval.py`) — **done** |
| next | M5 | Chat-template training data, mixed with general instruction data |
| | M5b | Tool-calling SFT on this project's own tools |
| | M6 | Supervised CLI agent |
| | M7 | Guardrails, autonomous loop, eval |
| | M8 | Publish — model card, dataset card, license |

**M4 result: `1.5955 → 1.3256`, a drop of `0.2699` nats** (perplexity
`4.931 → 3.764`) over 555 steps, best at step 500. That is `4.96x` twice the
`±0.0272` standard error, so it clears the noise floor by a wide margin.

Success in M4 was defined **not** as "it trained," but as a drop larger than
`2x` the standard error, measured as `loss_before` vs `loss_after` *from the
same 4-bit run* — never against `baseline.json`'s full-precision `1.5098`. That
distinction is not pedantic here: the 4-bit quantisation gap is `0.0857` nats,
which is a third of the entire measured effect. Comparing against the
full-precision baseline would have reported a drop of `0.1842` and called it a
narrow pass against a `0.178` threshold — the right answer, arrived at by a
method that would equally have "passed" a model that only recovered the
quantisation error and learned nothing.

---

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE). Copyright 2026
leafstudiosDot.

Built and published by leafstudiosDot. That name appears here on purpose: it is
the copyright holder, and an unambiguous attribution is what makes a licence
work. It is recorded in `pack.local.json` so the packaging leak scan knows the
difference between deliberate attribution and a folder path that leaked out of
the corpus — a distinction `pack.py` cannot make on its own.

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