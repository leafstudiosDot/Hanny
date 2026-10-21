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

Six steps, in order. Steps 0-4 take about ten minutes on this machine (steps 3
and 4 are most of it); step 5 is the 80-minute Kaggle fine-tune.

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
| `hanny-code.zip` | every Kaggle-side script, `baseline.json`, the public probe sets (`pack.py` and `upload.py` stay local) | **yes** |
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
# Cell 0 - restore or train.  This REPLACES Cell 3; you do not run Cell 3 at all.
#
# /kaggle/working is wiped when the session ends, so checkpoints/ is gone every
# time. Only two things survive it: a Dataset you uploaded by hand, or a
# *previous notebook's saved output* attached via Input > Add Input (Kaggle:
# "you can then reuse that data in any future Notebook").
#
# So this cell decides for itself:
#   an adapter attached  -> restore it, do not train      (saves 67 minutes)
#   no adapter attached  -> train one                     (the 67-minute run)
#
# There is no flag to set and no cell to comment out. Commenting Cell 3 is how
# you end up with a notebook that silently skips training forever and an empty
# checkpoints/, which looks like a broken upload.
import glob, json, os, shutil, subprocess, sys

WORK = "/kaggle/working"
os.chdir(WORK)

# 1. Anything already in this session wins -- do not clobber it.
present = sorted(glob.glob(os.path.join(WORK, "checkpoints", "*",
                                        "adapter-best", "adapter_model.safetensors")))

# 2. Otherwise pull one out of the attached inputs.
attached = sorted(glob.glob("/kaggle/input/**/adapter_model.safetensors",
                            recursive=True))
for a in attached:
    run = os.path.basename(os.path.dirname(os.path.dirname(a)))
    src = os.path.dirname(os.path.dirname(a))
    dst = os.path.join(WORK, "checkpoints", run)
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
        print(f"restored {run} from attached output")

present = sorted(glob.glob(os.path.join(WORK, "checkpoints", "*",
                                        "adapter-best", "adapter_model.safetensors")))

for p in present:
    run = p.split(os.sep)[-3]
    meta = os.path.join(WORK, "checkpoints", run, "result.json")
    if os.path.exists(meta):
        j = json.load(open(meta))
        print(f"  {run:<22} loss_after={j.get('loss_after')}  "
              f"seconds={j.get('seconds')}  kernels={j.get('kernels')}")
        print(f"  {'':<22} --adapter checkpoints/{run}/adapter-best")
    else:
        print(f"  {run:<22} NO result.json -- provenance unknown, treat as suspect")

if present:
    print("\n>>> adapter available, SKIPPING the 67-minute fine-tune.")
else:
    print("\n>>> no adapter anywhere, training one now (about 67 minutes)")
    r = subprocess.run([sys.executable, "finetune.py",
                        "--run", "m4", "--epochs", "3", "--batch", "8",
                        "--eval-batch", "8", "--corpus-cache", "upload"])
    if r.returncode != 0:
        raise SystemExit(f"finetune.py failed with exit code {r.returncode}")

# Before you leave the session: Save output ON, then Save Version.
# Next time: Input > Add Input > this notebook.
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

### Cell 6b — the ceiling sweep

Cell 6 is for one document. This is the experiment: **three models, three
documents each.** Paste it once and leave it running.

Three per model is not padding. At `--repetition-penalty 1.15` the generation is
sampled, and 0.8B produced documents of 1371, 432 and 121 characters from
identical settings — an 11x spread. One document per model would be an anecdote
about a temperature, not a comparison of models.

```python
# Cell 6b - the ceiling sweep.  ~15-25 min, plus model downloads.
#
# One variable changes across the whole loop: --model. Same question, same k,
# same gate, same corpus.
#
# 4-bit is not optional here. Qwen3.5-9B in fp16 is ~18 GB and a T4 has 15 GB;
# in 4-bit it is ~5.5 GB. lorewrite.py defaults to 4-bit for exactly this.
#
# --no-score skips the per-document gate so the log stays readable. exit=0 below
# means "did not crash", NOT "passed" -- the real verdicts come from Cell 7.
import subprocess, sys, time

MODELS = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
RUNS = 3
ASK = "Who is Gurigura's master and why?"

for m in MODELS:
    tag = m.rsplit("/", 1)[-1].replace("Qwen3.5-", "").replace("-", "")
    for r in range(1, RUNS + 1):
        name = f"{tag}-{r}"
        print(f"\n{'=' * 64}\n{name}   {m}\n{'=' * 64}", flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, "lorewrite.py",
                             "--ask", ASK,
                             "--k", "6",
                             "--max-new-tokens", "900",
                             "--repetition-penalty", "1.15",
                             "--model", m,
                             "--out-name", name,
                             "--no-score"]).returncode
        print(f"  exit={rc}  {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            print(f"  {name} FAILED -- continuing with the rest", flush=True)
print("\nsweep done. Run Cell 7 to score every document.")
```

`exit=0` means it did not crash. A model can generate a fluent lie and still
exit 0 — 0.8B did exactly that, and the gate passed it. **Cell 7 is where the
result is.**

If a model name is wrong, or 9B will not fit, the loop keeps going and you get
partial results rather than nothing.

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

`finetune.py` remains single-process. `sft.py` now has a separate **DDP** path
for one run across both T4s; see [DDP (both T4s, one run)](#ddp-both-t4s-one-run)
for the command and the comparability caveat.

### Nine things that will bite you

**Cell 1 does not survive a session restart.** Kaggle's pip installs are
per-session. Start a fresh session for a new Dataset — as you must when the
archive changes — and Cell 3 dies on `ImportError: ... requires
bitsandbytes>=0.46.1` until you re-run Cell 1. That is not a broken script; it
is a missing step, and it looks identical to a dependency conflict.

**Never `pip install -r requirements.txt` on Kaggle.** It pins
`torch==2.11.0+cu128` from the cu128 index and would overwrite the
CUDA-matched torch Kaggle already ships. Cell 1 installs the M4 subset instead.

**GPU 2 sits at 0% and that is expected.** This is a single-process run; this
`finetune.py` path does not use DDP, so the second T4 is idle. Two GPUs would
roughly halve the wall clock, but that is a different recipe, not a switch. For
two concurrent `finetune.py` experiments see
[Using both T4s](#using-both-t4s); for one DDP `sft.py` run see
[DDP (both T4s, one run)](#ddp-both-t4s-one-run).

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

`PUBLISHING.md` is the release checklist. This section explains what `upload.py`
decides and what must be true before you follow that checklist. Three separate
things can be published, and only some of them are safe with your current
corpus.

**1. The adapter.** 85 MB for the gated 4B run (25 MB for a 0.8B run).
`upload.py` writes the model card, hashes the weights, and refuses a public repo
when the training data was private unless you explicitly claim the rights.

The local `checkpoints/m5-4b-v2` folder is the old `0d98c1455a2e592c` adapter;
restore and verify the gated `7720503d592cdfae` artifact first, as described in
`PUBLISHING.md`. Once it is in place:

```powershell
.venv\Scripts\python.exe upload.py --adapter checkpoints/m5-4b-v2/adapter-best --dry-run
.venv\Scripts\python.exe upload.py --adapter checkpoints/m5-4b-v2/adapter-best `
    --repo your-name/hanny-4b-chat --private --license other
```

`--dry-run` needs no token and writes `checkpoints/<run>/MODEL_CARD.md` for you
to read first. A real upload needs `hf auth login` (huggingface_hub 2.x renamed
the CLI from `huggingface-cli`). Run it locally: `upload.py` imports `pack.py`,
which is deliberately not in the Kaggle archive because it contains the
forbidden-phrase list its own scanner looks for. A private push is the safe
first upload for the old adapter or for comparison; a public push is only for
the verified gated artifact.

The refusal is not a formality. `checkpoints/m4/result.json` records a training
run over your private source corpus, so that adapter is classified
`code-private`; publishing it publicly is a licence grant you may not be able to
make. The same applies to any SFT run that includes a local file, which the
script names by basename and classifies `mixed-private`. Hub-only training data
is allowed through, with every licence printed.

**2. A dataset.** Only if you can grant rights in every row. Hub-derived data is
usually fine; anything containing `lore/` or your projects is not:

```powershell
.venv\Scripts\python.exe upload.py --dataset my_chat.jsonl `
    --dataset-repo your-name/hanny-chat-sft --private
```

**3. A merged full model.** ~8 GB for 4B (~1.5 GB for 0.8B). Merge the adapter
into the base weights and upload that if you want a single file anyone can load
without `peft`. The same provenance rule applies, and the merged model hides the
adapter's identity — so publish the adapter too, or the card describes weights
nobody can check.

**The training corpus — do not, yet.** `data.py` is built to train on your own
private source code. You can only grant rights to data you hold rights to, so
publishing the corpus means either stripping it to code you wrote entirely from
scratch, or replacing it with a public dataset such as `bigcode/the-stack`. This
is a provenance decision, not a technical one. `upload.py` scans a dataset for
the same path fragments `pack.py` uses and refuses on a hit.

Note that Kaggle Models now largely mirrors Hugging Face — pushing to the Hub
is the primary move, and Kaggle will surface the model automatically if a
notebook uses it.

### Chatting with the published weights

An adapter repo is **not** a standalone chat model. It carries
`adapter_config.json` and the LoRA weights, not the base weights. `transformers`
can load it directly (`AutoModelForCausalLM.from_pretrained("you/hanny-chat")`),
but the Hub's inference widget and the Inference Providers serve the repository
they are pointed at, and applying an arbitrary adapter is not something to
assume. Three routes actually give you a chat box:

| route | what it is | the catch |
| --- | --- | --- |
| **Gradio Space** | loads the base in 4-bit, attaches the adapter with `PeftModel`, streams replies | a free CPU Space is far too slow for 4B; a GPU/ZeroGPU Space is metered |
| **Merged model** | `merge_and_unload()` into fp16 weights, upload as a normal causal LM | 0.8B ~1.6 GB, 4B ~8 GB, 9B ~18 GB; 9B does not fit a T4 in fp16, so merge on CPU |
| **Inference Endpoint** | a dedicated container running `transformers` + `peft` | paid, and Qwen3.5's hybrid layers make TGI adapter support a bet, not a plan |

For a 4B merge, use Kaggle (29 GB RAM), not the 1650:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3.5-4B"
base = AutoModelForCausalLM.from_pretrained(
    base_id, dtype="float16", device_map={"": "cpu"})
model = PeftModel.from_pretrained(base, "checkpoints/m5-4b-v2/adapter-best")
model = model.merge_and_unload()
model.save_pretrained("merged-4b", safe_serialization=True)
AutoTokenizer.from_pretrained(base_id).save_pretrained("merged-4b")
```

```powershell
hf upload your-name/hanny-4b-chat merged-4b --private
```

`upload.py` does **not** merge or publish a merged directory — it publishes
adapters and datasets, and its provenance gate is the reason to keep using it
for those. A merged model still carries the same training-data obligation; the
`--private` above is not decoration.

**Testing without publishing:** `chat.py --ask "..."` is the whole loop. On
Kaggle, a cell can run `!python chat.py --model Qwen/Qwen3.5-4B --adapter
checkpoints/m5-4b-v2/adapter-best --ask "..."` — one reply, then exit; the REPL
does not work there because Kaggle gives no stdin. Locally it works for 0.8B,
and for 4B only at a short context, if at all.

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
do not fit, and the 4B model does not fit at all. M4 and M5 run on Kaggle T4.
`sft.py` enforces this: training outside Kaggle exits unless
`--allow-local-training` is passed for a deliberate smoke test. `--dry-run` and
`--gate-only` do not train and are not blocked by that guard.

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
| `sft.py` | M5 chat-format SFT through the model's own template; single-GPU and DDP; runs the gate |
| `chat.py` | Talk to the base model and/or named adapters, streaming, 4-bit |
| `qwenkit.py` | Shared loader, chat-template, LoRA-target and fingerprint helpers |
| `chateval.py` | Scores base-vs-adapter on chat ability and refusal behaviour |
| `upload.py` | Publishes an adapter/dataset to the Hub, refusing private data |
| `PUBLISHING.md` | Hub model/dataset commands, Kaggle archive upload, GitHub push checklist |
| `DATASETS.md` | Which public data to mix, licences, GPU budget, the 5B question |
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

## Talking to Hanny

`chat.py` is the interactive side: base model, one adapter, or several named
adapters in one process. It streams, keeps history, and prints the adapter's
weight hash and provenance before the first token, so you know what answered.

```powershell
# base model, no adapter
.venv\Scripts\python.exe chat.py

# the M4 code adapter
.venv\Scripts\python.exe chat.py --adapter checkpoints/m4/adapter-best

# two adapters, switch with /adapter
.venv\Scripts\python.exe chat.py `
    --adapter m4=checkpoints/m4/adapter-best `
    --adapter chat=checkpoints/m5/adapter-best --use chat

# one-shot, scriptable
.venv\Scripts\python.exe chat.py --adapter checkpoints/m4/adapter-best --ask "def fib(n):" --json
```

Inside the REPL: `/reset`, `/system <text>`, `/temp`, `/max`, `/thinking`,
`/adapter`, `/adapters`, `/save`, `/quit`. `--precision fp16` is available and is
a *different* measurement; never compare numbers across it.

**The M4 adapter is not a chat model.** It was trained on raw concatenated source
text, so it learned "continue this file", not "answer this question". Prompted
through the chat template it will often answer with code, a file header, or a
continuation of the prompt. `chat.py` prints a warning when the adapter's
`result.json` has no `chat_template_sha16`, which is the signature of that
training data. That is evidence about M4, not a bug in the loader.

### What this machine can run

| model, 4-bit | 4 GB GTX 1650 | Kaggle T4 (15 GB) |
| --- | --- | --- |
| **0.8B** | chat and eval, ~2.4 tok/s | chat, eval, QLoRA |
| **4B** | may load at 256–512 tokens — ~2 GB of quantised blocks plus ~1.3 GB of tied embeddings; the default 4096-token context is an OOM. Training: no. | chat, eval, QLoRA |
| **9B** | no — the untied embedding and LM head are ~4 GB in fp16 before the quantised blocks | chat, eval, tight QLoRA |

Training a 4B model needs the weights **plus** the logits tensor for a 248k
vocabulary **plus** activations and their gradients. That is a 10 GB-class step
before batch size enters the picture, which is why the 1650 is not a *slow*
training card but a card that cannot hold the tensors. Use it for `chat.py` at
0.8B and for the two gates; use Kaggle for training. `chat.py` has no graceful
OOM fallback — the process raises. The one exception is
`finetune.py --max-steps 20 --batch 1`, which fits at 1×512 to prove the code
path and is not a training run; [Hardware limits](#hardware-limits) has the
measured VRAM table.

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

It scores the same system prompt `sft.py` trains with and `chat.py` sends by
default (`qwenkit.DEFAULT_SYSTEM`), and it forces `enable_thinking=False`.
Both are part of the measurement, not presentation: Qwen3.5's 4B template
defaults to an open think block, which is not prefix-stable, and a model asked
at inference for a turn shape it never trained on is being measured on the
wrong distribution. Pass `--system ""` only to reproduce a no-system baseline.

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
  disagreed. Both are far tighter than treating the same pairs and probes as two
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

**The probe sets are a floor, not a measurement.** 50 chat pairs (10 of them
multi-turn) and 26 safety probes clear the script's own 40-pair claim threshold
and are still too small to certify general ability. Treat them as a tripwire that
catches gross regressions; the paired arithmetic says whether an observed
difference is bigger than the noise this set can resolve.

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

## Training the chatbot (M5)

M4's data was raw text. M5's is turn structure: every example is rendered
through the base model's own chat template and the loss falls only on assistant
tokens. `sft.py` does that, expands each conversation into one example per
assistant turn (Qwen's template inserts the empty think block only before the
*last* assistant message — measured, not assumed), records the template hash,
and runs `chateval.py` against the finished adapter.

```powershell
# see the mix, the supervised span, and the drop counts — no GPU
.venv\Scripts\python.exe sft.py --dry-run --run m5-plan `
    --from-hub allenai/tulu-3-sft-mixture `
    --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft `
    --max-per-source 20000

# Kaggle T4, ~1.5 hours
.venv\Scripts\python.exe sft.py --run m5 `
    --from-hub allenai/tulu-3-sft-mixture `
    --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft `
    --max-per-source 40000 --max-steps 800 --max-len 1024
```

`DATASETS.md` has the licence-verified source list, the measured
tokens-per-GPU-hour budget, the Kaggle cells, and the answer to "can we do 5B".
The short version: 0.8B is the capability constraint, mixing general data with
code is mandatory, and a bigger base is a re-baseline, not a flag.

### If the target is 4B (or 9B)

Qwen3.5 in this repo is swept at 0.8B / 2B / 4B / 9B; the family also ships a
27B, which does not fit a 15 GB T4 even at 4-bit. There is no 7B. The 4B is the
direct step up and the largest one that leaves a useful training loop inside a
3-hour session. A 7B from another family is not a drop-in: `sft.py` renders
whatever template the tokenizer provides, but the template hash, the prefix
stability and every loss number are new. Run `--dry-run` and read the
supervised span before spending a GPU-hour.

Same command, with these changes:

```powershell
.venv\Scripts\python.exe sft.py --run m5-4b-v2 `
    --model Qwen/Qwen3.5-4B `
    --from-hub allenai/tulu-3-sft-mixture `
    --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft `
    --max-per-source 40000 --epochs 1 `
    --max-len 1024 --max-tokens-per-batch 2048 `
    --max-steps 500 --eval-every 100
```

Use a new `--run` name when the code that measured the previous run changes.
The first `m5-4b` run was *measured* under a shifted-loss bug in `sft.py`, so
its recorded loss and its empty gate are not interpretable; keep the folder but
do not quote its numbers or overwrite it. `m5-4b-v2` is the comparable run.

- **`--max-tokens-per-batch 2048`, not 4096.** At 4096 tokens the logits tensor
  is ~2 GB in fp16, and the fp32 softmax workspace plus its gradient add ~6 GB
  more. That is before the 4B weights and activations. 2048 fits a 15 GB T4
  with room; 4096 is a coin flip that costs a session.
- **500 steps at 2048 tokens is ~1M tokens, roughly 2.5–3 hours.** Measured instead: 500 steps at 2048 tokens took 83 minutes on one T4,
  ending at step 500 with 873,676 train tokens seen (0.64M tokens/hour). The
  earlier estimate of ~0.4M tokens/hour was pessimistic. Do not copy the 800-step
  0.8B recipe and expect it to finish in the same session.
- **9B needs `--max-tokens-per-batch 1024`** and still buys only a few hundred
  thousand tokens in a 3-hour session. Treat it as a format nudge on a stronger
  base, not an instruction-tuning run.
- **There is no resume.** `sft.py` saves `adapter-last` every eval but cannot
  continue a schedule. `--init-adapter` continues the weights and restarts the
  LR schedule — a workaround, not a resume. Keep a run inside one 12-hour
  Kaggle session.
- **The template hash and every loss number are model-specific.** `result.json`
  records the 4B template hash; do not compare it with the 0.8B run. `sft.py`
  computes its own paired before/after loss and does not need `baseline.json`,
  but the gate still runs and is still a smoke test.

Kaggle's weekly GPU quota is commonly 30 hours (check *Session options* for the
real number) and sessions cap at 12 hours. The single-process `sft.py` path uses
one card, like `finetune.py`. The DDP path uses both T4s for **one** run; it is a
different recipe and needs a new `--run` name.

**Status: the 4B chat-SFT run has a valid loss and a completed gate; the DDP
path is implemented but its first 2-GPU validation run is still pending.**
`m5-4b-v2` trained 500 steps over 161,208 examples, processing 873,676 training
tokens (the dataset holds 89.1M), in 83 minutes.
The gated adapter is the one with weights `sha256[:16] 7720503d592cdfae`.
Held-out assistant-token loss fell 1.413946 ± 0.067077 -> 1.067992, a paired
change of -0.345954 nats (paired se 0.031177) — format learning on the training
distribution, not chat quality.

The gate (`chateval.py`, 15 chat pairs, 26 safety probes) completed with exit 0:

| measure | base | adapter | paired change | result |
| --- | --- | --- | --- | --- |
| chat response loss | 2.3612 ± 0.1238 | 2.2886 ± 0.1365 | +0.0726 (se 0.03735) | within 2x se (0.0747) — inconclusive |
| refused when it should | 0.900 | 0.900 | 0.000 (McNemar p 1.000) | 10/10 probes identical |
| answered when it should | 1.000 | 1.000 | 0.000 (McNemar p 1.000) | 16/16 probes identical |

Positive `chat_loss_change` means the adapter's loss is lower, but this change
is inside the noise floor. **Do not report it as evidence of a better chatbot.**
The safety result means these 26 probes saw no collapse; it is not a safety
audit. The earlier `m5-4b` run predates the shifted-loss fix and is not
quotable. The local `checkpoints/m5-4b-v2` adapter is still the earlier
`0d98c1455a2e592c`; the gated `7720503d592cdfae` weights exist only in the
Kaggle artifact. Identify adapters by hash, not by path.

To rerun the gate on an adapter that already exists, without rebuilding data or
spending another training run:

```powershell
.venv\Scripts\python.exe sft.py --gate-only --run m5-4b-v2 `
    --model Qwen/Qwen3.5-4B --precision "4-bit nf4"
```

For Kaggle, `hanny-notebook.ipynb` is a ready-to-import version of the same
cells. It separates the training path from the gate-only path, rejects a stale
code archive, and refuses to retrain over an existing run unless
`ALLOW_RETRAIN = True`.

### DDP (both T4s, one run)

`sft.py` has a DDP path for Kaggle's two T4s. It is **Kaggle-only by policy**:
training on the local 4 GB card is refused unless `--allow-local-training` is
passed for a deliberate smoke test. The DDP run is a new experiment, not a
faster version of `m5-4b-v2`; its effective batch and loss are different.

```powershell
# On Kaggle, from /kaggle/working, after Cell 1 and Cell 2:
python -m torch.distributed.run --standalone --nproc_per_node=2 sft.py `
    --ddp --run m5-4b-ddp --model Qwen/Qwen3.5-4B `
    --from-hub allenai/tulu-3-sft-mixture `
    --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft `
    --max-per-source 40000 --epochs 1 `
    --max-len 1024 --max-tokens-per-batch 2048 `
    --max-steps 250 --eval-every 100
```

The dataset is built once on rank 0 and shared before training; batches are
disjoint across ranks; each local loss is scaled by `local_supervised_tokens /
global_supervised_tokens * world_size` before DDP's gradient average, so the
update is global supervised-token weighted. Only rank 0 evaluates, saves and
runs the gate, while rank 1 waits at a barrier so `torchrun` keeps both workers
alive. `--grad-accum` must be 1. A 250-step DDP run processes roughly the same
global tokens as the 500-step single-GPU recipe. The first 2-GPU run is the
validation of the collectives; a single-process smoke test cannot prove them.

`hanny-notebook.ipynb` defaults to the `4b-ddp` config and adds `--ddp`
automatically. Set `4b` in Cell 3 for the single-GPU path or a v2 gate-only
rerun.

### Publishing

The gated adapter was trained only on public data (`allenai/tulu-3-sft-mixture`,
ODC-BY; `HuggingFaceH4/ultrachat_200k`, MIT), so `upload.py` classifies it as
`public-mix`. The model card must preserve the smoke-test caveat above. See
[`PUBLISHING.md`](PUBLISHING.md) for the Hugging Face model/dataset commands and
the GitHub push checklist. Those commands are instructions, not steps this
repository runs for you; git writes remain the user's decision.

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
| gate | M5-gate | General-chat ability + refusal eval (`chateval.py`) — **done; smoke test, change within noise** |
| next | M5 | Chat-template SFT harness (`sft.py`, `chat.py`) — **4B gated; DDP validation pending** |
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