# Training data for M5 and beyond

`finetune.py` (M4) learns *your code* from raw text. That is the right data for a
code adapter and the wrong data for a chatbot: a model trained on concatenated
files learns to continue files. `sft.py` (M5) is the other half — real turn
structure, rendered through Qwen's own chat template, with loss on assistant
tokens only.

This file answers two questions: **which public data to mix in**, and **how much
of it a Kaggle GPU can actually consume**. Licence values below were read from
the Hub dataset cards on 2026-10-19; re-check them before publishing anything,
because a dataset card can change.

---

## The short version

| | |
| --- | --- |
| General chat / instructions | `allenai/tulu-3-sft-mixture` (ODC-BY) |
| Multi-turn conversation | `HuggingFaceH4/ultrachat_200k` (MIT) |
| Multi-turn, permissive alternative | `HuggingFaceTB/smol-smoltalk` (Apache-2.0) |
| Short human-written instructions | `databricks/databricks-dolly-15k` (CC BY-SA 3.0) |
| Code (public, permissive) | `BAAI/TACO` (Apache-2.0) |
| Tool calling (M5b, not yet supported by `sft.py`) | `Salesforce/xlam-function-calling-60k` (CC BY 4.0, gated) |

Start with **tulu + ultrachat**. That is enough to test whether the pipeline
works and whether the model learns turn structure at all. Add code only after
the gate shows the chat behaviour survived, and keep code a minority of the mix.

---

## Why the mix matters more than the size

`AGENTS.md` states the failure mode and it is measured, not hypothetical: narrow
SFT erodes general ability, and raw code SFT erodes refusal behaviour. A mix
that is 90% code will produce an adapter that is better at code and worse at
being a chatbot, and the code loss will not show the second half at all.

A defensible first mix, by tokens rather than rows:

| share | content | purpose |
| --- | --- | --- |
| ~70% | tulu-3-sft-mixture | broad instruction following, some code and math already inside it |
| ~25% | ultrachat_200k | multi-turn conversation, the thing a chat model is judged on |
| ~5% | dolly-15k | short human-written prompts; cheap diversity, not scale |

Then, if you want the agent direction, add tool-calling data as a **separate**
M5b run in the template's native tool format. Do not mix today's tool datasets
into this run: `sft.py` drops rows containing `tool`/`function` roles on purpose,
because rendering them as plain text teaches a syntax the template never asks
for. Dropping them is counted in `data_report.json`, so the omission is visible.

---

## Dataset notes and licence traps

- **`tatsu-lab/alpaca` is CC BY-NC 4.0.** Non-commercial. Do not publish an
  adapter trained on it with an Apache-2.0 tag. It is a fine local experiment.
- **`databricks/databricks-dolly-15k` is CC BY-SA 3.0.** Share-alike. A dataset
  you redistribute that contains it inherits the obligation; a model's licence is
  a separate but related question. Keep it a small share or leave it out of a
  public release.
- **`HuggingFaceTB/smoltalk` declares no licence field** on its card at the time
  of writing. Its sibling **`HuggingFaceTB/smol-smoltalk` is Apache-2.0** and uses
  the same `messages` schema, so use that one if you want an openly licensed
  multi-turn source. No declared licence is not the same as a permissive one.
- **`teknium/OpenHermes-2.5` declares no licence field either.** It is widely
  used, but "widely used" is not a licence. Keep it out of a public model.
- **`OpenAssistant/oasst2` is Apache-2.0 but is a message tree**, not rows of
  conversations. `sft.py` does not assemble trees, so it will read it as an
  unsupported format. Use a chatml conversion of it, or another dataset.
- **`Salesforce/xlam-function-calling-60k` is gated** and CC BY 4.0. You must
  accept the terms on the dataset page and be logged in. It is for M5b.
- **`BAAI/TACO` is Apache-2.0** but its rows are programming problems with
  solutions, not conversations. Converting them into user/assistant turns is
  M5b/M6 work; `sft.py` does not do it.

All of this is a *licensing* statement, not legal advice. The rule that matters
here: you can only grant rights you hold, and `upload.py` refuses to pretend
otherwise.

---

## Local data: your lore and your projects

`sft.py --data file.jsonl` accepts the same `messages` schema. This is where your
own notes, lore and project conventions go — and it is the one part of the mix
that `upload.py` will refuse to publish unless you supply `--rights-cleared`.

Schema (one JSON object per line):

```json
{"messages": [{"role": "user", "content": "Why are outlaws cursed?"},
              {"role": "assistant", "content": "..."}]}
```

Also accepted, and normalised: `{"conversations": [{"from": "human", "value":
"..."}, {"from": "gpt", "value": "..."}]}` and `{"prompt": "...", "response":
"..."}` / `{"instruction": "...", "input": "...", "output": "..."}`.

If you have lore as prose rather than Q&A, generate pairs from it *with the
source attached* and check them. A hallucinated question/answer pair teaches a
hallucination; this is the same faithfulness problem `loreeval.py` exists for.

---

## Commands

Dry run first. It builds the dataset, prints the mix and a decoded sample with
the supervised span marked, and writes `checkpoints/<run>/data_report.json`
without touching a GPU:

```powershell
.venv\Scripts\python.exe sft.py --dry-run --run m5-plan `
    --from-hub allenai/tulu-3-sft-mixture `
    --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft `
    --max-per-source 20000 --max-len 1024
```

Then a real run (Kaggle T4, ~1.5 hours for the step count below):

```powershell
.venv\Scripts\python.exe sft.py --run m5 `
    --from-hub allenai/tulu-3-sft-mixture `
    --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft `
    --max-per-source 40000 --epochs 1 `
    --max-len 1024 --max-tokens-per-batch 4096 `
    --max-steps 800 --eval-every 100
```

Add `--data my_chat.jsonl` alongside the Hub sources to mix your own turns in.
`--max-per-source` is the only mixing knob: equal caps give the datasets equal
row counts, not equal token counts. Read the token line in the report, not the
row counts.

For a mix that is entirely Apache-2.0/ODC-BY, swap `ultrachat_200k` for
`HuggingFaceTB/smol-smoltalk` (Apache-2.0, same `messages` schema). That avoids
the MIT attribution question if you care about a single permissive story; it
does not change the fact that the weights are a derivative of the base model's
licence.

On a 16 GB T4, `--max-len 2048 --max-tokens-per-batch 4096` truncates fewer
answers than 1024 at the same token count. Logits cost scales with tokens per
step; attention memory scales with sequence length, so 2x2048 is heavier than
8x512. Measure it on the free T4 before committing a long run.

---

## Budget: what a Kaggle GPU actually consumes

Measured, not estimated: M4 ran 555 steps of 8x512 tokens in 64 minutes on a T4,
about **2M tokens per GPU-hour** and ~7 s/step at 4096 tokens.

| steps at 4096 tok | tokens seen | T4 wall clock |
| --- | --- | --- |
| 400 | 1.6M | ~45 min |
| 800 | 3.3M | ~1.5 h |
| 1200 | 4.9M | ~2.3 h |
| 3000 | 12.3M | ~6 h |

Kaggle sessions are limited (9–12 h depending on verification), and the GPU is
not guaranteed on a re-run. Plan a run you can finish, save `adapter-best` every
eval, and download `artifacts.zip` at the end rather than assuming the session
survives.

Twelve million tokens is roughly one pass over 30–50k conversations. That is
enough to move the *format*; it is not enough to teach a new domain, and no
number of epochs fixes that.

---

## Kaggle cells for M5

Build the archives **after** the latest code change. A dataset version that
predates `sft.py` fails at Cell 2 with `missing ['sft.py']`, and that reads like
a broken notebook rather than a stale upload.

```powershell
.venv\Scripts\python.exe pack.py
```

Upload both zips to the same **private** Kaggle Dataset, add a new version, and
restart the notebook session so `/kaggle/input` sees it. Notebook settings:
**GPU T4 x2** (one card is used), **Internet on**, **Persistence: Files only**.

Cell 1 — dependencies. Re-run in every new session. Do **not** install torch and
do **not** `pip install -r requirements.txt`: it pins a cu128 torch that replaces
Kaggle's CUDA-matched build.

```python
!pip install -q "transformers==5.20.0" "accelerate==1.15.0" \
    "huggingface-hub==2.2.0" "peft==0.21.2" "bitsandbytes==0.50.2" \
    "datasets==5.1.0"
# Optional: measured at 1.09x on the 0.8B run; the reference path is correct.
!pip install -q flash-linear-attention || echo "fla unavailable -- reference path is fine"

from importlib.metadata import version
print({p: version(p) for p in
       ("transformers", "peft", "bitsandbytes", "datasets",
        "accelerate", "huggingface-hub")})
assert tuple(int(x) for x in version("bitsandbytes").split(".")[:2]) >= (0, 46)
```

Cell 2 — unpack both archives into `/kaggle/working` and verify. This searches
`/kaggle/input` instead of hardcoding a slug.

```python
import glob, os, subprocess, torch

ROOT = "/kaggle/working"
for name in ("hanny-code.zip", "hanny-data.zip"):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        print("mounted:", glob.glob("/kaggle/input/*"))
        raise SystemExit(f"{name} not found; attach the dataset or fix the path")
    print("unzip", hits[0])
    subprocess.run(["unzip", "-oq", hits[0], "-d", ROOT], check=True)

required = ["sft.py", "chat.py", "qwenkit.py", "chateval.py",
            "evalsets/chat_probe.jsonl"]
missing = [r for r in required if not os.path.exists(os.path.join(ROOT, r))]
if missing:
    raise SystemExit(f"missing {missing}; rebuild hanny-code.zip with pack.py")
print("safety probes:",
      "private" if os.path.exists(f"{ROOT}/evalsets/safety_probe.jsonl")
      else "PUBLIC FALLBACK (weaker gate)")
print("corpus:", glob.glob(f"{ROOT}/upload/corpus-*.json"))
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))   # (7, 5) = T4 = no bf16
```

Cell 3 — pick the run. Do `0.8b` once to validate the pipeline cheaply; `4b` is
the target.

```python
CONFIG = {
    "0.8b": dict(model="Qwen/Qwen3.5-0.8B", run="m5",    tokens=4096, steps=800),
    "4b":   dict(model="Qwen/Qwen3.5-4B",  run="m5-4b", tokens=2048, steps=500),
}["4b"]        # <-- change to "0.8b" for the cheap first pass
MODEL, RUN = CONFIG["model"], CONFIG["run"]
print(CONFIG)
```

Cell 4 — data check, no weights and no GPU spend. 2000 per source is enough to
see the mix and the supervised span; the training cell streams the full cap.

```python
import subprocess
subprocess.run([
    "python", "sft.py", "--dry-run", "--run", RUN + "-plan",
    "--model", MODEL,
    "--from-hub", "allenai/tulu-3-sft-mixture",
    "--from-hub", "HuggingFaceH4/ultrachat_200k::default::train_sft",
    "--max-per-source", "2000", "--max-len", "1024",
], cwd="/kaggle/working", check=True)
```

Cell 5 — train. About 1.5 h for 0.8B, about 3 h for 4B, plus the gate. A
non-zero `sft.py` exit is usually the gate failing, not training dying; read the
gate output and keep the adapter either way.

```python
import subprocess, time
t0 = time.time()
rc = subprocess.run([
    "python", "sft.py", "--run", RUN, "--model", MODEL,
    "--from-hub", "allenai/tulu-3-sft-mixture",
    "--from-hub", "HuggingFaceH4/ultrachat_200k::default::train_sft",
    "--max-per-source", "40000", "--epochs", "1",
    "--max-len", "1024",
    "--max-tokens-per-batch", str(CONFIG["tokens"]),
    "--max-steps", str(CONFIG["steps"]),
    "--eval-every", "100",
], cwd="/kaggle/working").returncode
print(f"sft.py exit={rc}  {(time.time()-t0)/60:.1f} min")
```

Cell 6 — one-shot chat against the adapter that just trained. Kaggle has no
stdin, so this is `--ask`, not the REPL.

```python
!cd /kaggle/working && python chat.py --model {MODEL} \
    --adapter checkpoints/{RUN}/adapter-best \
    --ask "Explain a hash table in three sentences, then give one Python example." \
    --max-new-tokens 160 --no-stream
```

Cell 7 — read the numbers and bundle the adapter, gate and curve.

```python
import json, os, zipfile
run_dir = f"/kaggle/working/checkpoints/{RUN}"
r = json.load(open(os.path.join(run_dir, "result.json"), encoding="utf-8-sig"))
print(json.dumps(r, indent=2))
gate = os.path.join(run_dir, "gate", "result.json")
if os.path.exists(gate):
    print("\nGATE:")
    print(json.dumps(json.load(open(gate, encoding="utf-8-sig")), indent=2))

out = f"/kaggle/working/{RUN}-artifacts.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(run_dir):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, "/kaggle/working"))
names = zipfile.ZipFile(out).namelist()
assert any(n.endswith("adapter_model.safetensors") for n in names), "no adapter in zip"
print(f"\n{len(names)} files, {os.path.getsize(out)/1e6:.1f} MB -> {out}")
```

Download `m5-4b-artifacts.zip` from the Output pane after **Save Version**, then
locally:

```powershell
.venv\Scripts\python.exe pack.py --restore m5-4b-artifacts.zip
.venv\Scripts\python.exe chat.py --adapter checkpoints/m5-4b/adapter-best
```

`pack.py --restore` refuses to overwrite an existing run; use `--force` only
when you mean to replace it. `result.json["gate"]` is a path, not a number; read
the gate file, and do not quote `loss_after` as chat quality.

---

## The 5B question

Two different questions hide inside "can we do 5B".

**Fine-tuning a 4B–9B base with QLoRA on Kaggle: yes, with caveats.** The
weights are not the problem — 4B in 4-bit is ~2 GB of quantised blocks plus
~1.3 GB of tied embeddings. The problems are compute and
the 248k-token vocabulary: logits at 2048 tokens cost ~1 GB in fp16 and ~2 GB in
the fp32 softmax workspace, so a 4B run is comfortable at batch 1x1024 and slow.
Measured scaling from 0.8B is roughly 5x per token, so a 3-hour session buys
~1M tokens instead of ~6M. You would be training behaviour, not a broad
instruction mix. `qwenkit.load` and `sft.py` take `--model Qwen/Qwen3.5-4B`; the
code path is the same. The result would need its own baseline and its own gate
run — loss numbers are model-specific.

**Pretraining a 5B model from scratch: no, not on Kaggle, not on this budget.**
Chinchilla-optimal is ~100B tokens for 5B parameters. At ~600 tokens/s measured
on a T4 that is years of wall clock, before counting the data pipeline, the
evaluation, and the failures. Nobody reaches GLM/ChatGPT class from scratch on a
single consumer GPU; they reach it by fine-tuning a base someone else spent
millions on.

So the honest next step is not "5B". It is:

1. **Measure the ceiling first.** The repo already has the sweep (README Cell
   6b) and it already found one failure mode: Qwen3.5's reasoner spent the
   whole token budget thinking and wrote no document. `lorewrite.py` now forces
   `enable_thinking=False`, so re-run the sweep — `Qwen/Qwen3.5-4B` or `9B`,
   4-bit, retrieval over the local corpus — and score the documents with
   `loreeval.py`. No training. Then read them: the gate checks citations, names
   and numbers, not whether a source was read correctly. If the base model with
   retrieval writes a supported document, fine-tuning is for *style and format*,
   not capability. If it cannot, a 0.8B fine-tune was never going to.
2. **Then train the gap you measured**, on the mixture above, and run the gate.
3. **Only then** consider a bigger base. Changing the base invalidates every
   loss number in the repo, so it is a deliberate re-baseline, not a flag flip.

---

## Publishing what comes out

- **Adapter trained on Hub data only** — `upload.py` allows a public repo, after
  printing every source licence. Check them.
- **Adapter trained on your code/lore** — `upload.py` refuses a public repo
  unless you supply `--rights-cleared`. It will still publish privately.
- **Dataset you assembled from Hub data** — allowed if every source licence
  permits redistribution. Dolly's share-alike is the one people miss.
- **Dataset containing lore or your projects** — private only.

The command writes the card for you:

```powershell
.venv\Scripts\python.exe upload.py --adapter checkpoints/m5/adapter-best --dry-run
.venv\Scripts\python.exe upload.py --adapter checkpoints/m5/adapter-best `
    --repo your-name/hanny-0.8b-chat --private
```
