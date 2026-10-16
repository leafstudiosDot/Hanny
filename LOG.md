# Hanny — build log

Reconstructed 2026-10-16 from the git history and the artifacts in
`checkpoints/`, because no LOG.md existed until now. Everything below marked
*reconstructed* is read off disk, not remembered — the numbers are real, the
reasoning is my inference from the code.

---

## M1 — from scratch to a working char-level GPT on your own code

**Status: done.** Commit `40b6157`.

The model is written in plain `torch`, no `transformers`, no downloaded
weights. It predicts the next *character*. Pipeline: `data.py` walks your
folders → builds a corpus → `model.py` trains a 4-layer GPT → prints a sample.

Config actually used (`checkpoints/first-own-code/best.pt`, `config` key):

| | |
| --- | --- |
| BLOCK | 128 characters of context |
| N_EMBD / N_HEAD / N_LAYER | 128 / 4 / 4 |
| DROPOUT, BATCH, LR | 0.1, 64, 1e-3 |
| steps | 4000 |
| parameters | 900,964 |
| vocabulary | 100 characters |
| corpus | 2,040 files, 27.4M train chars, 2.6M val chars (215 whole files held out) |

It works: `checkpoints/first-own-code/sample.txt` is coherent C — real
`#define`s, plausible struct fields, valid-looking media-library constant
names. That is the dominant vendored library's vocabulary, and for M1 that is
enough proof the architecture learns.

### Two things M1 left broken (both found while reconstructing this)

1. **The corpus is dominated by code you didn't write.** A single vendored
   media library accounts for **63.5%** of the corpus, and roughly 19% more is
   taken by third-party SDKs and extension libraries. Those are exactly what
   the `!` prefix in `data_dirs.txt` exists for. Right now Hanny mostly
   learns to imitate vendored C, while your own first-party source is under
   5% of the bytes. (Specific folder names are deliberately left out of this
   log — see `data_report.txt` in the run folder, which lists them locally.)
   Also: `--max-chars 30_000_000` was hit, so 5,438 files were never even
   read — the cap truncates in walk order.
2. **Validation was not comparable across runs.** `estimate_loss()` drew 30
   brand-new random windows on *every* call, so the same model scored
   differently each time and a hyperparameter change smaller than that noise
   was invisible. Every future milestone depends on fixing this first.

---

## M2 — trustworthy measurement

**Status: done.** This entry.

Goal: before scaling anything, be able to tell whether a change actually
helped. Two edits to `model.py`, no new files.

### What changed

**Fixed evaluation set.** Window start indices are now chosen once, spread
evenly across the split (`np.linspace` over the whole thing), and every run is
scored on exactly those 256 windows. Same model, same number, every time.

**A standard error.** Each eval reports `loss ± se`, where `se` is the
standard error of the mean over the 256 windows. A change worth less than
about 2× that `±` is noise, and you can now see that directly in the output
instead of guessing.

**`curve.csv` per run.** One row per eval, appended to
`checkpoints/<run>/curve.csv`, so you have an actual learning curve on disk
rather than scrolling a terminal.

**Bug fix (unrelated, but it was going to bite).** `print(sample)` crashed
with `UnicodeEncodeError` on Windows. The console defaults to a legacy
codepage that cannot encode `U+FFFD` — the unknown-character token `UNK`. Any
sample containing one died *after* the run finished and saved, so it looked
like a crash but had already written `sample.txt`. Forced UTF-8 stdout at the
top of `model.py`. Your earlier runs escaped this by luck; the samples simply
happened to contain no unknown chars.

### How it was verified

Two identical runs (`--steps 40 --eval-every 20`) produced **byte-identical**
`curve.csv`:

```
step,train,train_se,val,val_se,seconds
0,4.7715,0.0002,4.7732,0.0008,0.4
20,3.1238,0.0049,3.2002,0.0038,2.2
40,2.8280,0.0040,2.8915,0.0037,3.8
```

Before this change those rows would have differed in the third decimal every
run. `se ≈ 0.004`, so from here on **ignore any val-loss change smaller than
about 0.01**.

### Honest caveats

- The `se` assumes the 256 windows are independent. They come from
  concatenated whole files, so neighbouring windows do overlap in content and
  the true uncertainty is somewhat larger than printed. Treat `±` as a floor,
  not a guarantee.
- Old checkpoints (`hanny_run001/002.pt`, `first-own-code_run001.pt`) were
  scored with the old random-window method. **Their val loss is not comparable
  to any new number.** The architecture is unchanged, so they still load.
- Nothing about the corpus or the architecture changed in M2, so this is a
  clean baseline: re-run M1's exact config once and that curve is the number
  every later milestone is measured against.

### Next: M3 — the token-level line

M3 turned out to be a much bigger decision than "fix the corpus imbalance."
Re-read the brief: the target is a GLM-5-class coding agent. That is
unreachable (GLM-5 is 744B parameters / 40B active on 28.5T tokens; Hanny is
0.9M on ~10M). So the route is **fine-tune an existing small open model**, and
M3 establishes the thing fine-tuning is measured against.

---

## M3 — real tokenizer, real baseline

**Status: done.**

### What was decided, and why

| Decision | Answer | Reason |
| --- | --- | --- |
| Base model | `Qwen/Qwen3.5-0.8B` | Apache-2.0 (publishable), ungated, 752M params |
| Why not `Qwen3-Coder-1.5B` | **it does not exist** | Qwen retired the small Coder line — checked before recommending |
| Char-level vs token-level | token-level | ~2.9 chars/token, so a context window covers ~3x more code |
| Hardware | Kaggle T4 ×2 primary, 1650 for dev | see "what the hardware forced" below |

The decisive discovery: **its chat template already implements tool/function
calling.** So M5 does not have to teach a small model the tool-call protocol
from zero — it only has to teach it *our* tools. That one fact is worth more to
the agent goal than any amount of extra pretraining.

### Architecture, measured rather than guessed

```
Qwen3_5ForCausalLM — 752M params total, 1.50 GB in fp16
  text:  24 layers | hidden 1024 | 8 query heads, 2 KV heads (GQA 4:1)
         head_dim 256 | intermediate 3584 | context 262144 | tied embeddings
  vocab: 248,320          <- 40% of the model is this embedding table
  also carries a vision tower (hidden 768) that a text coding agent never uses
```

The 248k vocabulary is the number that shaped everything after it.

### What the hardware forced

Cross-entropy over 248k classes makes the **logits** the memory bottleneck, not
the weights. Measured peak on the 4GB card:

| batch × seq | peak VRAM | result |
| --- | --- | --- |
| 1 × 512 | 2.84 GB | safe |
| 1 × 1024 | 4.12 GB | over budget |
| 2 × 1024 | — | OOM |
| 4 × 1024 | — | OOM |

**Conclusion: the 1650 can evaluate this model at batch 1 / seq 512, and cannot
train it.** That settles the Kaggle question — not as a preference, but because
the card physically cannot hold the optimizer states.

Two more findings, both pointing the same way:

- Qwen3.5's layers need `flash-linear-attention` for its gated-delta-rule
  path. Without it transformers prints a warning and falls back to a correct
  but **much slower** reference implementation.
- `bitsandbytes` (needed for 4-bit QLoRA in M4) is a CUDA extension that
  generally fails to build on Windows.

Both are Linux-friendly. Training on Kaggle is not a workaround here, it is
the only sane path.

### The baseline

`baseline.py` scores the **untouched** model on the same 215 held-out files,
same fixed-window method as M2, and writes `baseline.json`:

```
loss         1.3240 +- 0.0608
perplexity   3.758
2.89 chars per token | 899,874 val tokens | 64 windows x 512 tokens | 146s
```

**This is the number M4 has to beat.** It matters that the reference point is
not `ln(vocab)` (= 12.42). Fine-tuning does not teach a model to code — this
one already can. It teaches it *this codebase's* idioms. Perplexity 3.76 means
the model weighs ~4 candidate tokens on average before getting it right. There
is real room to move, and there is no way to prove you moved it without a
number taken *before* you started.

### Problems found and fixed

1. **`transformers` 5.x renamed `torch_dtype=` to `dtype=`.** Caught by
   reading the installed source rather than writing from memory; the old kwarg
   is gone.
2. **The M2 console fix was copy-pasted and immediately needed again** in
   `baseline.py`. Centralised into `utf8console.py`, imported by every entry
   point. The failure mode is nasty because the error surfaces at *flush* time:
   the script runs to completion, saves everything, then exits 1 on the final
   print. It looks like a crash in code that succeeded.
3. **`std(ddof=1)` on a single window returns NaN**, which would have written a
   null noise floor into `baseline.json` without complaining. Now rejected at
   argument-check time.
4. A PowerShell pipeline artifact briefly looked like a Python exit-code
   failure. It was not — the script exits 0. Worth recording so it is not
   re-investigated later.

### Deliberately not done yet

`peft`, `trl` and `bitsandbytes` are **not installed**. They belong to M4 and
pull CUDA extensions that fail on Windows; installing them here would produce
a broken half-working environment and a confusing first error on Kaggle.

### The corpus experiment

M1 concluded the corpus was "96% code you didn't write," and left it as advice.
This is the measurement. One line was added to `data_dirs.txt` — a `!` prefix
excluding the vendored-requirements folder and everything beneath it — and
nothing else changed.

| | before | after |
| --- | --- | --- |
| train files / chars | 2,040 / 27,412,052 | 549 / 2,877,167 |
| val files / chars | 215 / 2,603,687 | 44 / 215,664 |
| **files dropped by the char cap** | **5,438** | **0** |
| top file type | `.c` 1234 | `.tsx` 150 |
| files skipped as possible secrets | 7 | **15** |
| biggest source | one vendored library, 63.5% | own application code, 25.5% |

**The cap stopped binding.** Re-running with `--max-chars 200000000` produced
byte-identical output: the entire remaining corpus is 3.1M characters, an
order of magnitude under the old limit. The 5,438 dropped files were never a
cap problem in the first place — they were a *consequence of the vendored
folders being walked first*. Excluding those folders removed the pressure.

**The secret scanner found twice as many files** (7 → 15) because first-party
files are now actually being read. More of the corpus is being inspected, not
less.

The sample confirms the change end to end. Before:

```
def ff_video_write_height && ff_vp9_video_field_height) {
```

After:

```
def from 'times';
import { username } from './ww.map.h'
export function hashWindowScriptort() { Color_->setWindow(tr("object") {
```

### Every number moved — in the "wrong" direction, on purpose

| measurement | before | after |
| --- | --- | --- |
| Qwen3.5-0.8B loss | 1.3240 ± 0.0608 | **1.5098 ± 0.0889** |
| Qwen3.5-0.8B perplexity | 3.758 | **4.526** |
| Hanny best val | 1.107 | 1.342 |
| Hanny vocabulary | 100 | 99 |

**Worse is the correct outcome.** The old 1.3240 was partly credit for
scoring well on ffmpeg — famously idiomatic C that saturates the public
internet and was 63.5% of the corpus. 1.5098 is the honest number for
first-party code, and it is the number fine-tuning actually has to move.
Reporting 1.3240 would have meant celebrating a measurement advantage that
came from the wrong data.

`baseline.json` is tracked in git precisely so that this change is a reviewable
diff rather than a silent overwrite.

### Known weakness in the current eval

The val split is now 59,549 tokens. Sixty-four windows of 512 tokens, spaced
evenly, gives ~930 tokens of spacing per window — so **adjacent windows
overlap by roughly 45%**. Overlapping windows are correlated, which makes the
reported `±0.089` optimistic: the true noise floor is wider than printed.

Options for M4, in order of preference: shorten windows to 256 (halves the
overlap), drop to 32 windows, or hold out more files. Until then, treat
`±0.089` as a floor and require ~`0.18` before believing an improvement.

Also outstanding: a Next.js build-output directory (`.open-next/.build`) still
contributes ~1.2%. `data.py` skips some generated files but not that one.

---

## M4 — LoRA fine-tune against this baseline

**Status: not started.**

Run on Kaggle T4. Success is not "it trained" — it is **1.5098 going down**,
measured with the identical window method, by more than about `0.18`. The
noise floor on this eval is `±0.089`, so a drop smaller than that is not
evidence of anything.

Constraint to plan around: the first-party training set is only **2.88M
characters (~1M tokens)**. That is enough to teach style and idiom, and not
enough to teach capability. If M4 stalls, the answer is more data — most
likely a public code corpus from HuggingFace for breadth, with the first-party
code used for the fine-tune itself.

---

## Log format going forward

Append a section per milestone with: what changed, the config that produced
it, the measured val loss with its `±`, and the `curve.csv` path. One
hyperparameter at a time — that rule is now enforceable, because a change
smaller than `0.01` is provably invisible.

**Note:** git is read-only for agents in this repo. Nothing here is committed.