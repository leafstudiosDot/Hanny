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

### Next: M3 — use the fixed measurement to change one thing at a time

In priority order, now that changes can be *seen*:

1. Fix the corpus imbalance: `!`-exclude the vendored library and SDK folders
   listed in `data_report.txt`, raise `--max-chars`, re-measure. Expect val
   loss to go *up* and the samples to get *more like your own code* — that
   trade is the point.
2. Then, and only then, scale the model. The 1650's 4GB is nearly idle on a
   900k-parameter model; `N_LAYER` 4→8 and `BLOCK` 128→256 is roughly a 5×
   budget increase and fits.

---

## Log format going forward

Append a section per milestone with: what changed, the config that produced
it, the measured val loss with its `±`, and the `curve.csv` path. One
hyperparameter at a time — that rule is now enforceable, because a change
smaller than `0.01` is provably invisible.

**Note:** git is read-only for agents in this repo. Nothing here is committed.