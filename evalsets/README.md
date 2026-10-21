# evalsets

The fixed probe sets `chateval.py` scores. They are versioned in git because a
measurement nobody can reproduce is not a measurement — the whole point of the
M2 discipline is that every number traces back to an identical input.

## Two safety files, on purpose

| File | In git? | What it is |
| --- | --- | --- |
| `chat_probe.jsonl` | yes | (prompt, reference response) pairs for chat loss |
| `safety_probe_public.jsonl` | yes | redacted probes, categories in `[brackets]` |
| `safety_probe.jsonl` | **no** | the real phrasings, gitignored |

The public safety file keeps the harm *categories* and replaces the specific
wording. That is deliberate. A short functional prompt is harmless on its own, but a repository full of them reads
as what HF's Content Policy describes under Restricted Content → Harmful or
Abusive Content ("content that attempts to
generate malicious code"), and account suspension is on that list. Whether the
intent is safety research is not something an automated flag, a future
employer's `git log`, or a casual reader of the diff gets to assess.

Keeping the categories but not the phrasings means a clone can still run the
eval and still get a number. What it cannot do is reproduce *your* number
exactly, because a bracketed placeholder is not a refusal probe — the model
answers the literal string rather than refusing it.

## What that means for your numbers

| Where you run it | Which file | Comparable across runs? |
| --- | --- | --- |
| your machine, Kaggle | `safety_probe.jsonl` | yes — this is the real measurement |
| a fresh clone | `safety_probe_public.jsonl` | **no**, and the run says so |

`chateval.py` prints which file it loaded and warns loudly when it falls back to
the public set. Do not put a public-set number in the same table as a real-set
number.

## Shipping the real file to Kaggle

The real probe is gitignored but **is** included in `hanny-data.zip`. `pack.py`
builds from an explicit file list rather than a directory sweep, so the
gitignore rule does not apply — and the archive is itself private and gitignored
in turn. Run:

```powershell
.venv\Scripts\python.exe pack.py
```

That keeps the archive private (it already is — it holds your corpus and is
gitignored in turn) while the git repo stays clean.

## `chat_probe.jsonl` — 50 pairs, 10 of them multi-turn

Each line is one conversation and one reference answer:

```json
{"category": "context",
 "history": [{"role": "user", "content": "..."},
             {"role": "assistant", "content": "..."}],
 "prompt": "...", "response": "..."}
```

`history` is optional; when present, `evalkit.build_chat_batch` renders those
turns as real chat turns before `prompt`. Folding them into the prompt string
would score a quotation, not a conversation. `category` is documentation for
readers (`instruction`, `concept`, `context`, `uncertainty`); nothing scores it.

Coverage, deliberately spread rather than concentrated: 8 instruction-following
pairs (format constraints, exact counts, single JSON object), 9 concept
explanations, 10 multi-turn pairs where the answer depends on the history, and 8
uncertainty pairs (unknowable personal facts, false premises, a paradox).

Validate the file without loading any weights:

```powershell
.venv\Scripts\python.exe chateval.py --check-probes
```

It renders every pair through the real chat template, asserts the prompt is a
token prefix of the full turn (so the response can be isolated), and prints the
token count per pair. That is the check to run after editing this file, because
a malformed history turn or an over-long response otherwise fails deep inside a
4-bit run.

**Overlap with training data was measured, not assumed.** Every prompt and every
history turn (68 strings) was searched verbatim, after case and whitespace
normalisation, in the shards `sft.py` streams: `allenai/tulu-3-sft-mixture` @
`b14afda60f1b` (six train shards, 939,343 rows) and
`HuggingFaceH4/ultrachat_200k` @ `8049631c405a` (`train_sft`, 207,865 rows) —
1,147,208 rows, 4.04G characters. Two prompts were present in tulu as generic
textbook questions ("What is compound interest?" inside a numbered list, and
"weather and climate"); both were replaced with versions that require reasoning,
and the replacements were re-scanned. Zero matches remain.

The check is **verbatim only**, and it scans prompts, not reference answers. A
paraphrase of a training item, or a copied *answer*, would not be caught.

### Composition matters as much as count

The first 15 pairs leaned technical (list vs tuple, null pointers, CI failures,
overfitting), which is the corner where a code adapter looks least bad. Measured
on one 0.8B fp16 harness, base vs the M4 code adapter:

| pairs | change | paired se | verdict |
| --- | --- | --- | --- |
| first 15 (the old file) | -0.0737 | 0.0239 | REAL regression |
| first 40 | -0.0291 | 0.0195 | within noise |
| all 50 | -0.0180 | 0.0164 | within noise |

The absolute numbers are a smoke test (fp16, 0.8B), but the comparison is
internal to one run: adding non-code conversation cut the estimated effect by
four. An old 15-pair result is therefore not a smaller-precision version of a
50-pair result — it is a differently *composed* measurement, and the two must
not be quoted side by side as if only n changed.

Numbers from the earlier 15-pair file are not comparable with a 50-pair run.
`result.json` records `chat_probe_sha256_16`, `chat_pairs` and
`chat_pairs_multiturn` so a result cannot be attributed to the wrong probe set.
