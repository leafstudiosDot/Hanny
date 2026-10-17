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
