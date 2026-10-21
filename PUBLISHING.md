# Publishing Hanny

This file is a checklist, not an automated release. It covers the two supported
targets:

- **Hugging Face Hub** — the adapter and, only when it is public-safe, a dataset.
- **GitHub** — the source tree, never the private corpus, checkpoints, or gate
  probe rows.

Git writes are deliberately **not** performed by the coding agent. Run the git
commands yourself after reading the checklist.

---

## Before anything leaves the machine

| Item | Rule |
| --- | --- |
| `cache/corpus-*.json.gz` | Contains private source text and folder/secret-file paths. Never publish. |
| `hanny-data.zip` | Private corpus archive. Never publish. |
| `checkpoints/m4/` | Trained on private code. Never publish publicly. |
| `evalsets/safety_probe.jsonl` | Gitignored real red-team prompts. Never publish. |
| `checkpoints/<run>/gate/result.json` | Contains `per_probe` rows with the private safety prompts. Keep the artifact zip private; quote aggregates only. |
| `data_dirs*.txt` | Gitignored local paths. Do not add them to git. |
| Adapter weights | Only publish an adapter whose `result.json` records public sources and whose card is honest about the gate. |

`pack.py --check` and `upload.py` scan for known private path fragments, but
that scan is **best-effort pattern matching**, not a guarantee. Before a public
release, also run `gitleaks` or `trufflehog` over the working tree and rotate any
key that was ever committed.

The current gated M5 adapter was trained only on:

- `allenai/tulu-3-sft-mixture` — ODC-BY
- `HuggingFaceH4/ultrachat_200k` — MIT

No private code was in that SFT mix. Keep the attribution and licence names in
the model card. The raw datasets are already hosted on the Hub; do not mirror
them unless you have a reason and can preserve their attribution.

---

## Hugging Face: model adapter

### 1. Get the gated artifact

The gated `m5-4b-v2` adapter is the one with weights
`sha256[:16] 7720503d592cdfae`, produced by the Kaggle run whose gate exited 0.
Download `m5-4b-v2-artifacts.zip` from the Kaggle Output pane (after **Save
Version**). A local `m5-4b-artifacts.zip` from an earlier run may contain the
superseded `0d98c1455a2e592c` adapter; do not publish that one by accident.

```powershell
# The local run folder is the old 0d98c145 adapter. Move it aside first;
# pack.py --restore refuses to overwrite an existing run by design.
Move-Item checkpoints\m5-4b-v2 checkpoints\m5-4b-v2-old-0d98c145
.venv\Scripts\python.exe pack.py --restore m5-4b-v2-artifacts.zip
```

If the archive carries a run folder that already exists, `pack.py --restore`
refuses to overwrite it. `--force` exists, but moving the old run aside is safer:
it keeps the superseded `0d98c145` weights available for comparison and makes
the replacement deliberate. Do not restore the older `m5-4b-artifacts.zip` and
publish it as gated. If `checkpoints\m5-4b-v2` is already absent, skip the
`Move-Item` line.

Then verify the weights and gate:

```powershell
.venv\Scripts\python.exe -c "import qwenkit; print(qwenkit.adapter_identity(r'checkpoints/m5-4b-v2/adapter-best'))"
.venv\Scripts\python.exe -c "import json; r=json.load(open(r'checkpoints/m5-4b-v2/result.json', encoding='utf-8-sig')); print(r.get('gate'), r.get('gate_exit'), r.get('world_size'))"
```

The hash must be `7720503d592cdfae` and `gate_exit` must be `0`. If you are
publishing a different DDP run later, record its own hash and gate.

### 2. Dry-run and read the card

```powershell
.venv\Scripts\python.exe upload.py `
    --adapter checkpoints/m5-4b-v2/adapter-best `
    --repo YOUR-NAME/hanny-4b-chat `
    --title "Hanny 4B chat adapter (M5 SFT)" `
    --license other `
    --rights-cleared "Trained only on allenai/tulu-3-sft-mixture (ODC-BY) and HuggingFaceH4/ultrachat_200k (MIT); no private code or user data was used. The publisher does not relicense the underlying datasets; attribution is in this card." `
    --dry-run
```

`upload.py` writes `checkpoints/m5-4b-v2/MODEL_CARD.md` and prints the upload
plan without touching the network. Read the card. It must say, in substance:

- the held-out loss is a **format-learning** measurement, not chat quality;
- the chat gate is a **floor**, not a certification (it scores a fixed probe
  set far smaller than a safety audit);
- the paired chat change is **within 2x its standard error**, so it is
  inconclusive, not an improvement;
- refusal rates were unchanged on these probes, which is not a safety audit;
- the adapter is not a general assistant and is limited by the 4B base.

The script now adds the DDP/world-size line when the run used it and the
within-noise caveat automatically. If the generated card does not match the
numbers above, stop and fix the card before uploading.

The `--license other` choice is deliberate: do not claim `apache-2.0` (or any
other licence) for weights unless you have checked the base model's terms and
can actually grant it. `upload.py` records the licence tag and your rights
statement verbatim.

### 3. Authenticate and publish

```powershell
hf auth login
```

Then choose one:

```powershell
# Public: allowed for a public-mix adapter, after the card/rights review.
.venv\Scripts\python.exe upload.py `
    --adapter checkpoints/m5-4b-v2/adapter-best `
    --repo YOUR-NAME/hanny-4b-chat `
    --title "Hanny 4B chat adapter (M5 SFT)" `
    --license other `
    --rights-cleared "Trained only on allenai/tulu-3-sft-mixture (ODC-BY) and HuggingFaceH4/ultrachat_200k (MIT); no private code or user data was used. The publisher does not relicense the underlying datasets; attribution is in this card."

# Private: allowed for any adapter; still subject to the leak scan.
.venv\Scripts\python.exe upload.py `
    --adapter checkpoints/m5-4b-v2/adapter-best `
    --repo YOUR-NAME/hanny-4b-chat-private `
    --private
```

`upload.py` refuses a public repo classified `code-private`, `mixed-private`, or
`unknown` unless you supply a rights statement. That refusal is intentional;
do not bypass it with a vague statement for the M4 adapter.

After upload, open the Hub page and check that `README.md`, `adapter_config.json`
and `adapter_model.safetensors` are present and that the card renders. The
adapter is only a delta; users must load `Qwen/Qwen3.5-4B` first, as the card
shows.

---

## Hugging Face: datasets

### Public sources (recommended)

The two training sources are already public and licensed. The model card lists
them in its `datasets:` metadata; no new dataset repository is needed for this
run. If you want a discoverable entry point, publish a small dataset card that
points at the sources and repeats their licences rather than re-uploading the
rows.

### A derived JSONL (only if you have the rights)

`upload.py` can publish a JSONL in the schema `sft.py` reads:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

```powershell
.venv\Scripts\python.exe upload.py `
    --dataset path\to\derived-chat.jsonl `
    --dataset-repo YOUR-NAME/hanny-chat-sft-derived `
    --dataset-title "Hanny chat SFT derived mix" `
    --license other `
    --rights-cleared "Derived only from allenai/tulu-3-sft-mixture (ODC-BY) and HuggingFaceH4/ultrachat_200k (MIT); attribution retained; no private code or user data." `
    --private   # use without --private only after the public review
```

A public dataset upload requires `--rights-cleared`; the card records it
verbatim. The current `sft.py` streams and encodes the sources directly, so it
does **not** emit a flat JSONL. Do not invent one by dumping private local data.
If you need a reproducible derived dataset, add an explicit export step and
review it before uploading.

---

## GitHub: source push checklist

This repository's current remote is the LAN Gitea remote `local`
(`http://192.168.50.2:9590/Incotelligence/Hanny.git`). To publish the source on
GitHub, add a second remote rather than replacing the existing one:

```powershell
git remote -v
git remote add github https://github.com/YOUR-NAME/Hanny.git
```

Before staging:

```powershell
.venv\Scripts\python.exe pack.py --check
git status --short
git diff --stat
```

Confirm that none of these are staged: `cache/`, `checkpoints/`,
`data_dirs.txt`, `hanny-data.zip`, `*.pt`, `evalsets/safety_probe.jsonl`, or any
gate `result.json` with `per_probe` rows. `.gitignore` covers most of them, but
`git add -A` can still surprise you; stage explicit paths:

```powershell
git add README.md AGENTS.md DATASETS.md PUBLISHING.md LOG.md
git add sft.py qwenkit.py upload.py pack.py hanny-notebook.ipynb
# add only the other source/doc files you actually changed
git status --short
git diff --cached --stat
```

Then commit and push. Replace the message and branch with your own:

```powershell
git commit -m "Add Kaggle DDP path and publishing checklist"
git push github main
```

Notes:

- **Never commit the private corpus or the M4 adapter.** The M5 public-data
  adapter is better published to the Hub than committed to git; GitHub LFS
  storage is not the right place for checkpoints.
- `.gitattributes` declares LFS for `*.pt` and archives. LFS is easy to get
  wrong and quota-limited; avoid adding binaries at all.
- If a public release includes the private safety probe file, stop. That file is
  gitignored for a reason; the committed fallback is
  `evalsets/safety_probe_public.jsonl`.
- A GitHub push is a publication. Run the secret scan first, and remember that
  history is harder to remove than a working-tree file.

---

## Kaggle input archives

`pack.py` writes two archives: `hanny-code.zip` (shareable scripts and public
probe sets) and `hanny-data.zip` (private corpus, real safety probes, lore).
Upload **both** to one **private** Kaggle Dataset and add a new version:

1. Kaggle → **Datasets** → your dataset → **New version**.
2. Upload the freshly built `hanny-code.zip` and `hanny-data.zip`.
3. Keep visibility **Private**. `hanny-data.zip` contains the corpus report and
   the real safety probes; `hanny-code.zip` alone is shareable, but the pair is
   not.
4. Re-import or restart `hanny-notebook.ipynb` so `/kaggle/input` sees the new
   version. Cell 2 intentionally rejects a stale archive.

A CLI upload is possible (`kaggle datasets version -p <folder> -m "..."`) but
requires the Kaggle CLI and credentials; this repository does not store them.
The notebook import path is the manual equivalent and is the one the checklist
uses.

---

## Kaggle-only training

M5 training is Kaggle-only by policy. `sft.py` detects a non-Kaggle environment
and exits before loading a model unless `--allow-local-training` is passed for a
deliberate smoke test. `--dry-run` and `--gate-only` are evaluation/data paths
and are not blocked.

The local GTX 1650 4 GB card is for inference and the gate only. It cannot fit the 4B
model plus optimizer state, and the 0.8B model already exceeds the safe training
budget at 1 x 1024. Use Kaggle T4 x2:

- both T4s for one DDP run: `4b-ddp` in the notebook;
- one T4 for the single-process recipe: `4b` or `0.8b`;
- `--gate-only` for an existing adapter, on one GPU.

The DDP path is new code. Its first 2-GPU run is the validation of the
collectives; a single-process smoke test does not prove multi-rank correctness.
Read the first Kaggle DDP log before quoting a speedup or a loss.
