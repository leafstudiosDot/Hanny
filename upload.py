"""Publish an adapter or an SFT dataset to Hugging Face, with the privacy gate on.

    # see exactly what would be uploaded and read the generated card first
    .venv\\Scripts\\python.exe upload.py --adapter checkpoints/m5/adapter-best --dry-run

    # a private repo: allowed for any adapter, still recorded
    .venv\\Scripts\\python.exe upload.py --adapter checkpoints/m5/adapter-best \\
        --repo your-name/hanny-0.8b-chat --private

    # a public repo needs a rights statement, and says so in the card
    .venv\\Scripts\\python.exe upload.py --adapter checkpoints/m5/adapter-best \\
        --repo your-name/hanny-0.8b-chat --license apache-2.0 \\
        --rights-cleared "trained only on ODC-BY/MIT/Apache-2.0 datasets listed below"

    .venv\\Scripts\\python.exe upload.py --dataset checkpoints/m5/sft.jsonl \\
        --dataset-repo your-name/hanny-chat-sft --private

Why this script refuses things
------------------------------
The M4 adapter was trained on the user's private source code. `README.md` and
`AGENTS.md` both say the same thing: rights in a checkpoint only extend as far
as rights in its training data, so that adapter cannot be published openly by
this repo. The refusal is not a formality -- a public model card is a licence
grant, and it is not ours to make.

`hf auth login` (huggingface_hub 2.x renamed the CLI from `huggingface-cli`) is
required only for a real upload; --dry-run is fully offline.
"""
import argparse
import json
import os
import sys

import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pack  # noqa: E402  (import-safe: guarded by __main__)
import qwenkit  # noqa: E402

CARD_HEADER = """---
base_model: {base_model}
library_name: peft
license: {license_}
tags:
- lora
- qwen3.5
- hanny
- chat
{datasets}language: en
---

# {title}

{summary}
"""


def rel(path):
    """Relative path for display; absolute when the target is on another drive."""
    try:
        return os.path.relpath(path, HERE)
    except ValueError:
        return os.path.abspath(path)


def load_result(adapter_dir):
    path = os.path.join(os.path.dirname(os.path.abspath(adapter_dir).rstrip("/\\")),
                        "result.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def classify(result):
    """Return (kind, human explanation). Never guesses in the permissive direction."""
    if result is None:
        return "unknown", ("No result.json was found beside this adapter, so its "
                           "training data cannot be checked at all.")
    if result.get("kind") == "chat-sft":
        sources = result.get("sources") or []
        if not sources:
            return "unknown", ("The SFT result records no training sources, so "
                               "its provenance cannot be checked.")
        local = [s for s in sources if s.get("kind") == "local"]
        if local:
            names = ", ".join(s.get("name", "?") for s in local)
            return "mixed-private", (f"Trained partly on local file(s): {names}. "
                                     f"Those may be your own projects or lore.")
        hub = [s for s in sources if s.get("kind") == "hub"]
        if len(hub) != len(sources):
            return "unknown", ("The SFT result contains source entries that are "
                               "neither Hub nor local; provenance is unclear.")
        return "public-mix", ("Trained only on Hub datasets, listed in the card. "
                              "Check each dataset's licence before publishing.")
    if result.get("corpus_fingerprint"):
        return "code-private", (
            f"Trained on the private code corpus "
            f"(fingerprint {result.get('corpus_fingerprint')}), "
            f"{result.get('steps_done', '?')} steps over raw source text "
            f"(no chat template recorded).")
    return "unknown", "result.json does not describe its training data."


def scan(text, tokens, allow):
    """Same leak rules as pack.py: private path fragments and probe phrases."""
    problems, low = [], text.lower()
    for bad in pack.FORBIDDEN_SUBSTRINGS:
        if bad in low and bad not in allow:
            problems.append(f"probe phrase {bad!r}")
    for tok in tokens:
        if tok in low and tok not in allow:
            problems.append(f"private path fragment {tok!r}")
    return problems


def gate_summary(run_dir):
    """Read chateval's result.json if sft.py ran the gate. Returns a dict or None."""
    path = os.path.join(run_dir, "gate", "result.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    return {
        "pairs": r.get("chat_pairs"),
        "chat_loss_change_paired": r.get("chat_loss_change_paired"),
        "chat_loss_change_paired_se": r.get("chat_loss_change_paired_se"),
        "refusal_p_exact": r.get("refusal_p_exact"),
        "safety_probe_kind": r.get("safety_probe_kind"),
    }


def build_model_card(info, result, kind, why, args, gate):
    base_model = (info.get("base_model") or (result or {}).get("model")
                  or qwenkit.DEFAULT_MODEL)
    run = (result or {}).get("run") or os.path.basename(
        os.path.dirname(os.path.abspath(args.adapter).rstrip("/\\")))
    is_sft = (result or {}).get("kind") == "chat-sft"
    title = args.title or (f"Hanny chat adapter ({run})" if is_sft
                           else f"Hanny code adapter ({run})")

    if is_sft:
        summary = (f"A LoRA adapter for `{base_model}`, trained through that model's "
                   f"own chat template with loss on assistant tokens only. "
                   f"It is a {'private' if args.private else 'public'} experiment, "
                   f"not a general-purpose assistant.")
    else:
        summary = (f"A LoRA adapter for `{base_model}` trained on raw source text, "
                   f"not on chat turns. It is a code-domain adapter: it continues "
                   f"code, and it is not a chatbot. The evaluation below says "
                   f"exactly what was measured and what was not.")

    hub_sources = [s["id"] for s in (result or {}).get("sources", [])
                   if s.get("kind") == "hub"]
    datasets_yaml = ("datasets:\n" + "".join(f"- {x}\n" for x in hub_sources)
                     if hub_sources else "")

    lines = [CARD_HEADER.format(
        base_model=base_model, license_=args.license,
        datasets=datasets_yaml,
        title=title, summary=summary)]

    precision = ((result or {}).get("precision")
                 or (result or {}).get("loss_before_precision") or "?")
    tokens_seen = ((result or {}).get("tokens") or {}).get("train_seen")
    if not tokens_seen and all((result or {}).get(k) for k in
                               ("steps_done", "batch", "seq_len")):
        tokens_seen = result["steps_done"] * result["batch"] * result["seq_len"]

    lines.append("\n## What this is\n")
    lines.append(f"- Base model: `{base_model}`\n")
    lines.append(f"- Adapter weights `sha256[:16]`: `{info.get('sha256_16')}` "
                 f"({info.get('bytes', 0) / 1e6:.1f} MB)\n")
    if info.get("lora_r"):
        lines.append(f"- LoRA r={info.get('lora_r')}, alpha={info.get('lora_alpha')}, "
                     f"targets: {', '.join(info.get('target_modules') or [])}\n")
    if result:
        lines.append(f"- Run: `{run}` | precision: {precision} | "
                     f"chat template: `{(result or {}).get('chat_template_sha16') or 'not recorded'}`\n")
        lines.append(f"- Optimizer steps: {result.get('steps_done', '?')} "
                     f"(best at {result.get('best_step', '?')}) | "
                     f"tokens processed: "
                     f"{f'{tokens_seen:,}' if tokens_seen else 'not recorded'}\n")
        world_size = result.get("world_size", 1)
        if world_size and world_size > 1:
            lines.append(
                f"- Distributed: DDP across {world_size} GPUs "
                f"(`{result.get('launcher', 'torch.distributed.run')}`); "
                f"the effective batch is the per-GPU batch times {world_size}.\n")

    lines.append("\n## Training data\n")
    if result and result.get("sources"):
        lines.append("| source | split | rows | licence |\n| --- | --- | --- | --- |\n")
        for s in result["sources"]:
            if s.get("kind") == "hub":
                lines.append(f"| `{s.get('id')}` | {s.get('split')} | "
                             f"{s.get('limit') or s.get('rows_read') or '?'} | "
                             f"{s.get('license') or 'unknown'} |\n")
            else:
                lines.append(f"| local file `{s.get('name')}` | - | - | "
                             f"not disclosed |\n")
        counts = result.get("data_counts", {})
        if counts:
            lines.append(f"\nRows read: {counts.get('rows_read', '?')}; "
                         f"conversations kept: {counts.get('conversations', '?')}; "
                         f"training examples: {result.get('examples', {}).get('train', '?')}; "
                         f"duplicates removed: {counts.get('duplicates', '?')}.\n")
        lines.append("\nOne example is produced per assistant turn, so the context "
                     "is repeated across turns of the same conversation.\n")
    else:
        lines.append("Training data is not disclosed by the producing run. "
                     f"{why}\n")

    lines.append("\n## Evaluation\n")
    if result and "loss_before" in result:
        what = ("assistant-token loss on held-out conversations"
                if is_sft else
                "next-token loss on held-out source-text windows")
        paired = result.get("paired_se")
        if paired is not None:
            improvement = (result.get("loss_before", 0.0)
                           - result.get("loss_after", 0.0))
            change = (f"| improvement (before - after) | "
                      f"{improvement:.4f} "
                      f"(paired se {paired:.4f}) |\n")
            note = ("Both numbers come from the same run at the same precision, and "
                    "the difference is paired item-by-item, so it is the tighter "
                    "comparison.")
        else:
            change = (f"| improvement (before - after) | "
                      f"{result.get('loss_change', 0):.4f} "
                      f"(no paired se recorded) |\n")
            note = ("No paired standard error exists for this run, so the drop is "
                    "compared against the base model's own se above -- a looser "
                    "noise floor than the SFT path uses.")
        lines.append(
            f"Held-out {what}, measured before and after training in the same "
            f"process at the same precision:\n\n"
            f"| | loss |\n| --- | --- |\n"
            f"| base | {result['loss_before']:.4f} "
            f"+- {result.get('loss_before_se') or 0:.4f} |\n"
            f"| adapter (best) | {result['loss_after']:.4f} |\n"
            f"{change}\n{note} This is not comparable with any fp16 number in the "
            f"producing repository; 4-bit quantisation itself costs ~0.09 nats.\n")
        if not is_sft:
            lines.append("\nThis measures code modelling on the private corpus. It "
                         "says nothing about conversation quality -- that is what "
                         "`chateval.py` measures, and it was not run for this "
                         "adapter.\n")
    else:
        lines.append("No held-out loss is recorded for this adapter.\n")

    if gate and gate.get("chat_loss_change_paired") is not None:
        change = gate["chat_loss_change_paired"]
        se = gate.get("chat_loss_change_paired_se")
        if se is None:
            se = float("nan")
        gate_exit = result.get("gate_exit") if result else None
        if gate_exit not in (None, 0):
            status = (f"The gate process exited {gate_exit}; its numbers are "
                      f"incomplete.\n")
        else:
            status = "The gate process completed (exit 0).\n"
        inconclusive = ""
        if se is not None and abs(change) < 2 * se:
            inconclusive = (
                " That change is within 2x its paired standard error, so it is "
                "inconclusive on this probe set, not evidence of an improvement. ")
        lines.append(
            f"\nThe M5 gate (`chateval.py`) scored {gate.get('pairs')} chat pairs and "
            f"reported a paired chat-loss change of {change:+.4f} "
            f"+- {se:.4f} (positive = the adapter's loss is lower)."
            f"{inconclusive}{status}"
            f"That probe set is a smoke test, not a measurement, and the script says "
            f"so on every run. Refusal p-values: "
            f"{json.dumps(gate.get('refusal_p_exact') or {})}.\n")
    elif is_sft:
        lines.append("\nThe M5 gate was not run for this adapter, so general chat "
                     "ability and refusal behaviour are unmeasured.\n")

    lines.append("\n## Use\n```python\n"
                 "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
                 "from peft import PeftModel\n\n"
                 f"tok = AutoTokenizer.from_pretrained({base_model!r})\n"
                 f"base = AutoModelForCausalLM.from_pretrained({base_model!r}, dtype='float16', device_map='auto')\n"
                 "model = PeftModel.from_pretrained(base, 'YOUR-NAME/adapter-repo')\n"
                 "model.eval()\n\n"
                 "messages = [{'role': 'user', 'content': 'Hello'}]\n"
                 "ids = tok.apply_chat_template(messages, add_generation_prompt=True,\n"
                 "                              return_tensors='pt', enable_thinking=False)\n"
                 "out = model.generate(ids, max_new_tokens=256, do_sample=True,\n"
                 "                     temperature=0.7, top_p=0.9)\n"
                 "print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))\n"
                 "```\n")

    lines.append("\n## Limitations\n")
    lines.append(f"- `{base_model}` is a small model. Fine-tuning changes behaviour; "
                 f"it does not add knowledge the base model lacks.\n")
    lines.append("- This is not a ChatGPT- or Claude-class assistant, and no amount "
                 "of further LoRA training on this base will make it one.\n")
    lines.append("- Refusal behaviour is not guaranteed. The gate is a 15-pair / "
                 "26-probe smoke test and cannot certify safety.\n")
    if not is_sft:
        lines.append("- Trained on raw text rather than chat turns, so expect code "
                     "continuation and file-shaped output more often than answers.\n")
    if kind != "public-mix":
        lines.append("- Training data is private and not disclosed, so the adapter "
                     "cannot be audited for memorised content.\n")
    lines.append("- Generation settings used during development: temperature 0.7, "
                 "top_p 0.9, top_k 20, thinking off. Different settings are a "
                 "different model behaviour.\n")

    lines.append("\n## Provenance and rights\n")
    lines.append(f"- Provenance class: **{kind}**. {why}\n")
    if args.rights_cleared:
        lines.append(f"- The publisher states: *{args.rights_cleared}*\n")
    else:
        lines.append("- No rights statement was supplied. The licence above covers "
                     "this repository's packaging only; rights in the weights are "
                     "limited by rights in the training data.\n")
    if args.description:
        lines.append(f"\n{args.description}\n")
    return "".join(lines)


def dataset_card(args, report):
    lines = ["---\n",
             "task_categories:\n- text-generation\n",
             "language:\n- en\n",
             f"license: {args.license}\n",
             "---\n\n",
             f"# {args.dataset_title or 'Hanny chat SFT data'}\n\n",
             "Conversational supervised fine-tuning data in the schema `sft.py` "
             "reads. Each line is one conversation; `sft.py` expands it into one "
             "example per assistant turn at training time.\n\n",
             "```json\n{\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, "
             "{\"role\": \"assistant\", \"content\": \"...\"}]}\n```\n\n"]
    if report:
        lines.append(f"- Rows read by the last build: {report.get('counts', {}).get('rows_read', '?')}\n")
        lines.append(f"- Conversations kept: {report.get('counts', {}).get('conversations', '?')}\n")
        lines.append(f"- Training examples after per-turn expansion: "
                     f"{report.get('examples', {}).get('train', '?')}\n")
        lines.append(f"- Chat template: `{report.get('chat_template_sha16', '?')}`\n")
        if report.get("sources"):
            lines.append("\n## Sources\n")
            for s in report["sources"]:
                if s.get("kind") == "hub":
                    lines.append(f"- `{s.get('id')}` ({s.get('split')}, "
                                 f"licence {s.get('license') or 'unknown'})\n")
                else:
                    lines.append(f"- local file `{s.get('name')}` (not disclosed)\n")
    lines.append("\n## Privacy\n\n")
    if args.rights_cleared:
        lines.append(f"The publisher states: *{args.rights_cleared}*.\n")
    else:
        lines.append("No rights statement was supplied. Do not redistribute this "
                     "file unless you can grant rights in every source above.\n")
    return "".join(lines)


def dataset_report(path):
    """Look for sft.py's data_report.json beside the JSONL."""
    for candidate in (os.path.join(os.path.dirname(os.path.abspath(path)),
                                   "data_report.json"),):
        if os.path.exists(candidate):
            try:
                with open(candidate, encoding="utf-8-sig") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None
    return None


def dataset_leaks(path, tokens, allow):
    problems = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            found = scan(line, tokens, allow)
            if found:
                problems.append(f"{os.path.basename(path)}:{lineno}: {found[0]}")
                if len(problems) >= 5:
                    break
    return problems


def confirm_and_upload(args, files, dataset_card_path):
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as e:  # noqa: BLE001 - any auth failure is the same message
        raise SystemExit(f"not logged in ({e}). Run: hf auth login")

    user = who.get("name") or "unknown"
    if args.repo and "/" not in args.repo:
        args.repo = f"{user}/{args.repo}"
    if args.dataset_repo and "/" not in args.dataset_repo:
        args.dataset_repo = f"{user}/{args.dataset_repo}"
    print(f"logged in as {user}")

    if args.repo:
        print(f"creating model repo {args.repo} (private={args.private}) ...")
        api.create_repo(args.repo, repo_type="model", private=args.private,
                        exist_ok=True)
        for path, name in files:
            print(f"  uploading {name} ({os.path.getsize(path) / 1e6:.1f} MB) ...")
            api.upload_file(path_or_fileobj=path, path_in_repo=name,
                            repo_id=args.repo, repo_type="model")
        print(f"  https://huggingface.co/{args.repo}")

    if args.dataset:
        print(f"creating dataset repo {args.dataset_repo} (private={args.private}) ...")
        api.create_repo(args.dataset_repo, repo_type="dataset", private=args.private,
                        exist_ok=True)
        api.upload_file(path_or_fileobj=args.dataset,
                        path_in_repo=os.path.basename(args.dataset),
                        repo_id=args.dataset_repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=dataset_card_path, path_in_repo="README.md",
                        repo_id=args.dataset_repo, repo_type="dataset")
        print(f"  https://huggingface.co/datasets/{args.dataset_repo}")


def main():
    p = argparse.ArgumentParser(
        description="Publish an adapter and/or an SFT dataset, refusing private data.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", help="adapter-best directory to publish")
    p.add_argument("--repo", help="model repo id, e.g. your-name/hanny-0.8b-chat")
    p.add_argument("--dataset", help="JSONL file to publish as a dataset")
    p.add_argument("--dataset-repo", help="dataset repo id")
    p.add_argument("--private", action="store_true", help="create private repos")
    p.add_argument("--license", default="other",
                   help="model card licence tag (apache-2.0, mit, cc-by-4.0, other)")
    p.add_argument("--title", help="model card title")
    p.add_argument("--dataset-title", help="dataset card title")
    p.add_argument("--description", help="extra Markdown appended to the model card")
    p.add_argument("--rights-cleared", metavar="REASON",
                   help="state why you may publish this training data. Required "
                        "for a public repo built on local/private data; recorded "
                        "verbatim in the card.")
    p.add_argument("--allow", action="append", default=[], metavar="TOKEN",
                   help="accept one leak-scan hit as deliberate (repeatable)")
    p.add_argument("--dry-run", action="store_true",
                   help="write the card locally and print the upload plan; no network")
    args = p.parse_args()

    if not args.adapter and not args.dataset:
        raise SystemExit("nothing to do: pass --adapter and/or --dataset")
    if args.dataset and not args.dataset_repo:
        raise SystemExit("--dataset needs --dataset-repo")
    if args.adapter and not args.repo and not args.dry_run:
        raise SystemExit("--adapter needs --repo (or --dry-run to just write the card)")
    if not args.repo:
        args.repo = "YOUR-NAME/unnamed"
        args.adapter_repo = args.repo
    else:
        args.adapter_repo = args.repo

    tokens, cache = pack.private_path_tokens()
    allow = {a.lower() for a in args.allow}
    print(f"leak scan: {len(tokens)} private path fragment(s) "
          f"from {cache or 'no corpus cache'} + "
          f"{len(pack.FORBIDDEN_SUBSTRINGS)} fixed phrases")

    files, card_text, dataset_card_text, report = [], None, None, None
    dataset_card_path = None

    if args.adapter:
        if not os.path.isdir(args.adapter):
            raise SystemExit(f"--adapter {args.adapter!r} is not a directory")
        info = qwenkit.adapter_identity(args.adapter)
        if "error" in info:
            raise SystemExit(info["error"])
        run_dir = os.path.dirname(os.path.abspath(args.adapter).rstrip("/\\"))
        result = load_result(args.adapter)
        kind, why = classify(result)
        gate = gate_summary(run_dir)

        print(f"\nadapter: {args.adapter}")
        print(f"  weights sha256[:16] {info['sha256_16']} "
              f"({info['bytes'] / 1e6:.1f} MB)")
        print(f"  provenance class: {kind}")
        print(f"  {why}")

        public = not args.private
        if public and kind in ("code-private", "mixed-private", "unknown") \
                and not args.rights_cleared:
            print("\n" + "!" * 72)
            print("REFUSING TO PUBLISH PUBLICLY")
            print("!" * 72)
            print(f"{why}\n")
            print("Options, in order of how much this repo recommends them:")
            print("  1. --private                 publish to a repo only you can see")
            print("  2. train sft.py on public data, then publish that adapter")
            print("  3. --rights-cleared \"why\"    only if you really hold the rights;")
            print("                              the statement goes into the card")
            raise SystemExit(1)
        if public and kind == "mixed-private" and args.rights_cleared:
            print("\nNOTE: --rights-cleared given for a local-data adapter. The card "
                  "will name\nthe local files by basename and state your claim. "
                  "Verify it is true.")
        if public and kind == "public-mix":
            print("\nNOTE: check every dataset licence in the card before publishing. "
                  "A\nnon-commercial or undeclared dataset inside the mix makes the "
                  "adapter\nnon-commercial or unclear, regardless of the tag below.")

        card_text = build_model_card(info, result, kind, why, args, gate)
        problems = scan(card_text, tokens, allow)
        cfg = os.path.join(args.adapter, "adapter_config.json")
        if os.path.exists(cfg):
            with open(cfg, encoding="utf-8-sig") as f:
                problems += [f"adapter_config.json: {x}"
                             for x in scan(f.read(), tokens, allow)]
        if problems:
            print("\nREFUSING: the files to publish contain private strings:")
            for x in problems[:10]:
                print(f"  - {x}")
            print("\nPass --allow <token> for a deliberate attribution, or fix the "
                  "card.")
            raise SystemExit(1)

        files = [(os.path.join(args.adapter, "adapter_model.safetensors"),
                  "adapter_model.safetensors")]
        if os.path.exists(cfg):
            files.append((cfg, "adapter_config.json"))
        card_path = os.path.join(run_dir, "MODEL_CARD.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_text)
        files.append((card_path, "README.md"))
        print(f"\nwrote {rel(card_path)}")

    if args.dataset:
        if not os.path.exists(args.dataset):
            raise SystemExit(f"--dataset {args.dataset!r} does not exist")
        report = dataset_report(args.dataset)
        problems = dataset_leaks(args.dataset, tokens, allow)
        if problems:
            print("\nREFUSING: the dataset file contains private strings:")
            for x in problems[:5]:
                print(f"  - {x}")
            print("\nThis is the check that stops lore/ or a source dump from being "
                  "uploaded by\naccident. Pass --allow only for a deliberate, "
                  "already-public string.")
            raise SystemExit(1)
        if not args.private and not args.rights_cleared:
            print("\nREFUSING TO PUBLISH A DATASET PUBLICLY without --rights-cleared.")
            print("A dataset card is a licence grant; you have to be able to make it.")
            raise SystemExit(1)
        dataset_card_text = dataset_card(args, report)
        dataset_card_path = os.path.join(
            os.path.dirname(os.path.abspath(args.dataset)), "DATASET_CARD.md")
        with open(dataset_card_path, "w", encoding="utf-8") as f:
            f.write(dataset_card_text)
        print(f"\nwrote {rel(dataset_card_path)}")

    if args.dry_run:
        print("\n" + "=" * 72)
        print("DRY RUN -- nothing uploaded")
        print("=" * 72)
        for path, name in files:
            print(f"  model   {name:<28} {os.path.getsize(path) / 1e6:8.1f} MB")
        if args.dataset:
            print(f"  dataset {os.path.basename(args.dataset):<28} "
                  f"{os.path.getsize(args.dataset) / 1e6:8.1f} MB")
            print(f"  dataset {'README.md (dataset card)':<28} "
                  f"{len(dataset_card_text or '') / 1e3:8.1f} kB")
        print(f"\nrepo: {args.repo}  private={args.private}" if args.adapter
              else f"\nprivate={args.private}")
        if args.dataset:
            print(f"dataset repo: {args.dataset_repo}")
        print("\nRead the card before publishing:")
        if args.adapter:
            print(f"  {rel(os.path.join(run_dir, 'MODEL_CARD.md'))}")
        if args.dataset:
            print(f"  {rel(dataset_card_path)}")
        return 0

    confirm_and_upload(args, files, dataset_card_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
