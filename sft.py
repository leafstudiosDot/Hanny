"""M5 - chat-format SFT: teach Hanny turns, not files.

Why this is not finetune.py
---------------------------
`finetune.py` trains on raw concatenated source text, which is correct for
learning a codebase and wrong for staying a chatbot. A model trained that way
learns to continue a file; asked a question, it may answer in code, emit a
header, or ignore the question entirely -- `chat.py` demonstrates this on the
M4 adapter. This script trains on real turn structure rendered through the base
model's OWN chat template, with loss computed only on assistant tokens. The
model is never taught to generate the user's half of the conversation.

What this does NOT do
---------------------
It is supervised fine-tuning, not pretraining and not preference tuning. It
cannot add knowledge the base model does not have; it changes which behaviours
are likely. A 0.8B model given a broad instruction mix will get better at the
*shape* of an answer, not at knowing more. Measuring whether it did is the job
of `chateval.py`, which this script runs as a subprocess against the finished
adapter -- there is deliberately no second gate implementation here.

Cost, measured rather than guessed: a T4 ran 555 steps of 8x512 tokens in 64
minutes, so ~2M tokens per GPU-hour. Budget 1-3 hours per Kaggle session and
cap with --max-steps / --max-train-tokens.

    # what data would be used, no GPU, no training
    .venv\\Scripts\\python.exe sft.py --dry-run \\
        --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft \\
        --max-per-source 5000

    # local JSONL (see DATASETS.md for the schema)
    .venv\\Scripts\\python.exe sft.py --run m5 --data chat.jsonl \\
        --max-steps 400 --batch-tokens 4096

    # Kaggle T4, mixed public instruction + chat data
    .venv\\Scripts\\python.exe sft.py --run m5 --epochs 1 \\
        --from-hub allenai/tulu-3-sft-mixture \\
        --from-hub HuggingFaceH4/ultrachat_200k::default::train_sft \\
        --max-per-source 40000 --max-steps 1200 --eval-every 100

    # rerun only the gate against an existing adapter; no data, no training
    .venv\\Scripts\\python.exe sft.py --gate-only --run m5-4b-v2 \\
        --model Qwen/Qwen3.5-4B

Privacy: a local file can contain your own projects or lore. `sft.py` records
only its basename and hash, never the full path, because result.json is read by
upload.py and may end up in a model card. Publishing any adapter or dataset
trained on private text is refused by upload.py unless you explicitly claim the
rights.
"""
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from datetime import timedelta

import numpy as np

import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwenkit  # noqa: E402
import hub_sync  # noqa: E402


def is_kaggle():
    """True inside a Kaggle notebook/kernel. Training is Kaggle-only by policy."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or os.environ.get("KAGGLE_URL_BASE")
                or os.path.isdir("/kaggle/working"))

ROLE_MAP = {
    "user": "user", "human": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "chatgpt": "assistant",
    "model": "assistant", "bot": "assistant",
    "system": "system",
}
# Tool calling is a different format contract (the template's native tools=
# argument), and M5b owns it. Rendering these as plain text teaches a syntax the
# template will never ask for, so rows containing them are dropped and counted.
SKIP_ROLES = {"tool", "function", "observation", "ipython", "search", "plugin"}


def content_text(value):
    """Normalise a message body to text, or None if it is not text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                return None
        return "\n".join(parts).strip()
    return None


def normalize_row(row):
    """Any supported row shape -> ([{role, content}], None) or (None, reason).

    Supported: `messages`, `conversations` (ShareGPT), and prompt/response-style
    pairs. Everything else is rejected by name rather than guessed at.
    """
    messages = []
    raw = row.get("messages")
    if raw is None:
        raw = row.get("conversations")
    if isinstance(raw, list):
        for m in raw:
            if not isinstance(m, dict):
                return None, "bad_message"
            role = str(m.get("role") or m.get("from") or "").lower()
            if role in SKIP_ROLES:
                return None, "tool_roles"
            role = ROLE_MAP.get(role)
            body = content_text(m.get("content", m.get("value")))
            if role is None:
                return None, "bad_role"
            if body is None:
                return None, "bad_content"
            if body:
                messages.append({"role": role, "content": body})
    else:
        prompt = row.get("prompt", row.get("instruction", row.get("question")))
        response = row.get("response", row.get("output", row.get("answer")))
        extra = row.get("input")
        if not isinstance(prompt, str) or not isinstance(response, str):
            return None, "unsupported_format"
        prompt = prompt.strip()
        if isinstance(extra, str) and extra.strip():
            prompt = f"{prompt}\n\n{extra.strip()}"
        response = response.strip()
        if not prompt or not response:
            return None, "empty"
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]

    if len(messages) < 2:
        return None, "too_short"
    while messages and messages[0]["role"] == "assistant":
        messages.pop(0)
    if not messages:
        return None, "no_assistant"
    if not any(m["role"] == "assistant" for m in messages):
        return None, "no_assistant"
    return messages, None


def canonical(messages):
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fetch_license(name):
    """Best-effort dataset license from the Hub card. None when unavailable."""
    try:
        from huggingface_hub import dataset_info
        info = dataset_info(name)
        card = getattr(info, "card_data", None)
        if isinstance(card, dict):
            return card.get("license")
        return getattr(card, "license", None)
    except Exception:  # noqa: BLE001 - offline or gated must not stop a run
        return None


def parse_hub_spec(spec):
    """`name::config::split` -> (name, config, split). `::` is safe in HF ids."""
    parts = spec.split("::")
    name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    return name, config, split


def iter_hub(spec, limit, streaming, revision):
    """Yield rows from one Hub dataset, plus a metadata dict for the report."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("--from-hub needs the `datasets` package: "
                         "pip install datasets (do NOT reinstall torch on Kaggle)")
    name, config, split = parse_hub_spec(spec)
    license_ = fetch_license(name)
    meta = {"kind": "hub", "id": name, "config": config, "split": split,
            "revision": revision, "license": license_,
            "sampling": "head" if streaming else "full-split", "limit": limit}
    print(f"  hub {name} [{config or 'default'}/{split}]"
          f"{f' limit {limit}' if limit else ''}"
          f"{f' license {license_}' if license_ else ' license unknown'}")
    ds = load_dataset(name, config, split=split, streaming=streaming,
                      revision=revision)
    rows = 0
    for row in ds:
        yield row, meta
        rows += 1
        if limit and rows >= limit:
            break
    meta["rows_read"] = rows


def iter_jsonl(path):
    meta = {"kind": "local", "name": os.path.basename(path),
            "sha256_16": qwenkit.sha256_16_file(path)}
    print(f"  local {meta['name']} (sha256[:16] {meta['sha256_16']})")
    rows = 0
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), meta
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno} is not valid JSON: {e}")
            rows += 1
    meta["rows_read"] = rows


def trim_context(tok, context, turn, max_len, thinking):
    """Drop whole context turns from the front until context+turn fits.

    The supervised turn itself is never truncated -- half an answer is worse
    than no example -- so if the turn alone is too long the example is dropped.
    Returns (context or None, was_trimmed).
    """
    def size(msgs):
        return len(qwenkit.template_ids(tok, msgs, add_generation_prompt=False,
                                        thinking=thinking))

    ctx = list(context)
    was = False
    if not any(m["role"] == "user" for m in ctx):
        return None, False
    while size(ctx + [turn]) > max_len:
        idx = next((i for i, m in enumerate(ctx) if m["role"] != "system"), None)
        if idx is None:
            return None, True
        dropped = ctx.pop(idx)
        if (dropped["role"] == "user" and idx < len(ctx)
                and ctx[idx]["role"] == "assistant"):
            ctx.pop(idx)
        was = True
        # Qwen's template raises "No user query found" on system-only input, so
        # stop before constructing a context that cannot be rendered at all.
        if not any(m["role"] == "user" for m in ctx):
            return None, was
    while ctx and ctx[-1]["role"] == "assistant":
        ctx.pop()
        was = True
    if not any(m["role"] == "user" for m in ctx) or size(ctx + [turn]) > max_len:
        return None, was
    return ctx, was


def encode_conversation(tok, messages, max_len, thinking):
    """Expand one conversation into one example per assistant turn.

    Why per-turn rather than one masked sequence. Qwen3.5's template inserts the
    empty `<think>\\n\\n</think>\\n\\n` block before the LAST assistant message
    only. In a single rendering of a multi-turn conversation, turn 1 is not
    last, so a prefix rendered up to turn 1 (which makes it last) does not match
    the full rendering -- measured on the real tokenizer, not assumed. Splitting
    at turn boundaries makes every example end with the turn being learned,
    which is also the exact shape generation uses: prompt with the empty think
    block, then the answer. The cost is re-encoding the context once per turn.

    Returns ([(ids, labels)], {reason: count}, n_trimmed).
    """
    out, reasons, trimmed = [], {}, 0

    def bump(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    for i, turn in enumerate(messages):
        if turn["role"] != "assistant":
            continue
        ctx = list(messages[:i])
        if not ctx or ctx[-1]["role"] != "user":
            bump("turn_without_prompt")
            continue
        ctx, was_trimmed = trim_context(tok, ctx, turn, max_len, thinking)
        trimmed += int(was_trimmed)
        if ctx is None:
            bump("turn_too_long")
            continue
        full = qwenkit.template_ids(tok, ctx + [turn], add_generation_prompt=False,
                                    thinking=thinking)
        prefix = qwenkit.template_ids(tok, ctx, add_generation_prompt=True,
                                      thinking=thinking)
        if full[:len(prefix)] != prefix:
            bump("template_prefix_unstable")
            continue
        labels = [-100] * len(full)
        labels[len(prefix):] = full[len(prefix):]
        if all(x == -100 for x in labels):
            bump("no_supervised_tokens")
            continue
        out.append((np.asarray(full, dtype=np.int32),
                    np.asarray(labels, dtype=np.int32)))
    return out, reasons, trimmed


def build_dataset(tok, args):
    """Read, normalise, dedupe, split and encode. Returns (train, val, report)."""
    counts = {k: 0 for k in (
        "rows_read", "conversations", "duplicates", "too_short", "no_assistant",
        "bad_role", "bad_content", "bad_message", "unsupported_format",
        "tool_roles", "empty", "turns_seen", "turns_encoded",
        "turn_without_prompt", "turn_too_long", "template_prefix_unstable",
        "no_supervised_tokens", "trimmed_from_front")}
    sources, seen = [], set()
    examples = []

    iters = []
    if args.data:
        iters += [("jsonl", p) for p in args.data]
    iters += [("hub", s) for s in args.from_hub]

    for kind, spec in iters:
        gen = iter_jsonl(spec) if kind == "jsonl" else iter_hub(
            spec, args.max_per_source, args.streaming, args.hub_revision)
        meta = None
        try:
            while True:
                row, meta = next(gen)
                counts["rows_read"] += 1
                messages, why = normalize_row(row)
                if messages is None:
                    counts[why] += 1
                    continue
                if messages[0]["role"] != "system" and args.system and not args.no_system:
                    messages = [{"role": "system", "content": args.system}] + messages
                key = canonical(messages)
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)
                counts["conversations"] += 1
                examples.append(messages)
        except StopIteration:
            pass
        if meta is not None:
            meta = dict(meta)
            meta.pop("rows_read", None)
            sources.append(meta)

    if not examples:
        raise SystemExit("no usable examples -- check the schema in DATASETS.md")

    # Determinism before encoding: a stable hash decides train/val, so the split
    # does not depend on load order or on which subset a later run reads.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(examples))
    examples = [examples[i] for i in order]
    if args.limit_examples:
        examples = examples[:args.limit_examples]

    train, val = [], []
    for messages in examples:
        digest = qwenkit.sha256_16_text(canonical(messages))
        bucket = int(digest, 16) % 1000
        (val if bucket < args.val_per_mille else train).append(messages)

    def encode_all(rows, label):
        out, tokens, supervised = [], 0, 0
        for i, messages in enumerate(rows):
            counts["turns_seen"] += sum(1 for m in messages if m["role"] == "assistant")
            encoded, reasons, trimmed = encode_conversation(
                tok, messages, args.max_len, args.thinking)
            for reason, n in reasons.items():
                counts[reason] += n
            counts["trimmed_from_front"] += trimmed
            counts["turns_encoded"] += len(encoded)
            for ids, labels in encoded:
                tokens += len(ids)
                supervised += int((labels != -100).sum())
            out.extend(encoded)
            if (i + 1) % 1000 == 0:
                print(f"    {label}: {i + 1:,}/{len(rows):,} conversations -> "
                      f"{len(out):,} examples", flush=True)
        return out, tokens, supervised

    print(f"\nencoding {len(train):,} train / {len(val):,} val conversations ...")
    train_enc, train_tokens, train_sup = encode_all(train, "train")
    val_enc, val_tokens, val_sup = encode_all(val, "val")

    if args.val_per_mille and not val_enc:
        raise SystemExit("the val split is empty. Raise --val-per-mille, or pass "
                         "--val-per-mille 0 to score on the train split (which is "
                         "not a held-out measurement).")
    if not val_enc:
        val_enc, val_tokens, val_sup = train_enc, train_tokens, train_sup
        print("WARNING: val_per_mille=0 -- scoring on training examples. This is "
              "not a held-out number and must not be reported as one.")

    if args.max_train_tokens and train_tokens > args.max_train_tokens:
        keep, used = [], 0
        for ex in train_enc:
            if used + len(ex[0]) > args.max_train_tokens:
                break
            keep.append(ex)
            used += len(ex[0])
        train_enc = keep
        train_tokens = used
        train_sup = sum(int((x[1] != -100).sum()) for x in train_enc)
        print(f"  capped training set at {train_tokens:,} tokens")

    if len(val_enc) > args.eval_examples:
        val_enc = val_enc[:args.eval_examples]
        val_tokens = sum(len(x[0]) for x in val_enc)
        val_sup = sum(int((x[1] != -100).sum()) for x in val_enc)

    if not train_enc:
        raise SystemExit("empty train split after filtering; lower --max-len or "
                         "use less aggressive trimming")

    report = {
        "model": args.model,
        "chat_template_sha16": qwenkit.chat_template_sha16(tok),
        "thinking": args.thinking,
        "max_len": args.max_len,
        "system": None if args.no_system else args.system,
        "val_per_mille": args.val_per_mille,
        "sources": sources,
        "counts": counts,
        "conversations": {"train": len(train), "val": len(val)},
        "examples": {"train": len(train_enc), "val": len(val_enc)},
        "tokens": {"train": train_tokens, "val": val_tokens,
                   "supervised_train": train_sup,
                   "supervised_val": val_sup},
        "vocab": len(tok),
    }
    return train_enc, val_enc, report


def print_report(report, train, val, tok):
    c = report["counts"]
    print("\n" + "=" * 70)
    print(f"rows read  {c['rows_read']:,}   conversations {c['conversations']:,}   "
          f"duplicates {c['duplicates']:,}")
    skipped_keys = {"rows_read", "conversations", "duplicates", "turns_seen",
                    "turns_encoded", "trimmed_from_front"}
    dropped = {k: v for k, v in c.items() if v and k not in skipped_keys}
    if dropped:
        print("dropped    " + "  ".join(f"{k} {v:,}" for k, v in sorted(dropped.items())))
    if c["trimmed_from_front"]:
        print(f"trimmed    {c['trimmed_from_front']:,} turns had context cut from "
              f"the front to fit --max-len")
    print(f"turns      {c['turns_seen']:,} assistant turns -> "
          f"{c['turns_encoded']:,} training examples "
          f"(one per turn, with its context)")
    t = report["tokens"]
    print(f"examples   {report['examples']['train']:,} train / "
          f"{report['examples']['val']:,} val")
    print(f"tokens     {t['train']:,} train ({t['supervised_train']:,} supervised) | "
          f"{t['val']:,} val")
    print(f"template   {report['chat_template_sha16']} | thinking {report['thinking']} "
          f"| vocab {report['vocab']:,}")
    print("=" * 70)

    ids, labels = train[0]
    n_sup = int((labels != -100).sum())
    first = int(np.argmax(labels != -100))
    print(f"\nsample 1 of the training set: {len(ids)} tokens, {n_sup} supervised")
    sup = ids[labels != -100]
    print("  PROMPT up to the first assistant turn (loss ignored):")
    print("    " + tok.decode(ids[:first]).replace("\n", "\n    "))
    print("  SUPERVISED (loss computed here):")
    print("    " + tok.decode(sup).replace("\n", "\n    "))


def mean_se(values):
    """Mean and standard error, ignoring NaN. se is NaN below two values."""
    clean = [x for x in values if x == x]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) < 2:
        return float(clean[0]), float("nan")
    return (float(np.mean(clean)),
            float(np.std(clean, ddof=1) / math.sqrt(len(clean))))


def make_batches(n, lengths, max_tokens, rng):
    """Group similar-length examples under a token cap, then shuffle batch order.

    Padding is wasted compute, and for SFT the waste is large: a mix of 30-token
    and 900-token conversations padded to the longest one per batch can spend
    most of its FLOPs on pad tokens.
    """
    order = sorted(range(n), key=lambda i: lengths[i])
    batches, cur, used = [], [], 0
    for i in order:
        if cur and used + lengths[i] > max_tokens:
            batches.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += lengths[i]
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def pad_batch(examples, pad_id):
    import torch

    width = max(len(ids) for ids, _ in examples)
    xb = torch.full((len(examples), width), pad_id, dtype=torch.long)
    yb = torch.full((len(examples), width), -100, dtype=torch.long)
    am = torch.zeros((len(examples), width), dtype=torch.long)
    for row, (ids, labels) in enumerate(examples):
        xb[row, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        yb[row, :len(labels)] = torch.as_tensor(labels, dtype=torch.long)
        am[row, :len(ids)] = 1
    return xb, yb, am


def score_examples(model, examples, pad_id, max_tokens, device):
    """Per-example mean loss over supervised tokens. Order is preserved.

    Per-example rather than one pooled number because base-vs-adapter on the
    same examples is paired, and `evalkit.paired_se` needs the per-item values
    to cancel the shared part of the two models' errors.
    """
    import torch
    import torch.nn.functional as F

    batches = make_batches(len(examples), [len(e[0]) for e in examples],
                           max_tokens, np.random.default_rng(0))
    out = [float("nan")] * len(examples)
    with torch.no_grad():
        for b in batches:
            xb, yb, am = pad_batch([examples[i] for i in b], pad_id)
            xb, yb, am = xb.to(device), yb.to(device), am.to(device)
            logits = model(input_ids=xb, attention_mask=am).logits
            # Causal LM: logits at position t predict the token at t+1. The
            # training path gets this shift from `labels=yb`; scoring must do
            # it by hand. Without the shift the eval compares logits[t] with
            # labels[t] and reports a loss worse than uniform (the 4B run read
            # 16.3 nats where ln(vocab) is 12.4), which is not a measurement.
            shift_logits = logits[:, :-1, :]
            shift_labels = yb[:, 1:]
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100, reduction="none").reshape(shift_labels.shape)
            for row, i in enumerate(b):
                mask = shift_labels[row] != -100
                n = int(mask.sum())
                if n:
                    out[i] = float(token_loss[row][mask].mean())
            del logits, shift_logits, shift_labels, token_loss
    return out


def write_resume_point(out, core_model, state):
    """adapter-last + train_state.pt, written together. train_state.pt goes via a
    temp file and os.replace so a kill mid-write never leaves a torn file."""
    import torch

    core_model.save_pretrained(os.path.join(out, "adapter-last"))
    tmp = os.path.join(out, "train_state.pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(out, "train_state.pt"))


def run_gate(out, model_name, precision, system=None, env=None):
    """Run chateval.py against an existing run folder's adapter-best.

    Returns (relative gate path or None, exit code). The gate is a separate
    process by design: it loads its own 4-bit copy of the base model, and the
    training process must have released its model first. `--gate-only` calls
    this without training, so a failed gate does not cost another T4 run.

    system must be the string the run trained with. `None` means the run was
    trained with --no-system; chateval needs the empty flag to reproduce that,
    not its own default prompt.

    env lets the DDP parent strip RANK/WORLD_SIZE/etc. before spawning the
    single-process gate; inheriting torchrun's variables is a silent way to
    make the child look distributed when it is not.
    """
    gate_dir = os.path.join(out, "gate")
    cmd = [sys.executable, os.path.join(HERE, "chateval.py"),
           "--model", model_name, "--precision", precision,
           "--adapter", os.path.join(out, "adapter-best"),
           "--system", system or "",
           "--out", gate_dir]
    print("\nrunning the M5 gate (base vs adapter, one process, same precision):")
    print("  " + " ".join(cmd))
    gate_exit = subprocess.call(cmd, env=env)
    gate_path = None
    gate_result = os.path.join(gate_dir, "result.json")
    if gate_exit == 0 and os.path.exists(gate_result):
        gate_path = os.path.relpath(gate_result, HERE)
    elif gate_exit == 0:
        print("WARNING: chateval.py exited 0 but wrote no result.json; "
              "no gate number is quoted.")
        gate_exit = 1
    else:
        print(f"WARNING: chateval.py exited {gate_exit}; no gate number is quoted.")
    return gate_path, gate_exit


def main():
    p = argparse.ArgumentParser(
        description="Chat-format SFT through the base model's own chat template.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=qwenkit.DEFAULT_MODEL)
    p.add_argument("--run", default=time.strftime("m5-%Y%m%d-%H%M"))
    p.add_argument("--precision", default="4-bit nf4", choices=list(qwenkit.PRECISIONS))
    p.add_argument("--data", action="append", default=[], metavar="JSONL",
                   help="repeatable local JSONL; schema in DATASETS.md")
    p.add_argument("--from-hub", action="append", default=[], metavar="NAME[::CONFIG[::SPLIT]]",
                   help="repeatable Hub dataset; needs the datasets package")
    p.add_argument("--hub-revision", default=None)
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                   help="stream the Hub split instead of downloading it (default on)")
    p.add_argument("--max-per-source", type=int, default=20000,
                   help="rows taken per source; streaming reads the head of the "
                        "split, which the report records as sampling='head'")
    p.add_argument("--limit-examples", type=int, default=0,
                   help="cap conversations before expansion; 0 = all kept rows")
    p.add_argument("--max-train-tokens", type=int, default=0, help="0 = no cap")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--max-tokens-per-batch", type=int, default=4096,
                   help="the memory knob: ~4096 tokens is what M4 fit on a T4")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--batch-tokens", type=int, default=0,
                   help="alias for --max-tokens-per-batch (the two must agree)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="0 = run all epochs")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-examples", type=int, default=128)
    p.add_argument("--val-per-mille", type=int, default=10,
                   help="val share x1000, assigned by hash of the example")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--system", default=qwenkit.DEFAULT_SYSTEM,
                   help="added only to examples that have no system message")
    p.add_argument("--no-system", action="store_true", help="never inject one")
    p.add_argument("--thinking", action="store_true",
                   help="train with reasoning on. NOT supported: Qwen's template "
                        "makes the assistant prefix non-prefix-stable in that mode, "
                        "so examples would be masked wrongly.")
    p.add_argument("--init-adapter", help="start from an existing adapter (stacks; "
                                          "attribution becomes ambiguous)")
    p.add_argument("--resume", action="store_true",
                   help="continue an INTERRUPTED run of the same --run from "
                        "checkpoints/<run>/train_state.pt + adapter-last (optimizer, "
                        "LR schedule and data position restored). Needs identical "
                        "--max-steps/--seed/--max-tokens-per-batch/world size and "
                        "--epochs 1. To go past a finished run use --init-adapter.")
    p.add_argument("--save-every", type=int, default=50,
                   help="every N optimizer steps write adapter-last + train_state.pt "
                        "and push them to --hub-repo (0 = only at the end)")
    p.add_argument("--hub-repo", default=os.environ.get("HANNY_HUB_REPO"),
                   help="PRIVATE Hugging Face model repo (user/name) that mirrors "
                        "checkpoints so they survive the Kaggle session. Needs "
                        "HF_TOKEN. Default: $HANNY_HUB_REPO")
    p.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True,
                   help="run chateval.py on the best adapter afterwards")
    p.add_argument("--gate-only", action="store_true",
                   help="run only the gate against checkpoints/<run>/adapter-best; "
                        "no data build, no training, and result.json is updated")
    p.add_argument("--ddp", action="store_true",
                   help="Distributed Data Parallel, one process per GPU. Launch "
                        "with: python -m torch.distributed.run --standalone "
                        "--nproc_per_node=2 sft.py --ddp ...")
    p.add_argument("--allow-local-training", action="store_true",
                   help="override the Kaggle-only guard. For a deliberate smoke "
                        "test only; the local 4 GB card cannot train the 4B model.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dry-run", action="store_true",
                   help="build and report the dataset, write nothing downstream")
    args = p.parse_args()

    if args.ddp and args.gate_only:
        raise SystemExit("--ddp is a training mode; --gate-only is single-process.")
    if args.ddp and args.dry_run:
        raise SystemExit("--ddp is for an actual training run; --dry-run already "
                         "builds the dataset without a model.")
    if args.ddp and args.grad_accum != 1:
        raise SystemExit("DDP currently requires --grad-accum 1. With variable "
                         "token batches, per-microbatch gradient scaling across "
                         "ranks would need a second all-reduce per step; not "
                         "implemented.")

    ddp = bool(args.ddp)
    rank, world_size, local_rank = 0, 1, 0
    if ddp:
        import torch
        import torch.distributed as dist

        if "WORLD_SIZE" not in os.environ:
            raise SystemExit(
                "--ddp must be launched with torchrun, e.g.\n"
                "  python -m torch.distributed.run --standalone "
                "--nproc_per_node=2 sft.py --ddp ...")
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size < 1 or rank < 0 or local_rank < 0:
            raise SystemExit(f"invalid DDP ranks: RANK={rank} "
                             f"LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(
                f"LOCAL_RANK={local_rank} but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible")
        if not is_kaggle() and not args.allow_local_training:
            raise SystemExit(
                "DDP training is Kaggle-only by policy. Use a Kaggle T4 x2 "
                "session, or pass --allow-local-training for a deliberate smoke "
                "test.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        if rank != 0:
            # Keep stderr for real errors; suppress the duplicate data-build and
            # training progress from worker ranks.
            sys.stdout = open(os.devnull, "w")
        elif world_size == 1:
            print("note: WORLD_SIZE=1 -- DDP smoke path, not a speedup.")

    if args.gate_only:
        if args.ddp:
            raise SystemExit("--gate-only cannot be combined with --ddp")
        out = os.path.join(HERE, "checkpoints", args.run)
        adapter = os.path.join(out, "adapter-best")
        if not os.path.isdir(adapter):
            raise SystemExit(
                f"{os.path.relpath(adapter, HERE)} does not exist. --gate-only "
                "needs an already-trained run; point --run at it.")
        result_path = os.path.join(out, "result.json")
        result = None
        if os.path.exists(result_path):
            # utf-8-sig: PowerShell 5.1 writes a BOM, and a valid result.json
            # must not read as missing. Writers stay plain UTF-8.
            with open(result_path, encoding="utf-8-sig") as f:
                result = json.load(f)
            # An adapter is not portable across bases or precisions. Catch the
            # common mistake (forgetting --model on a 4B run) before spending
            # minutes loading the wrong weights.
            for key, given in (("model", args.model), ("precision", args.precision)):
                recorded = result.get(key)
                if recorded and recorded != given:
                    raise SystemExit(
                        f"--gate-only {key} {given!r} does not match this run's "
                        f"recorded {key} {recorded!r}. Pass the recorded value.")
        if result is not None and "system" in result:
            system = result["system"]
        else:
            system = qwenkit.DEFAULT_SYSTEM
        if result is not None and "system" not in result:
            print("note: this run predates system recording; assuming the "
                  "default system prompt. Pass --no-system only if that is "
                  "what it trained with.")
        gate_path, gate_exit = run_gate(out, args.model, args.precision, system)
        if result is not None:
            result["system"] = system
            result["gate"] = gate_path
            result["gate_exit"] = gate_exit
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"updated {os.path.relpath(result_path, HERE)}")
        return 0 if gate_exit == 0 else (gate_exit or 1)

    if args.thinking:
        raise SystemExit(
            "sft.py refuses --thinking: Qwen3.5's template only becomes "
            "prefix-stable once the think block is closed, so assistant-only "
            "masking is exact with thinking OFF and wrong with it ON. Training "
            "reasoning traces needs a different data format; not implemented.")
    if args.batch_tokens:
        if args.max_tokens_per_batch not in (4096, args.batch_tokens):
            raise SystemExit("--batch-tokens and --max-tokens-per-batch disagree")
        args.max_tokens_per_batch = args.batch_tokens
    if args.max_tokens_per_batch < args.max_len:
        raise SystemExit("--max-tokens-per-batch must be >= --max-len, or a single "
                         "example can never fill a batch")

    kaggle = is_kaggle()
    if not args.dry_run and not kaggle and not args.allow_local_training:
        raise SystemExit(
            "Training is Kaggle-only by policy. The local GTX 1650 has 4 GB and "
            "cannot fit the 4B optimizer/activations; use a Kaggle T4 x2 "
            "session. Pass --allow-local-training only for a deliberate smoke "
            "test, never for a result you intend to quote.")

    from transformers import AutoTokenizer
    tok = qwenkit.configure_tokenizer(AutoTokenizer.from_pretrained(args.model))
    print(f"tokenizer {args.model} | template {qwenkit.chat_template_sha16(tok)}")
    print("\nsources:" if (args.data or args.from_hub) else "\nno sources given")

    out = os.path.join(HERE, "checkpoints", args.run)
    os.makedirs(out, exist_ok=True)

    state_path = os.path.join(out, "train_state.pt")
    sync = hub_sync.HubSync(args.hub_repo, args.run)
    if args.resume:
        if args.init_adapter:
            raise SystemExit("--resume restores an interrupted run; --init-adapter "
                             "starts a new one from old weights. Pick one.")
        if args.epochs != 1:
            raise SystemExit("--resume supports --epochs 1 only; extend the run "
                             "with --max-steps instead.")
        if not (os.path.exists(state_path)
                and os.path.isdir(os.path.join(out, "adapter-last"))):
            raise SystemExit(f"--resume: {out} has no train_state.pt + adapter-last. "
                             "Pull them from the Hub first (hub_sync.pull).")
    elif args.save_every and not sync.enabled:
        print("\nWARNING: no --hub-repo/HF_TOKEN. Checkpoints stay on this machine "
              "only and are lost when the Kaggle session ends.")

    if ddp:
        # Build on rank 0 only, then share the exact object. Two independent
        # streaming passes are not guaranteed to see the same rows in the same
        # order; if their batch counts differ, one rank waits forever in DDP's
        # all-reduce. This also avoids tokenising 89M tokens twice.
        cache = os.path.join(out, ".ddp_data.pt")
        if rank == 0:
            print("building the DDP dataset on rank 0 ...")
            train, val, report = build_dataset(tok, args)
            print_report(report, train, val, tok)
            torch.save({"train": train, "val": val, "report": report}, cache)
        dist.barrier()
        if rank != 0:
            payload = torch.load(cache, map_location="cpu", weights_only=False)
            train, val, report = payload["train"], payload["val"], payload["report"]
        dist.barrier()
        if rank == 0:
            os.remove(cache)
    else:
        train, val, report = build_dataset(tok, args)
        print_report(report, train, val, tok)

    if rank == 0:
        with open(os.path.join(out, "data_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {os.path.relpath(os.path.join(out, 'data_report.json'), HERE)}")

    if args.dry_run:
        print("\n--dry-run: dataset built, nothing trained.")
        return 0

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise SystemExit("SFT needs a GPU; a 1650 cannot fit the optimizer state "
                         "(see README 'Hardware limits'). Use Kaggle, or --dry-run.")
    device = f"cuda:{local_rank}" if ddp else "cuda"
    # Rank 0's model state is broadcast by DDP at construction. Different seeds
    # for the other ranks keep dropout masks from being identical; reproducible
    # only at the same WORLD_SIZE, which is recorded in result.json.
    torch.manual_seed(args.seed + (rank if ddp else 0))
    np.random.seed(args.seed)

    print(f"\nloading {args.model} at {args.precision} on {device} ...")
    model, tok = qwenkit.load(args.model, args.precision, device, tok=tok)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Score the untouched base on the same val examples BEFORE the LoRA wrapper.
    # This is the M4 discipline: the comparison is within one run and one
    # precision, so quantisation cancels out. Only rank 0 needs the number; the
    # other ranks have no result.json to write and would duplicate the work.
    before = None
    before_mean = before_se = float("nan")
    if rank == 0:
        print("scoring base model on the held-out examples ...")
        before = score_examples(model, val, tok.pad_token_id,
                                args.max_tokens_per_batch, device)
        before_mean, before_se = mean_se(before)
        print(f"BEFORE  held-out loss {before_mean:.4f} +- {before_se:.4f}")
        # A chat model can be bad, but it cannot be worse than a uniform guess
        # by accident on this data. A number above ln(vocab) is the signature of
        # an eval misalignment (the first 4B run scored unshifted logits at 16.3
        # nats, against ln(248077)=12.4). Refuse to train on a metric that
        # cannot be read.
        uniform = math.log(len(tok))
        if not math.isfinite(before_mean) or before_mean > uniform:
            raise SystemExit(
                f"base held-out loss {before_mean:.3f} is not below uniform "
                f"{uniform:.3f}. The eval is misaligned or the model is not a "
                "chat model; fix the measurement before spending GPU time on "
                "training.")

    targets, missing = qwenkit.lora_targets(model)
    print(f"\nLoRA targets: {targets}")
    if missing:
        print(f"  (not present in this architecture: {missing})")
    if not targets:
        raise SystemExit("found no LoRA target modules -- architecture changed?")

    resume_state = None
    load_from = args.init_adapter
    if args.resume:
        # Own file, but it may have come from the Hub: tensors and plain
        # containers only, no arbitrary unpickling.
        resume_state = torch.load(state_path, map_location="cpu", weights_only=True)
        load_from = os.path.join(out, "adapter-last")
        print(f"\nRESUMING {args.run} at step {resume_state['step']} from {load_from}")
    if load_from:
        if not args.resume:
            print(f"\nWARNING: starting from {args.init_adapter}. The finished adapter "
                  f"will contain this one's changes too, so its result cannot be "
                  f"attributed to this dataset alone.")
        model = PeftModel.from_pretrained(model, load_from, is_trainable=True)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
    model.print_trainable_parameters()

    core_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

    lengths = [len(x[0]) for x in train]
    mean_len = sum(lengths) / len(lengths)
    approx_batch = max(1, int(args.max_tokens_per_batch // mean_len))
    batches_per_epoch = max(1, math.ceil(len(train) / approx_batch))
    # One optimizer step consumes one local batch on every rank, so the global
    # number of steps is divided by WORLD_SIZE. For WORLD_SIZE=1 this is the
    # unchanged single-GPU estimate.
    steps_per_epoch = max(
        1, math.ceil(batches_per_epoch / (args.grad_accum * world_size)))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"\n{len(train):,} examples, mean {mean_len:.0f} tokens -> "
          f"~{approx_batch} per batch | {total_steps:,} optimizer steps")

    vocab = len(tok)
    logit_gb = args.max_tokens_per_batch * vocab * 2 / 1e9
    softmax_gb = args.max_tokens_per_batch * vocab * 4 / 1e9
    print(f"memory estimate: ~{logit_gb:.1f} GB logits + ~{softmax_gb:.1f} GB "
          f"fp32 softmax workspace at {args.max_tokens_per_batch} tokens/batch")
    if logit_gb + softmax_gb > 10:
        print("  WARNING: above what a 16 GB T4 comfortably holds with activations; "
              "lower --max-tokens-per-batch")

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total_steps),
        pct_start=0.03, anneal_strategy="cos", cycle_momentum=False,
        div_factor=10, final_div_factor=10)

    # Anything that changes the batch order or the LR schedule makes a resume
    # silently wrong, so it is compared, not assumed.
    meta = {"total_steps": total_steps, "world_size": world_size,
            "tokens_per_batch": args.max_tokens_per_batch, "seed": args.seed,
            "grad_accum": args.grad_accum, "n_train": len(train)}
    start_step, prior_seconds = 0, 0.0
    if resume_state is not None:
        if resume_state["meta"] != meta:
            raise SystemExit(
                "cannot resume: this run's setup differs from the checkpoint's.\n"
                f"  checkpoint: {resume_state['meta']}\n  now:        {meta}")
        opt.load_state_dict(resume_state["opt"])
        sched.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"])
        prior_seconds = float(resume_state["seconds"])

    curve = os.path.join(out, "curve.csv")
    if rank == 0 and resume_state is None:
        with open(curve, "w", encoding="utf-8") as f:
            f.write("step,train_loss,val_loss,val_se,lr,seconds,tokens\n")

    if rank == 0:

        print("\n" + "=" * 78)
        print(f"{'step':>6} {'train':>8} {'val':>8} {'se':>7} {'lr':>9} "
              f"{'time':>7} {'tokens':>10}")
        print("=" * 78)

    rng = np.random.default_rng(args.seed)
    model.train()
    t0 = time.time()
    running, step, tokens_seen, tokens_since_log = [], start_step, 0, 0
    best_val, best_step, best_per_example = float("inf"), 0, None
    if resume_state is not None:
        tokens_seen = int(resume_state["tokens_seen"])
        best_val = float(resume_state["best_val"])
        best_step = int(resume_state["best_step"])
        best_per_example = resume_state["best_per_example"]
    stop = False

    def make_state():
        # tokens_since_log is this rank's count only; x world_size is an
        # approximation, good enough for a resume (curve.csv is the record).
        return {
            "step": step, "meta": meta,
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "tokens_seen": tokens_seen + tokens_since_log * world_size,
            "best_val": best_val, "best_step": best_step,
            "best_per_example": (None if best_per_example is None
                                 else [float(x) for x in best_per_example]),
            "seconds": prior_seconds + (time.time() - t0),
        }

    for epoch in range(args.epochs):
        batches = make_batches(len(train), lengths, args.max_tokens_per_batch, rng)
        if ddp:
            # Each rank gets an equal number of disjoint batches. Dropping the
            # remainder is deliberate: one rank taking one extra step would
            # leave the other rank waiting in DDP's all-reduce forever.
            per_rank = len(batches) // world_size
            if per_rank < 1:
                raise RuntimeError(
                    f"only {len(batches)} batch(es) for WORLD_SIZE={world_size}; "
                    "raise the dataset or lower --max-tokens-per-batch")
            batches = batches[:per_rank * world_size]
            batches = batches[rank * per_rank:(rank + 1) * per_rank]

        for bi, b in enumerate(batches):
            if bi < start_step * args.grad_accum:
                continue  # resume: this batch was already trained on
            xb, yb, am = pad_batch([train[i] for i in b], tok.pad_token_id)
            xb, yb, am = xb.to(device), yb.to(device), am.to(device)
            loss = model(input_ids=xb, attention_mask=am, labels=yb).loss
            local_tokens = int(am.sum())
            # The model's loss is a mean over supervised target tokens, not over
            # every attention token. DDP's gradient average must therefore be
            # weighted by the number of labels != -100 (after the causal shift),
            # or a rank whose batch has a higher prompt/padding fraction is
            # silently over-weighted. --grad-accum is fixed at 1 in this path.
            local_sup = int((yb[:, 1:] != -100).sum())
            if local_sup < 1:
                raise RuntimeError("a DDP batch had no supervised tokens; "
                                   "refusing to average a NaN loss")

            if ddp:
                # DDP averages gradients across ranks. Local losses are means
                # over local supervised tokens; scaling by local/global *
                # world_size makes that average equal the global supervised-
                # token mean. Without it, a rank with a shorter final batch
                # would be over-weighted.
                token_t = torch.tensor([local_sup], device=device,
                                       dtype=torch.float32)
                dist.all_reduce(token_t, op=dist.ReduceOp.SUM)
                global_tokens = max(1.0, float(token_t.item()))
                scaled_loss = loss * (local_sup / global_tokens) * world_size
            else:
                scaled_loss = loss
            (scaled_loss / args.grad_accum).backward()
            # Keep the supervised-token weight with the loss so the logged
            # train_loss is a global token mean, not a mean of rank means.
            running.append((float(loss.detach()), local_sup))
            tokens_since_log += local_tokens

            last = bi == len(batches) - 1
            if (bi + 1) % args.grad_accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.eval_every == 0 or step == total_steps or last:
                    recent = running[-50:]
                    local_loss = sum(loss_v * n for loss_v, n in recent)
                    local_n = sum(n for _, n in recent)
                    stats = torch.tensor(
                        [local_loss, float(local_n), float(tokens_since_log)],
                        device=device, dtype=torch.float32)
                    if ddp:
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    tr = float(stats[0].item()) / max(1.0, float(stats[1].item()))
                    tokens_seen += int(stats[2].item())
                    tokens_since_log = 0

                    # Evaluation and saving are rank-0 only. Other ranks go on
                    # to the next forward pass and wait in DDP's next backward;
                    # no extra barrier is needed, and DDP's 60-minute timeout is
                    # comfortably above this 128-example eval.
                    if rank == 0:
                        was_training = model.training
                        model.eval()
                        per_ex = score_examples(core_model, val, tok.pad_token_id,
                                                args.max_tokens_per_batch, device)
                        model.train(was_training)
                        va, vse = mean_se(per_ex)
                        star = ""
                        if va < best_val:
                            best_val, best_step, best_per_example = va, step, per_ex
                            core_model.save_pretrained(
                                os.path.join(out, "adapter-best"))
                            star = "  <- best"
                        print(f"{step:>6} {tr:>8.4f} {va:>8.4f} {vse:>7.4f} "
                              f"{sched.get_last_lr()[0]:>9.2e} "
                              f"{time.time()-t0:>6.0f}s {tokens_seen:>10,}{star}")
                        with open(curve, "a", encoding="utf-8") as f:
                            f.write(f"{step},{tr:.4f},{va:.4f},{vse:.4f},"
                                    f"{sched.get_last_lr()[0]:.3e},"
                                    f"{time.time()-t0:.1f},{tokens_seen}\n")
                if (rank == 0 and args.save_every and step % args.save_every == 0
                        and step < total_steps):
                    # Rank 0 only; the other rank just waits in the next all-reduce.
                    write_resume_point(out, core_model, make_state())
                    sync.push(out, step)  # snapshot now, upload in the background
                if step >= total_steps:
                    stop = True
                    break
        if stop:
            break

    if rank == 0 and best_step == 0:
        model.eval()
        best_per_example = score_examples(core_model, val, tok.pad_token_id,
                                          args.max_tokens_per_batch, device)
        best_val, _ = mean_se(best_per_example)
        best_step = step
        core_model.save_pretrained(os.path.join(out, "adapter-best"))

    if rank == 0:
        write_resume_point(out, core_model, make_state())
        # Weights are safe off-box before the gate runs (the gate has OOMed
        # before, and a crash must not cost the trained adapter).
        sync.push(out, step, wait=True)
    seconds = prior_seconds + (time.time() - t0)

    # chateval.py loads its own 4-bit copy of the base model. This process still
    # holds the training model, optimizer and last batch; on a 15 GB T4 the 4B
    # child then OOMs while the parent holds 13.5 GB (seen on the first m5-4b
    # run, which left gate=null). Free them before spawning it -- on every DDP
    # rank, so rank 1's card is free while rank 0 runs the gate.
    try:
        del xb, yb, am, loss
    except NameError:
        pass
    try:
        del sched
    except NameError:
        pass
    try:
        del opt
    except NameError:
        pass
    try:
        del params
    except NameError:
        pass
    try:
        del model
    except NameError:
        pass
    try:
        del core_model
    except NameError:
        pass
    try:
        # The tokenised examples are no longer needed, and the gate loads its
        # own model. On a 30 GB Kaggle host this matters: the corpus is still
        # hundreds of megabytes and the gate child needs room for the base copy.
        del train, val
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    if ddp:
        dist.barrier()

    gate_path = None
    gate_exit = None
    if rank == 0:
        import evalkit
        d, se = evalkit.paired_se(before, best_per_example)
        print("\n" + "=" * 78)
        print(f"BEFORE  {before_mean:.4f} +- {before_se:.4f}  "
              "(4-bit base, same val set)")
        print(f"AFTER   {best_val:.4f}          (best at step {best_step} of {step})")
        print(f"CHANGE  {-d:+.4f} nats paired  (paired se {se:.4f}; "
              f"perplexity {math.exp(before_mean):.3f} -> "
              f"{math.exp(best_val):.3f})")
        print("=" * 78)
        if abs(d) < 2 * se:
            print("The paired change is inside 2x its standard error. That is not "
                  "evidence\nof an effect on this val set -- report it as "
                  "inconclusive, not as a win.")
        print("\nHeld-out loss is from the same distribution as training data. It "
              "says the\nformat was learned; it cannot say the model became a "
              "better assistant.\nThat question belongs to chateval.py, which is "
              "the only gate this repo has.")

        result = {
            "model": args.model,
            "run": args.run,
            "kind": "chat-sft",
            "chat_template_sha16": report["chat_template_sha16"],
            "thinking": False,
            "system": report["system"],
            "sources": report["sources"],
            "data_counts": report["counts"],
            "examples": report["examples"],
            "tokens": {"train_seen": tokens_seen, **report["tokens"]},
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "init_adapter": args.init_adapter,
            "resumed_from_step": start_step or None,
            "hub_repo": args.hub_repo,
            "lr": args.lr,
            "epochs": args.epochs,
            "max_len": args.max_len,
            "max_tokens_per_batch": args.max_tokens_per_batch,
            "grad_accum": args.grad_accum,
            "steps_done": step,
            "best_step": best_step,
            "loss_before": round(before_mean, 6),
            "loss_before_se": round(before_se, 6),
            "loss_after": round(best_val, 6),
            "loss_change": round(-d, 6),
            "paired_se": round(se, 6),
            "precision": args.precision,
            "kernels": qwenkit.kernels(),
            "seed": args.seed,
            "seconds": round(seconds, 1),
            "ddp": ddp,
            "world_size": world_size,
            "launcher": "torch.distributed.run" if ddp else "python",
            "platform": "kaggle" if kaggle else "local",
            "allow_local_training": bool(args.allow_local_training),
        }

        if args.gate:
            # Strip torchrun's environment before spawning the single-process
            # gate. chateval.py does not initialise DDP, but inheriting RANK
            # would make an accidental future distributed path look active.
            gate_env = None
            if ddp:
                drop = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                        "GROUP_RANK", "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT"}
                gate_env = {k: v for k, v in os.environ.items() if k not in drop}
            gate_path, gate_exit = run_gate(
                out, args.model, args.precision, report["system"], env=gate_env)

        result["gate"] = gate_path
        result["gate_exit"] = gate_exit
        with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {os.path.relpath(out, HERE)}/result.json, curve.csv, "
              f"adapter-best/, adapter-last/")

        adapter = os.path.relpath(os.path.join(out, "adapter-best"), HERE)
        print("\ntry it:")
        print(f"  .venv\\Scripts\\python.exe chat.py --adapter {adapter}")
        print("publish it (dry run first; upload.py will refuse a private-data "
              "adapter):")
        print(f"  .venv\\Scripts\\python.exe upload.py --adapter {adapter} --dry-run")

    if rank == 0:
        # result.json exists now. gate/ is deliberately not pushed (private probes).
        sync.push(out, step, wait=True)

    if ddp:
        # Rank 1 waits here while rank 0 runs the gate, keeping torchrun's
        # worker set alive until both sides have finished. It then exits 0.
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return 0
    return 0 if gate_exit in (None, 0) else (gate_exit or 1)


if __name__ == "__main__":
    raise SystemExit(main())