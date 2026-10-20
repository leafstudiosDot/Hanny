"""Talk to Hanny: the base model, one adapter, or several, in one process.

    .venv\\Scripts\\python.exe chat.py
    .venv\\Scripts\\python.exe chat.py --adapter checkpoints/m4/adapter-best
    .venv\\Scripts\\python.exe chat.py --adapter chat=checkpoints/m5/adapter-best --ask "hi"
    .venv\\Scripts\\python.exe chat.py --adapter m4=checkpoints/m4/adapter-best \\
                            --adapter chat=checkpoints/m5/adapter-best --use chat

Why this exists separately from chateval.py: chateval scores a fixed probe set
and must stay deterministic and frozen -- it is the gate. This is the thing you
sit in front of and type into, so it may sample, stream and keep history. It
must never become the place a number is quoted from; that is what the gate is
for.

Read this before judging the M4 adapter's conversation:

    M4 was trained on raw concatenated source text, not on chat turns. The
    base model's chat template still applies (it is the same tokenizer), but the
    adapter learned "continue this file", not "answer this question". A code
    adapter will frequently continue code, ignore the question, or emit a file
    header. That is the documented M4 result, not a bug in this script.
    `chat.py` prints a warning when the adapter's result.json has no
    chat_template_sha16, which is exactly the signature of that training data.

Commands inside the REPL: /help, /reset, /system, /temp, /max, /thinking,
/adapter, /adapters, /save, /quit. Ctrl+C stops a reply without ending the
session; Ctrl+C at the prompt exits.
"""
import argparse
import json
import os
import sys
import threading
import time

import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwenkit  # noqa: E402


class StopOnEvent:
    """Break generate() as soon as the event is set. Used by Ctrl+C."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def build(model_name, precision, adapter_specs, use):
    """Load base + named adapters. Returns (model, tok, active_name)."""
    model, tok = qwenkit.load(model_name, precision)
    names = []
    if not adapter_specs:
        return model, tok, None

    from peft import PeftModel

    for spec in adapter_specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = os.path.basename(os.path.dirname(os.path.abspath(path))) or "adapter"
        if not os.path.isdir(path):
            raise SystemExit(f"adapter path {path!r} is not a directory; point at "
                             f"the adapter-best folder itself")
        info = qwenkit.adapter_identity(path)
        if "error" in info:
            raise SystemExit(f"cannot fingerprint {path!r}: {info['error']}")
        print(f"adapter {name}: sha256[:16] {info['sha256_16']}  "
              f"({info['bytes']/1e6:.1f} MB)")
        if info.get("base_model") and info["base_model"] != model_name:
            print(f"  WARNING: adapter was trained on {info['base_model']!r}, "
                  f"not {model_name!r}. Weights will load; the result is not "
                  f"a fine-tune of the model you are running.")
        produced = info.get("produced_by")
        if isinstance(produced, dict):
            print(f"  produced by run {produced.get('run')!r} "
                  f"(loss_after {produced.get('loss_after')})")
            if not produced.get("chat_template_sha16"):
                print("  NOTE: no chat_template_sha16 in that run -- it was trained on")
                print("        raw text, not chat turns (M4). Expect code continuation,")
                print("        not conversation. See this file's docstring.")
        elif produced is None:
            print("  NOTE: no result.json beside this adapter -- provenance unknown.")

        if not names:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
        names.append(name)
    model.eval()

    active = use or names[0]
    if active not in names:
        raise SystemExit(f"--use {active!r} is not one of {names}")
    model.set_adapter(active)
    print(f"active adapter: {active}")
    return model, tok, active


def render(tok, system, turns, thinking, add_generation_prompt=True):
    messages = ([{"role": "system", "content": system}] if system else []) + turns
    return qwenkit.template_ids(
        tok, messages, add_generation_prompt=add_generation_prompt, thinking=thinking)


def trim(tok, system, turns, thinking, max_ctx):
    """Drop the oldest turns until the prompt fits. Returns (turns, dropped).

    Never drops the last turn. A prompt consisting only of a system message is
    not renderable by Qwen's template ("No user query found"), and the user's
    current question is the one turn worth keeping even if it is over budget --
    generate() then refuses it with a message the user can act on.
    """
    dropped = 0
    while len(turns) > 1 and len(render(tok, system, turns, thinking)) > max_ctx:
        # Drop a whole user/assistant pair where possible, so history never
        # starts with an assistant turn the model has no question for.
        cut = 2 if len(turns) >= 2 and turns[0]["role"] == "user" else 1
        turns = turns[cut:]
        dropped += cut
    return turns, dropped


def generate(model, tok, system, turns, args, thinking, stop_event, stream):
    """Generate one reply. Returns (text, n_prompt, seconds)."""
    import torch
    from transformers import TextIteratorStreamer

    ids = render(tok, system, turns, thinking)
    if len(ids) > args.max_context:
        raise SystemExit(f"prompt is {len(ids)} tokens, above --max-context "
                         f"{args.max_context}; /reset or lower --max-new-tokens")
    input_ids = torch.as_tensor([ids], dtype=torch.long, device=model.device)

    kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        stopping_criteria=[StopOnEvent(stop_event)],
    )
    if args.temperature > 0:
        kwargs.update(temperature=args.temperature, top_p=args.top_p,
                      top_k=args.top_k, repetition_penalty=args.repetition_penalty)

    t0 = time.time()
    if not stream:
        with torch.no_grad():
            out = model.generate(**kwargs)
        text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip(), len(ids), time.time() - t0

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True,
                                    timeout=180.0)
    kwargs["streamer"] = streamer
    box = {}

    def run():
        with torch.no_grad():
            box["out"] = model.generate(**kwargs)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    chunks = []
    try:
        for piece in streamer:
            chunks.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[stopped]")
    thread.join(timeout=10.0)
    text = "".join(chunks)
    return text.strip(), len(ids), time.time() - t0


def parse_thinking(value):
    return {"off": False, "on": True, "auto": None}[value]


def print_stats(args, n_prompt, n_new_est, seconds):
    print(f"\n[{n_prompt} prompt tokens | ~{n_new_est} new | {seconds:.1f}s"
          f"{f' | {n_new_est/seconds:.1f} tok/s' if seconds > 0 else ''}"
          f" | temp {args.temperature} | thinking {args.thinking}]")


def repl(model, tok, args, active):
    system = args.system
    thinking = parse_thinking(args.thinking)
    turns = []
    print("\nType /help for commands, /quit to leave. Enter sends.")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            rest = rest.strip()
            if cmd in ("/quit", "/exit"):
                return
            if cmd == "/help":
                print("  /reset              clear the conversation\n"
                      "  /system <text>      replace the system prompt (empty clears)\n"
                      "  /temp <float>       0 = greedy, 1.0 = default-ish sampling\n"
                      "  /max <int>          max new tokens per reply\n"
                      "  /thinking off|on|auto\n"
                      "  /adapter <name>     switch adapter loaded at startup\n"
                      "  /adapters           list loaded adapters\n"
                      "  /save [path]        write this session as JSON\n"
                      "  /quit")
                continue
            if cmd == "/reset":
                turns = []
                print("  conversation cleared")
                continue
            if cmd == "/system":
                system = rest
                print(f"  system prompt: {system!r}" if system else "  system prompt cleared")
                continue
            if cmd == "/temp":
                args.temperature = float(rest)
                print(f"  temperature {args.temperature}")
                continue
            if cmd == "/max":
                args.max_new_tokens = int(rest)
                print(f"  max new tokens {args.max_new_tokens}")
                continue
            if cmd == "/thinking":
                if rest not in ("off", "on", "auto"):
                    print("  use off, on or auto")
                    continue
                args.thinking = rest
                thinking = parse_thinking(rest)
                print(f"  thinking {rest}")
                continue
            if cmd == "/adapters":
                print("  " + (", ".join(getattr(model, "peft_config", {}) or {}) or "base only")
                      + (f" (active: {active})" if active else ""))
                continue
            if cmd == "/adapter":
                if not rest or rest not in (getattr(model, "peft_config", {}) or {}):
                    print("  unknown adapter; /adapters lists them")
                    continue
                model.set_adapter(rest)
                active = rest
                print(f"  active adapter {active}")
                continue
            if cmd == "/save":
                path = rest or os.path.join(
                    HERE, "checkpoints", time.strftime("chat-%Y%m%d-%H%M%S.json"))
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": args.model, "precision": args.precision,
                        "adapter": args.adapter, "active_adapter": active,
                        "system": system, "messages": turns,
                        "sampling": {"temperature": args.temperature,
                                     "top_p": args.top_p, "top_k": args.top_k,
                                     "repetition_penalty": args.repetition_penalty,
                                     "max_new_tokens": args.max_new_tokens,
                                     "thinking": args.thinking},
                    }, f, indent=2, ensure_ascii=False)
                print(f"  wrote {path}")
                continue
            print(f"  unknown command {cmd}; /help")
            continue

        turns.append({"role": "user", "content": line})
        turns, dropped = trim(tok, system, turns, thinking, args.max_context)
        if dropped:
            print(f"  [{dropped} old message(s) dropped to fit context]")
        sys.stdout.write("\nhanny> ")
        sys.stdout.flush()
        stop_event = threading.Event()
        reply, n_prompt, seconds = generate(
            model, tok, system, turns, args, thinking, stop_event, stream=not args.no_stream)
        reply, cut = qwenkit.strip_think(reply)
        if cut:
            print(f"\n[stripped {cut} chars of reasoning trace]")
        if reply:
            turns.append({"role": "assistant", "content": reply})
        print_stats(args, n_prompt, len(tok(reply, add_special_tokens=False)["input_ids"]),
                    seconds)


def main():
    p = argparse.ArgumentParser(
        description="Chat with the base model and/or LoRA adapters.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=qwenkit.DEFAULT_MODEL)
    p.add_argument("--precision", default="4-bit nf4", choices=list(qwenkit.PRECISIONS),
                   help="4-bit nf4 matches finetune.py/chateval.py. fp16 is a "
                        "different measurement -- never compare across the two.")
    p.add_argument("--adapter", action="append", default=[], metavar="[NAME=]PATH",
                   help="repeatable; multiple adapters load as named adapters")
    p.add_argument("--use", help="adapter name to activate (default: first)")
    p.add_argument("--system", default=qwenkit.DEFAULT_SYSTEM,
                   help="system prompt; pass '' for none")
    p.add_argument("--thinking", default="off", choices=["off", "on", "auto"],
                   help="Qwen3.5 reasoner switch. 'on' spends the token budget on "
                        "a reasoning trace before answering; SFT in this repo is "
                        "trained with it off.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0 = greedy/deterministic")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-context", type=int, default=4096,
                   help="prompt budget; oldest turns are dropped beyond it")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--ask", help="one-shot prompt, then exit (no REPL)")
    p.add_argument("--json", action="store_true", help="with --ask, print JSON")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    model, tok, active = build(args.model, args.precision, args.adapter, args.use)
    print(f"model {args.model} at {args.precision} | template "
          f"{qwenkit.chat_template_sha16(tok)}")

    thinking = parse_thinking(args.thinking)

    if args.ask:
        turns = [{"role": "user", "content": args.ask}]
        stop_event = threading.Event()
        reply, n_prompt, seconds = generate(
            model, tok, args.system, turns, args, thinking, stop_event,
            stream=not args.no_stream and not args.json)
        if not args.json and not args.no_stream:
            print()
        reply, cut = qwenkit.strip_think(reply)
        if not args.json:
            if args.no_stream:
                # --no-stream collects the reply instead of printing it during
                # generation. Without this the answer is generated and silently
                # dropped: a Kaggle `--ask --no-stream` cell showed the banner
                # and the kernel warnings, then nothing.
                print(reply)
            if cut:
                print(f"[stripped {cut} chars of reasoning trace]")
        if args.json:
            print(json.dumps({
                "model": args.model, "precision": args.precision, "adapter": args.adapter,
                "prompt": args.ask, "reply": reply,
                "prompt_tokens": n_prompt,
                "reply_tokens": len(tok(reply, add_special_tokens=False)["input_ids"]),
                "seconds": round(seconds, 3),
                "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                             "top_k": args.top_k,
                             "repetition_penalty": args.repetition_penalty,
                             "thinking": args.thinking},
            }, indent=2, ensure_ascii=False))
        return 0

    repl(model, tok, args, active)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
