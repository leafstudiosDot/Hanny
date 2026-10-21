"""Shared Qwen loading, templating and provenance helpers. Safe to import.

Why this module exists in a repo that otherwise has no package layout: four
scripts now load the same model at the same precision, and a fifth will load a
4B. `lorewrite.py` already demonstrated how that goes wrong -- it passed
`bnb_use_double_quant`, a kwarg `BitsAndBytesConfig` does not have, so the flag
was silently dropped and that tool ran a different quantiser than `finetune.py`
and `chateval.py` while looking identical. One loader, one set of flags.

Deliberately NOT here: any scoring. `evalkit.py` owns every measurement, and a
second implementation of window selection or chat masking is exactly how two
"comparable" numbers drift apart. This file moves bytes and renders prompts.
"""
import hashlib
import json
import os

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
PRECISIONS = ("4-bit nf4", "fp16")

# Used by chat.py as the default system prompt and by sft.py to fill in examples
# that have none. Sharing the string is the point: a model trained without a
# system turn and then prompted with one is being asked a distribution it never
# saw, and that difference is invisible unless both scripts agree by construction.
DEFAULT_SYSTEM = ("You are Hanny, a helpful AI assistant. Answer in clear Markdown. "
                  "If you are not sure about something, say so instead of guessing.")


def sha256_16_file(path):
    """First 16 hex chars of a file's sha256. Identity for weights and datasets."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sha256_16_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def adapter_identity(adapter_dir):
    """Identify an adapter by its weights and report what produced it.

    A path is a location, not an identity: `finetune.py` is not bit-reproducible,
    so re-running a run overwrites `adapter-best/` with different weights that
    still call themselves by the same name. Same rule as chateval.py, and the
    same reason AGENTS.md forbids comparing numbers across corpora.
    """
    adapter_dir = os.path.abspath(adapter_dir)
    weights = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(weights):
        return {"error": f"no adapter_model.safetensors in {adapter_dir}"}
    info = {
        "path": adapter_dir,
        "sha256_16": sha256_16_file(weights),
        "bytes": os.path.getsize(weights),
    }
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            info["base_model"] = cfg.get("base_model_name_or_path")
            info["lora_r"] = cfg.get("r")
            info["lora_alpha"] = cfg.get("lora_alpha")
            info["target_modules"] = cfg.get("target_modules")
        except (OSError, ValueError) as e:
            info["adapter_config"] = f"unreadable: {e}"
    run_result = os.path.join(os.path.dirname(adapter_dir.rstrip("/\\")), "result.json")
    if os.path.exists(run_result):
        try:
            with open(run_result, encoding="utf-8-sig") as f:
                r = json.load(f)
            info["produced_by"] = {
                "run": r.get("run"),
                "loss_after": r.get("loss_after"),
                "corpus_fingerprint": r.get("corpus_fingerprint"),
                "chat_template_sha16": r.get("chat_template_sha16"),
                "sources": r.get("sources"),
                "kernels": r.get("kernels"),
            }
        except (OSError, ValueError) as e:
            info["produced_by"] = f"unreadable: {e}"
    else:
        info["produced_by"] = None
    return info


def configure_tokenizer(tok):
    """Qwen ships no pad token in some revisions and one in others; either way
    padding is only safe because every caller masks padded positions out of the
    labels. Set it once here rather than in four call sites."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def cuda_index(device):
    """Map ``cuda`` / ``cuda:0`` / ``cuda:1`` to a device index.

    Training with DDP gives each process its own GPU, so a hardcoded 0 would
    put both ranks on GPU 0. This is deliberately strict: an unrecognised
    device string is a programming error, not something to silently coerce.
    """
    if device == "cuda":
        return 0
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            index = int(device.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"invalid CUDA device {device!r}") from None
        if index < 0:
            raise ValueError(f"invalid CUDA device {device!r}")
        return index
    raise ValueError(f"device must be 'cuda' or 'cuda:N', got {device!r}")


def load(model_name=DEFAULT_MODEL, precision="4-bit nf4", device=None, tok=None):
    """Load a causal LM + tokenizer at a stated precision. Returns (model, tok).

    `tok` lets a caller that already loaded a tokenizer (sft.py does, so it can
    build the dataset before deciding to train at all) reuse it instead of
    constructing a second object with the same template.

    `device` is 'cpu' or a CUDA string such as 'cuda:0' / 'cuda:1'. The latter
    is what DDP needs: each rank loads its own copy on its own card.

    Precision is not a speed knob here. fp16 and 4-bit NF4 differ by ~0.086 nats
    on this model -- a third of the effect M4 measured -- so a number produced
    at one precision must never be compared with a number produced at the other.
    The flag is required to be explicit at every call site that records a score.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tok is None:
        tok = configure_tokenizer(AutoTokenizer.from_pretrained(model_name))

    if device == "cpu":
        # half-precision matmuls on CPU are slow and partly unimplemented, so a
        # CPU load is float32 whatever the flag says. Print it: a silent
        # precision change is the class of bug this module exists to prevent.
        if precision != "fp16":
            print("note: no CUDA -- loading float32 on CPU; 4-bit needs a GPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, device_map={"": "cpu"})
    else:
        index = cuda_index(device)
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"device {device!r} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible")
        if precision == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16, device_map={"": index})
        else:
            from transformers import BitsAndBytesConfig
            # Identical flags to finetune.py, chateval.py and lorewrite.py. If
            # this block changes, every recorded loss changes with it.
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, dtype=torch.float16,
                device_map={"": index})
    model.eval()
    return model, tok


def template_ids(tok, messages, add_generation_prompt=False, thinking=None):
    """apply_chat_template -> a flat list of token ids, or raise.

    `thinking=None` leaves the template's own default alone; False/True pass
    Qwen's `enable_thinking` switch. Qwen3.5's template, measured on the real
    tokenizer at revision in use:

      thinking=False  -> prefix ends '...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
                         and an assistant turn is a PREFIX-EXTENSION of it, so
                         assistant-only loss masking is exact.
      thinking=True   -> the prefix ends INSIDE the open think block and the
                         template closes it only when rendering a message, so
                         the prefix is NOT a prefix of the full turn. SFT with
                         thinking on therefore needs reasoning-trace data in a
                         format this repo does not have yet; sft.py refuses
                         rather than masking the wrong tokens.

    Shapes: transformers 5.x returns a BatchEncoding from
    apply_chat_template(tokenize=True). Older code in this repo assumed a list.
    Normalise here once; never index the return value directly.
    """
    kw = {} if thinking is None else {"enable_thinking": bool(thinking)}
    try:
        out = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kw)
    except TypeError:
        # A tokenizer whose template does not accept the switch. Ignoring it is
        # correct for a model with no reasoner; for Qwen it would mean the
        # caller's intent was silently dropped, so say so.
        if thinking is not None:
            print("note: this tokenizer's chat template rejects enable_thinking; "
                  "the template default is being used instead.")
        out = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt)

    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    while isinstance(out, list) and len(out) == 1 and isinstance(out[0], list):
        out = out[0]
    if not isinstance(out, list) or not all(isinstance(t, int) for t in out):
        raise TypeError(
            f"apply_chat_template returned {type(out).__name__}; refusing to "
            f"guess (same rule as evalkit._template_ids).")
    return out


def chat_template_sha16(tok):
    """Hash of the chat template itself.

    Training and inference must render the same turn structure. The template is
    a remote string that ships with the tokenizer revision, so it can change
    under a model name. Recording its hash in result.json is how a later reader
    knows whether an adapter was trained against the template they are using.
    """
    return sha256_16_text(getattr(tok, "chat_template", "") or "")


_THINK_CLOSE = "</think>"


def strip_think(text):
    """Cut a leaked reasoning block at </think>; return (text, chars_removed).

    Presentation only. lorewrite.cut_think does the same job for documents and
    is deliberately left alone: it also strips trailing special tokens, and its
    character count is recorded in generated-document metadata.
    """
    removed = 0
    if _THINK_CLOSE in text:
        head, tail = text.split(_THINK_CLOSE, 1)
        removed = len(head) + len(_THINK_CLOSE)
        text = tail.lstrip()
    return text, removed


def lora_targets(model):
    """Detect LoRA target modules by inheritance, never by class name.

    After 4-bit quantisation the projections are Linear4bit. bitsandbytes makes
    that a subclass of nn.Linear, so isinstance finds them; a class-name check
    would find nothing and report "architecture changed", which is why AGENTS.md
    states the rule. Returns (targets, missing).
    """
    import torch

    available = {n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)}
    wanted = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
    return [t for t in wanted if t in available], [t for t in wanted if t not in available]


def kernels():
    """Which fast kernels were importable for THIS process.

    Both are recorded per run because wall clock depends on them, and neither is
    the order-of-magnitude win its fallback warning implies: flash-linear-
    attention measured 4305s -> 3960s on a 555-step T4 run (1.09x) and
    causal-conv1d ~0.4%. See README "Hardware limits".
    """
    out = {"flash_linear_attention": None, "causal_conv1d": None}
    try:
        import fla  # noqa: F401
        out["flash_linear_attention"] = getattr(fla, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import causal_conv1d  # noqa: F401
        out["causal_conv1d"] = getattr(causal_conv1d, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - a half-built extension raises OSError
        out["causal_conv1d"] = False
    return out
