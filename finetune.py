r"""
M4 - LoRA fine-tune Qwen3.5-0.8B on this codebase's own source.

What fine-tuning actually does
------------------------------
The base model already writes code. It scored perplexity 4.53 on this
codebase before we touched a single weight. Fine-tuning does not teach it to
code -- it teaches it *your* idioms: your naming, your comment style, which
library you reach for first. That is a small thing to learn and a real thing
to learn, and the only honest way to show it happened is a before/after
number on identical evaluation windows.

How the comparison stays honest
-------------------------------
1. The evaluation comes from evalkit.py, the same module baseline.py uses, so
   the windows are provably identical.
2. Before training starts, this script scores the BASE model on those windows
   and prints it. If that number does not match baseline.json, something is
   wrong and you should stop -- do not trust a comparison against a harness
   that has drifted.
3. Every result file records the corpus fingerprint, so a number can never be
   compared across two different corpora by accident.

Hardware notes
--------------
Qwen3.5 is a hybrid of linear-attention (gated delta rule) and full-attention
layers. Turing GPUs -- both a GTX 1650 and a Kaggle T4, compute capability 7.5
-- have NO bf16, so this is fp16 on both. Recipes copied from an H100 notebook
will not run here.

Run locally:
    .venv\Scripts\python.exe finetune.py --run m4-test --max-steps 20 --batch 1
Run for real (Kaggle T4):
    .venv\Scripts\python.exe finetune.py --run m4 --epochs 3 --batch 8
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import data
import evalkit
import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser(description="M4 LoRA fine-tune", formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
p.add_argument("--run", default=time.strftime("m4-%Y%m%d-%H%M"))
p.add_argument("--corpus-cache", default=os.path.join(HERE, "cache"),
               help="folder holding corpus-*.json.gz")
p.add_argument("--rebuild-corpus", action="store_true",
               help="walk data_dirs.txt locally instead of loading the cache "
                    "(only possible on the machine that owns those folders)")
p.add_argument("--seq-len", type=int, default=512)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--eval-windows", type=int, default=64,
               help="must match the value baseline.py used, or the numbers are not comparable")
p.add_argument("--eval-batch", type=int, default=1)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--max-steps", type=int, default=0, help="stop early; 0 = run all epochs")
p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--lora-r", type=int, default=16)
p.add_argument("--lora-alpha", type=int, default=32)
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise SystemExit("M4 needs a GPU. Use Kaggle.")
# The recipe trains fp16 master weights with plain AdamW -- no autocast, no
# GradScaler. That is deliberate and verified for THIS model (loss_before was
# bit-identical across re-runs and the M4 drop was 4.96x its own standard
# error), not a property of the hardware: small fp16 gradients can underflow,
# so if the model or lr is replaced, re-verify rather than assume it carries
# over. A GradScaler is the conventional belt-and-braces if this ever proves
# unstable.
if torch.cuda.get_device_capability(0)[0] < 8:
    print(f"note: compute capability {torch.cuda.get_device_capability(0)} is Turing -- "
          f"no bf16, using fp16. This is correct for both GTX 1650 and T4.")
    print("      fp16 master weights + plain AdamW, no GradScaler -- tested stable "
          "on this model; re-verify if model or lr changes.")

OUT = os.path.join(HERE, "checkpoints", args.run)
os.makedirs(OUT, exist_ok=True)

# 1. Corpus
if args.rebuild_corpus:
    include, exclude = data.load_or_explain(os.path.join(HERE, "data_dirs.txt"))
    train_text, val_text, report = data.get_corpus(
        include, exclude, max_chars=30_000_000, cache_dir=args.corpus_cache)
    cache_path = data.find_cached(args.corpus_cache)
else:
    cache_path = data.find_cached(args.corpus_cache)
    # This is the step that makes cloud training possible at all: the local
    # folders in data_dirs.txt do not exist on Kaggle, so get_corpus() would
    # compute a different cache key and silently return nothing.
    train_text, val_text, report, fingerprint = data.load_cached(cache_path)

fingerprint = data.fingerprint_of(cache_path)
print(f"corpus: {len(train_text):,} train chars | {len(val_text):,} val chars")
print(f"fingerprint: {fingerprint}   ({os.path.basename(cache_path)})")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model)
tok.model_max_length = int(1e12)

train_ids = np.asarray(tok(train_text, add_special_tokens=False)["input_ids"], dtype=np.int64)
val_ids = np.asarray(tok(val_text, add_special_tokens=False)["input_ids"], dtype=np.int64)
print(f"tokenized: {len(train_ids):,} train | {len(val_ids):,} val "
      f"({len(train_text)/len(train_ids):.2f} chars per token)")

# 2. Fixed evaluation windows (shared with baseline.py)
eval_windows = evalkit.fixed_windows(val_ids, args.eval_windows, args.seq_len)
cov, ov, _ = evalkit.window_geometry(len(val_ids), args.eval_windows, args.seq_len)
print(f"eval: {len(eval_windows)} windows x {args.seq_len} tokens, "
      f"overlap {ov*100:.0f}%, covering {cov*100:.0f}% of the val split")

# 3. Base model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import BitsAndBytesConfig  # noqa: E402

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("loading base model in 4-bit...")
base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=torch.float16)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
base.gradient_checkpointing_enable()
base.enable_input_require_grads()
base.config.use_cache = False
base = base.to(device)

# 4. Verify the harness BEFORE changing anything
# If this does not match baseline.json, any later comparison is meaningless.
pre, pre_se = evalkit.score(base, eval_windows, args.eval_batch, device)
print("\n" + "=" * 68)
print(f"BEFORE  val loss {pre:.4f} +- {pre_se:.4f}   (perplexity {math.exp(pre):.3f})")
baseline_file = os.path.join(HERE, "baseline.json")
baseline_full_precision = None
if os.path.exists(baseline_file):
    with open(baseline_file, encoding="utf-8") as f:
        b = json.load(f)
    print(f"\nbaseline.json records {b['loss']:.4f} on {b.get('model')} "
          f"at {b.get('dtype')} precision")

    # Hard checks: these make any comparison meaningless, so stop.
    fatal = []
    if b.get("corpus_fingerprint") not in (None, fingerprint):
        fatal.append(f"corpus differs (baseline {b.get('corpus_fingerprint')}, "
                     f"this run {fingerprint})")
    if b.get("windows") != len(eval_windows) or b.get("seq_len") != args.seq_len:
        fatal.append(f"eval windows differ (baseline {b.get('windows')}x{b.get('seq_len')}, "
                     f"this run {len(eval_windows)}x{args.seq_len})")
    if b.get("model") != args.model:
        fatal.append(f"model differs (baseline {b.get('model')}, this run {args.model})")

    # Informational: 4-bit NF4 costs real accuracy. Measured ~0.086 nats here.
    # That gap is the price of quantisation, NOT a harness bug, and it is why
    # the honest comparison is BEFORE-vs-AFTER *within this run* rather than
    # against the full-precision baseline.json number.
    drift = pre - b["loss"]
    print(f"drift {drift:+.4f} nats -- expected for a 4-bit base model "
          f"(full precision measured {b['loss']:.4f})")
    print("compare AFTER against BEFORE below, not against baseline.json")

    if fatal:
        print("\n*** HARNESS CHECK FAILED -- do not trust this run ***")
        for f_ in fatal:
            print(f"  - {f_}")
        raise SystemExit("Fix the above before training.")
    baseline_full_precision = b["loss"]
    print("provenance check passed: same corpus, same windows, same model")
else:
    print("no baseline.json found; this run establishes a new one")
print("=" * 68)

# 5. LoRA
# After 4-bit quantization the projections are Linear4bit, not nn.Linear, so
# target by name and verify against what the module tree actually contains.
linear_names = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)]
available = sorted({n.split(".")[-1] for n in linear_names})
wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
targets = [t for t in wanted if t in available]
missing = [t for t in wanted if t not in available]
print(f"\nLoRA targets: {targets}")
if missing:
    print(f"  (not present in this architecture: {missing})")
if not targets:
    raise SystemExit("found no LoRA target modules -- architecture changed?")

base = get_peft_model(base, LoraConfig(
    r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM", target_modules=targets))
base.print_trainable_parameters()
print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 6. Training windows
n_train = len(train_ids) - args.seq_len - 1
if n_train < 1:
    raise SystemExit("training corpus too small for one window")
n_batches = n_train // (args.seq_len * args.batch)
total_epochs = args.epochs if args.max_steps == 0 else args.max_steps / max(1, n_batches)
steps_per_epoch = max(1, n_batches)
total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
print(f"\n{len(train_ids):,} tokens -> {n_batches} batches/epoch of "
      f"{args.batch}x{args.seq_len} | {total_steps:.0f} total steps")

rng = np.random.default_rng(args.seed)
params = [q for q in base.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=args.lr, total_steps=max(1, int(total_steps)),
    pct_start=0.1, anneal_strategy="cos", cycle_momentum=False, div_factor=10, final_div_factor=10)

# Which fast kernels were actually live when this measurement was taken. Both are
# recorded in result.json, because wall clock depends on them and a result that
# does not say which kernels it ran on cannot be compared with another result on
# speed. Neither is "several times faster" -- see the measured numbers below.
#
# Qwen3.5 is hybrid: gated-delta-rule (linear attention) layers sit beside
# full-attention layers. With flash-linear-attention present they take a fused
# Triton path; without it transformers falls back to a reference implementation.
# A fresh `pip install flash-linear-attention` usually is NOT importable until
# the Kaggle kernel restarts, so this check is worth printing rather than
# assuming -- the symptom is "it just runs slowly" with no error anywhere.
kernels = {"flash_linear_attention": None, "causal_conv1d": None}
try:
    import fla  # noqa: F401
    kernels["flash_linear_attention"] = getattr(fla, "__version__", "unknown")
    print(f"flash-linear-attention {kernels['flash_linear_attention']}: "
          f"gated-delta-rule layers take the fast path")
except ImportError:
    print("WARNING: flash-linear-attention is not importable. The gated-delta-rule "
          "layers are on the reference implementation. Measured on a Kaggle T4, 555 "
          "steps at batch 8x512: 4305s without, 3960s with -- 1.09x, not the order of "
          "magnitude the fallback path suggests. Fix: pip install "
          "flash-linear-attention, then RESTART the Kaggle kernel and re-run.")

# causal_conv1d is the other kernel transformers may fall back on, and this one
# is a CUDA extension that builds against nvcc -- on Kaggle's current image the
# build fails outright, so "not installed" is the normal state of a Cell 1 run.
# Exception rather than ImportError because a half-built extension raises OSError
# on import. Measured, because the claim it used to carry was wrong: a run with
# fla and WITHOUT causal_conv1d finished in 3977s (7.17 s/step) against 3960s
# (7.14 s/step) for the reference run. It is not the bottleneck; ~7 s/step is
# simply what this model costs on a T4.
try:
    import causal_conv1d  # noqa: F401
    kernels["causal_conv1d"] = getattr(causal_conv1d, "__version__", "unknown")
except Exception:  # noqa: BLE001 - presence probe, any failure means "not there"
    kernels["causal_conv1d"] = False
print(f"causal_conv1d: {kernels['causal_conv1d'] or 'ABSENT -- transformers uses its reference conv path'}"
      " (not the bottleneck; see README)")

curve = os.path.join(OUT, "curve.csv")
with open(curve, "w", encoding="utf-8") as f:
    f.write("step,train_loss,val_loss,val_se,lr,seconds\n")

print("\n" + "=" * 68)
print(f"{'step':>6} {'train':>8} {'val':>8} {'se':>7} {'lr':>9} {'time':>7}")
print("=" * 68)

base.train()
t0 = time.time()
running = []
best_val, best_step = float("inf"), 0
done = 0

for step in range(1, int(total_steps) + 1):
    ix = rng.integers(0, n_train, size=args.batch)
    xb = torch.from_numpy(
        np.stack([train_ids[i:i + args.seq_len] for i in ix])).to(device)

    loss = base(xb, labels=xb).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    running.append(float(loss.detach()))
    done = step

    if step % args.eval_every == 0 or step == total_steps:
        va, vse = evalkit.score(base, eval_windows, args.eval_batch, device)
        el = time.time() - t0
        tr = float(np.mean(running[-50:]))
        star = ""
        if va < best_val:
            best_val, best_step = va, step
            base.save_pretrained(os.path.join(OUT, "adapter-best"))
            star = "  <- best"
        print(f"{step:>6} {tr:>8.4f} {va:>8.4f} {vse:>7.4f} "
              f"{sched.get_last_lr()[0]:>9.2e} {el:>6.0f}s{star}")
        with open(curve, "a", encoding="utf-8") as f:
            f.write(f"{step},{tr:.4f},{va:.4f},{vse:.4f},"
                    f"{sched.get_last_lr()[0]:.3e},{el:.1f}\n")
        base.train()

if best_step == 0:  # eval never ran, e.g. --max-steps smaller than --eval-every
    va, vse = evalkit.score(base, eval_windows, args.eval_batch, device)
    best_val, best_step = va, done
    base.save_pretrained(os.path.join(OUT, "adapter-best"))
    print(f"final eval at step {done}: val {va:.4f} +- {vse:.4f} (saved)")

# 7. Result
drop = pre - best_val
print("\n" + "=" * 68)
print(f"BEFORE  {pre:.4f} +- {pre_se:.4f}   (4-bit base, never trained)")
print(f"AFTER   {best_val:.4f}          (best at step {best_step} of {done})")
print(f"CHANGE  {drop:+.4f} nats   perplexity {math.exp(pre):.3f} -> {math.exp(best_val):.3f}")
print("=" * 68)
print("\nThis BEFORE-vs-AFTER is the valid comparison. Both numbers come from\n"
      "the same 4-bit model on the same fixed windows, so quantisation cancels\n"
      "out. Do not compare AFTER to the full-precision baseline.json number --\n"
      "that one is ~0.09 lower purely because it is not quantised.\n")
print("A drop smaller than about 2x the standard error is not evidence of\n"
      "anything. If the change is small, the honest conclusion is that 1M\n"
      "tokens of one codebase is not enough to move this model much -- which\n"
      "is itself a useful result.")

result = {
    "model": args.model,
    "run": args.run,
    "corpus_fingerprint": fingerprint,
    "windows": len(eval_windows),
    "seq_len": args.seq_len,
    "eval_coverage": round(cov, 3),
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "lr": args.lr,
    "batch": args.batch,
    "epochs": args.epochs,
    "steps_done": done,
    "best_step": best_step,
    "loss_before": round(pre, 6),
    "loss_before_se": round(pre_se, 6),
    "loss_before_precision": "4-bit nf4",
    # Which fast kernels were importable for THIS run. "seconds" below is only
    # comparable against a run whose values here match.
    "kernels": kernels,
    # Recorded for context only, never as the comparison. fp16-vs-4bit differ by
    # ~0.09 nats, which is comparable to the effect being measured here.
    "baseline_json_loss_full_precision": baseline_full_precision,
    "loss_after": round(best_val, 6),
    "loss_change": round(drop, 6),
    "perplexity_before": round(math.exp(pre), 4),
    "perplexity_after": round(math.exp(best_val), 4),
    "seconds": round(time.time() - t0, 1),
}
with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nwrote {OUT}/result.json and adapter-best/")
print("The adapter is the fine-tune. To use it, load the base model and apply it:")
print(f"    from peft import PeftModel")
print(f"    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('{args.model}'),")
print( '                                    r"checkpoints/' + args.run + r'/adapter-best")')