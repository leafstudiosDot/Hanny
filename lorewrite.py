"""Retrieve, prompt, generate one Markdown document, then score it.

This is the whole loop in one command: it is the first thing in the repository
that produces a document, and it deliberately exists so the document can be
scored rather than admired.

    .venv\\Scripts\\python.exe lorewrite.py --ask "why are outlaws cursed?"
    .venv\\Scripts\\python.exe lorewrite.py --ask "..." --model Qwen/Qwen3.5-0.8B

The point of running this on the 0.8B model FIRST, before any ceiling test on
something bigger, is that it proves the pipeline end to end -- retrieval, prompt
assembly, generation, scoring -- with a model already on this machine. When the
bigger model arrives later, every command is identical except --model, so the
comparison is one variable.

Whatever comes out is written to lore/generated/ WITH the prompt and the
retrieved chunks beside it, because a generated document without its inputs is
not evidence of anything and cannot be reproduced.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lorekit  # noqa: E402

OUT_DIR = os.path.join(HERE, "lore", "generated")
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"

# One instruction, stated once. Everything else about faithfulness comes from
# loreeval.py, not from begging the model in the prompt -- a prompt is not a
# guarantee, and the gate has to be able to fail.
INSTRUCTION = """You are writing a reference document from the source notes below.

CITATIONS -- the single most important rule.
Every factual sentence ends with a citation copied character-for-character from
the SOURCE FILE and HEADING of the note it came from, in this exact form:

    [[corpus/path/to/file.md#the-heading-slug]]

Use the full path and the full heading every time. Never write [1], [2], (see
above), or any other shorthand. A citation that is not in that exact form cannot
be checked and counts as no citation at all.

OTHER RULES
- Use ONLY the source notes. Do not add facts, names, dates or numbers that are not in them.
- If a note is ambiguous, say it is ambiguous rather than picking one reading.
- If the notes do not cover something, say so plainly under a
  "Not covered by the sources" heading. Do not fill the gap.
- Quote the source's own wording for anything surprising; do not paraphrase a
  rule into its opposite.
- Plain Markdown. No preamble, no "here is your document".

SOURCES
{sources}

QUESTION
{question}
"""


def build_sources(chunks, limit):
    """Render retrieved chunks with their citation target visible.

    Deliberately NOT numbered [1], [2], ... The first generated document cited
    "[1]" and "[2]" -- it had picked up the marker style from this function
    rather than the citation format the instruction asked for. Offering the
    model two conventions and hoping it picks the checkable one is asking for
    exactly the failure the gate exists to catch. One convention, only.
    """
    parts = []
    for c in chunks:
        parts.append(f"SOURCE FILE {c['rel']}\nHEADING {c['heading']}\n"
                     f"TEXT\n{c['text']}")
    return "\n\n".join(parts)


def load_model(model_name, precision):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, dtype=torch.float16)
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ask", required=True, help="the question to write about")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"default {DEFAULT_MODEL}. Change this and nothing else "
                        f"to compare models on the same question.")
    p.add_argument("--precision", default="4-bit nf4", choices=["4-bit nf4", "fp16"])
    p.add_argument("--k", type=int, default=6, help="chunks to retrieve")
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 = pure greedy, which is what evalkit.generate_greedy "
                        "does and is right for measurement. Raise it to break the "
                        "repetition loop greedy decoding falls into on long-form "
                        "output. Running one question at 1.0 and 1.1 separates "
                        "'the model cannot do this' from 'greedy decoding is the "
                        "wrong tool here' -- do that before concluding anything "
                        "about a model's ceiling.")
    p.add_argument("--no-score", action="store_true",
                   help="skip the loreeval pass (useful when it is failing)")
    p.add_argument("--out-name", help="basename for the saved files")
    args = p.parse_args()

    chunks = lorekit.load_chunks()
    index = lorekit.BM25Index(chunks)
    hits = index.search(args.ask, k=args.k)
    if not hits:
        print(f"nothing retrieved for {args.ask!r} -- every query term is absent "
              f"from the corpus. Try different words.", file=sys.stderr)
        return 2

    print(f"model      {args.model} at {args.precision}")
    print(f"corpus     {len(chunks)} claims, retrieved {len(hits)}\n")
    for i, (c, score) in enumerate(hits, 1):
        print(f"  {score:5.2f}  {c['rel']}#{c['anchor']}")

    prompt = INSTRUCTION.format(sources=build_sources([c for c, _ in hits], args.k),
                                question=args.ask)

    import torch
    import evalkit  # generate_greedy lives with the other chat measurements

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, args.precision)
    print(f"\ngenerating on {device} "
          f"(repetition_penalty={args.repetition_penalty}) ...", flush=True)
    t0 = time.time()
    if args.repetition_penalty == 1.0:
        doc = evalkit.generate_greedy(model, tok, prompt, args.max_new_tokens)
    else:
        # Deliberately not evalkit.generate_greedy: that one is greedy by design
        # and must stay so, because it is what the safety measurement depends
        # on. Sampling here is a different question -- "what does this model
        # write when it is allowed to not repeat itself" -- so it gets its own
        # three lines rather than a flag on the measurement path.
        import torch
        from evalkit import _template_ids
        ids = torch.as_tensor(
            [_template_ids(tok, [{"role": "user", "content": prompt}],
                           add_generation_prompt=True)],
            dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        doc = tok.decode(out[0, ids.shape[1]:],
                         skip_special_tokens=True).strip()
    secs = time.time() - t0
    chars = len(doc)
    print(f"generated {chars} chars in {secs:.0f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = args.out_name or hashlib.sha256(
        f"{args.model}|{args.ask}".encode()).hexdigest()[:8]
    doc_path = os.path.join(OUT_DIR, f"{stem}.md")
    meta_path = os.path.join(OUT_DIR, f"{stem}.json")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc.rstrip() + "\n")
    meta = {
        "model": args.model, "precision": args.precision,
        "question": args.ask, "k": args.k,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "decoding": ("greedy" if args.repetition_penalty == 1.0
                     else f"sampled temp=0.7 top_p=0.9 rp={args.repetition_penalty}"),
        "corpus_chunks": len(chunks),
        "retrieved": [{"rel": c["rel"], "anchor": c["anchor"],
                       "score": round(float(s), 4)} for c, s in hits],
        "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "prompt_chars": len(prompt),
        "generated_chars": chars, "seconds": round(secs, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {os.path.relpath(doc_path, HERE)}")
    print(f"wrote {os.path.relpath(meta_path, HERE)}")
    print("\n" + "=" * 72)
    print("DOCUMENT")
    print("=" * 72)
    print(doc)

    if args.no_score:
        return 0

    print("\n" + "=" * 72)
    print("SCORING")
    print("=" * 72)
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "loreeval.py"), "--doc", doc_path],
        cwd=HERE)
    # loreeval exits 1 when the gate fails, which is the expected outcome more
    # often than not on a 0.8B model. Report it as a number rather than
    # pretending the command failed for a technical reason.
    print(f"\nloreeval exit code {r.returncode} "
          f"({'PASS' if r.returncode == 0 else 'FAIL -- read the reasons above'})")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
