"""Does a generated document hold up against the corpus that produced it?

The measurement the stated deliverable turns on. A loss number cannot see a
fluent, confidently wrong sentence, and "it wrote a nice document" is not a
result. This scores a document three ways, all of them string-level and all of
them falsifiable without a judge model:

  citations  every [[corpus/file.md#anchor]] resolves to a file and a heading
             that actually exist
  names      every proper noun in the prose appears somewhere in the corpus
  numbers    every number, date and quantity appears in the corpus

It also scores the retriever itself, because a document can be perfectly
faithful and still useless if the retrieval handed the model the wrong chunks.

WHAT THIS DOES NOT DO, and the list matters as much as the checks:

  - It cannot tell a correct reading of a source from a fluent misreading of
    it. "The shelf broke in 1147" against a source that says 1147 passes, even
    if the source attributes it to a different cause.
  - The proper-noun check has false positives. Capitalisation is a heuristic,
    not a part-of-speech tagger, and a report that hid its false positives
    would be a report nobody could act on.
  - A document with no citations and no numbers passes trivially. This is a
    floor, not a measure of quality.

Run:  .venv\\Scripts\\python.exe loreeval.py
      .venv\\Scripts\\python.exe loreeval.py --retrieval      (instrument check)
      .venv\\Scripts\\python.exe loreeval.py --doc lore/generated/thing.md
"""
import argparse
import json
import os
import re
import sys

import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lorekit  # noqa: E402

# Numbers, dates, quantities. Kept separate from the name check because these
# are the claims a model most confidently invents: a plausible year, a stat
# that sounds right, a version number that never existed.
_NUM_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\b"          # 1, 1998, 1.5, 12,800
)
# A number inside a word ("Rance01", "eve2") is a name, not a quantity.
_NUM_CONTEXT_RE = re.compile(r"[A-Za-z0-9_]")


def load_document(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def prose_only(text):
    """Document body with headings and fenced code removed.

    Headings are excluded on purpose: the check is about claims the document
    makes in prose. Section headings are labels, and '## Overview' asserting a
    proper noun is not a citation problem.
    """
    out, in_fence, fence = [], False, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            if not in_fence:
                in_fence, fence = True, s
            elif s == fence:
                in_fence, fence = False, None
            continue
        if in_fence or s.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


# lorewrite.py labels each retrieved chunk with these three lines before its
# text. A model that paraphrases the harness back out as prose --
# "**Lore/Corpus Shared /Hero-Syndrome.md**" -- then produces proper nouns that
# are the prompt's own scaffolding rather than anything it invented. Scoring
# that as a hallucination trains you to ignore the output, which is how a real
# hallucination gets missed. Stripped before the noun check.
_SOURCE_HEADER_RE = re.compile(
    r"^\s*(?:\*\*)?(?:source\s+file|heading|text)(?:\*\*)?\s*:?\s*$", re.I)

# The model does not always echo the header cleanly. One generated document
# rewrote 'SOURCE FILE lore/corpus/shared/world-of-eve.md' as a bullet reading
# '**Lore/Corpus Shared /World of Eve.md**' -- which is neither a standalone
# header nor anything it invented. Any path-shaped span ending in .md is
# scaffolding, so it goes before the noun check regardless of how it got there.
_PATH_ECHO_RE = re.compile(r"[*_\s]*[\w./*\-]*[\w\-]+\.md[*_\s:]*", re.I)


def strip_source_headers(text):
    """Remove prompt scaffolding the model may have echoed back as prose."""
    lines = [l for l in text.splitlines() if not _SOURCE_HEADER_RE.match(l)]
    return _PATH_ECHO_RE.sub(" ", "\n".join(lines))


def proper_nouns(text):
    """Capitalised tokens that are not sentence-initial and not stopwords.

    Sentence-initial words are skipped because "The shelf broke" would otherwise
    report 'The' and 'Shelf' from the same line. This over-reports rather than
    under-reports on purpose: a missed invented name is the failure this whole
    project exists to catch, and a false positive costs a reader five seconds.

    The trailing-character rule matters. The first version used
    ``[A-Za-z][A-Za-z'\\-]*``, which captures the *hyphen and stops* at the next
    digit -- so the document's arrived here,
    reported as absent from a corpus that plainly contains it. Two of the three
    "not in corpus" flags on the first generated document were artifacts of that
    regex, not model errors. Requiring an alphanumeric on both ends fixes the
    truncation and leaves genuine hyphenated names intact.
    """
    found = {}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Split into sentences, then drop the first word of each: that is the
        # one capitalised by position rather than by being a name.
        for sentence in re.split(r"(?<=[.!?])\s+", s):
            words = re.findall(r"[A-Za-z][A-Za-z'’-]*[A-Za-z]", sentence)
            for i, w in enumerate(words):
                if i == 0:
                    continue
                if not w[0].isupper():
                    continue
                lw = w.lower()
                if lw in lorekit.STOPWORDS or len(w) < 3:
                    continue
                found[w] = found.get(w, 0) + 1
    return found


def numbers_in(text):
    found = {}
    for m in _NUM_RE.finditer(text):
        # Reject numbers glued to letters/digits: Rance01, eve2, 3d.
        start, end = m.span()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if _NUM_CONTEXT_RE.match(before or "") or _NUM_CONTEXT_RE.match(after or ""):
            continue
        found[m.group(0)] = found.get(m.group(0), 0) + 1
    return found


# A resolvable corpus citation, matched whole. Loose-reference detection has to
# skip these FIRST: the original pattern matched the inner [...] of a [[...]],
# so every citation that resolved was ALSO reported as an uncheckable
# "bracketed reference" -- the first Kaggle run to pass printed
#     citations   1/1 resolve
#     1 bracketed reference(s) that are not corpus citations: [...the one that resolved...]
# which is a report contradicting itself on the same line.
_CITATION_SPAN_RE = re.compile(r"\[\[corpus/[^\]]+\]\]")

# Any bracketed marker that is NOT a corpus citation -- [1], [2], (see above).
# Counted separately because "the model cited nothing" and "the model cited in
# a format that does not resolve" are different failures with different fixes,
# and reporting them as the same thing sends you looking in the wrong place.
_LOOSE_CITE_RE = re.compile(r"(?<!\!)\[([^\]\[]+)\]")


def loose_refs(text):
    """Bracketed references that are not corpus citations.

    Corpus citations are blanked before the scan, so a resolved citation is
    never double-counted as an uncheckable one.
    """
    body = _CITATION_SPAN_RE.sub("  ", prose_only(text))
    return [m.group(1).strip() for m in _LOOSE_CITE_RE.finditer(body)]


def check_citations(doc, corpus_root=None):
    """Every citation resolves to an existing file AND an existing heading.

    Also counts loose [bracketed] references. The first generated document came
    back citing "[1]" and "[2]" -- it had cited, just not in a form that
    resolves -- and this function reported "no citations at all", which is
    false. It sent the diagnosis somewhere else entirely.
    """
    corpus_root = corpus_root or lorekit.CORPUS_DIR
    base = os.path.join(HERE, "lore")  # citations are written corpus/...
    refs = lorekit.citations_in(doc)
    loose = loose_refs(doc)
    rows, resolved, broken = [], 0, []
    for raw, rel, anchor in refs:
        path = os.path.join(base, rel.replace("/", os.sep))
        exists = os.path.isfile(path)
        headings = lorekit.headings_in_file(path) if exists else None
        anchor_ok = bool(headings) and anchor in headings
        ok = exists and anchor_ok
        resolved += ok
        if not ok:
            if not exists:
                why = "file does not exist"
            else:
                why = f"no '## ' heading slugifies to {anchor!r}"
            broken.append((raw, why))
        rows.append({"citation": raw, "file": rel, "anchor": anchor, "resolves": ok})
    return {"total": len(refs), "resolved": resolved,
            "unresolvable": [{"citation": c, "reason": r} for c, r in broken],
            "loose_count": len(loose), "loose_examples": loose[:10],
            "broken": [{"citation": c, "reason": r} for c, r in broken],
            "rows": rows}


def check_against_corpus(doc, chunks):
    """Names and numbers in the document, checked against the whole corpus.

    Strips the source-header lines first. lorewrite.py labels each retrieved
    chunk 'SOURCE FILE'/'HEADING'/'TEXT', and a model that paraphrases the
    layout back out -- "**Lore/Corpus Shared /Hero-Syndrome.md**" -- produces
    proper nouns that are really the prompt's own scaffolding. That is the
    model echoing the harness, not the model inventing lore, and counting it as
    a hallucination trains you to ignore the output.
    """
    body = strip_source_headers(prose_only(doc))
    corpus_text = "\n".join(c["text"] for c in chunks)
    corpus_lower = corpus_text.lower()
    # Normalise thousands separators so 12,800 matches 12800 in the source.
    corpus_noncomma = corpus_text.replace(",", "").lower()
    # The corpus spells; a document that hyphenates or re-slashes them is still referring to the same thing.
    corpus_flat = re.sub(r"[^a-z0-9]+", " ", corpus_lower)
    corpus_flat_compact = corpus_flat.replace(" ", "")

    name_hits, name_misses = [], []
    for name, count in sorted(proper_nouns(body).items()):
        key = name.lower()
        flat = re.sub(r"[^a-z0-9]+", "", key)
        (name_hits if (key in corpus_lower or flat in corpus_flat_compact)
         else name_misses).append({"name": name, "count": count})

    num_hits, num_misses = [], []
    for num, count in sorted(numbers_in(body).items()):
        bare = num.replace(",", "")
        ok = num.lower() in corpus_lower or bare in corpus_noncomma
        (num_hits if ok else num_misses).append(
            {"value": num, "count": count})

    return {"names": {"in_corpus": name_hits, "not_in_corpus": name_misses},
            "numbers": {"in_corpus": num_hits, "not_in_corpus": num_misses}}


def check_retrieval(chunks, queries, k=3):
    """Does the retriever find the right chunk for a hand-written question?

    The expected answer is a heading substring. Deliberately hand-written and
    deliberately few: this is a smoke test that the index is wired up, not a
    benchmark. A benchmark needs relevance judgements over a real query set,
    and inventing those produces a number that measures the inventor.
    """
    index = lorekit.BM25Index(chunks)
    rows = []
    for q in queries:
        hits = index.search(q["query"], k=k)
        anchors = [(c["rel"], c["anchor"]) for c, _ in hits]
        got = any(q["expect"] in a for _, a in anchors)
        rows.append({"query": q["query"], "expect": q["expect"],
                     "hit_at_k": got,
                     "top_anchor": anchors[0][1] if anchors else None})
    hits = sum(r["hit_at_k"] for r in rows)
    return {"k": k, "hits": hits, "total": len(rows),
            "recall_at_k": hits / len(rows) if rows else 0.0, "rows": rows}


# Hand-written probe set. Small on purpose -- see the docstring.
#
# Every probe here was checked against the actual corpus first. The first
# version of this set contained a question about, both leftovers from the invented example world that used to live in
# lore/_template.md. There is no shelf in that corpus, so the probe was
# unanswerable and the retriever was right to miss it. A probe that cannot be
# answered measures the probe author, not the retriever.
#
# KNOWN LIMITATION, measured rather than assumed: relational questions lose.
# "What is X's connection to Y?" MISSES, when one of the two names is a
# high-frequency term -- chunks *about* that entity outrank the single chunk
# that states the relation between the two. BM25 counts words; it has no notion
# of which two entities are related. The probe was reworded to be lexical
# rather than hidden, because a permanently-failing test is a test nobody reads.
#
# Fixing this properly means a dense retriever or a query expansion step, both
# of which need a model and a second unaudited component. Not yet. BM25 first,
# measured.
#
# The probes are DATA ABOUT YOUR CORPUS, so they live in lore/probes/ -- which
# is gitignored, for the same reason the corpus is. A probe set names the
# characters and places of whatever it was written for, and that is your
# material, not this repository's. A clone with no corpus has no probes and
# skips the retrieval check rather than failing.

PROBE_FILE = os.path.join(HERE, "lore", "probes", "retrieval.jsonl")


def load_probes(path=None):
    """Read the probe set. Returns [] when there is none, which is normal.

    A fresh clone has no lore/probes/, because it has no corpus either. That is
    not an error worth a traceback -- it just means the retriever cannot be
    scored until somebody writes a corpus.
    """
    path = path or PROBE_FILE
    if not os.path.isfile(path):
        return []
    probes = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno} is not valid JSON: {e}")
            if "query" not in row or "expect" not in row:
                raise SystemExit(
                    f"{path}:{lineno} needs both 'query' and 'expect'. 'expect' "
                    f"is a substring of the slugified heading of the chunk that "
                    f"should answer the question.")
            probes.append(row)
    return probes


def print_report(report):
    print("\n" + "=" * 72)
    print("FAITHFULNESS")
    print("=" * 72)

    cit = report["citations"]
    print(f"\ncitations   {cit['resolved']}/{cit['total']} resolve")
    for b in cit["broken"]:
        print(f"  BROKEN  {b['citation']}")
        print(f"          {b['reason']}")
    if cit["loose_count"]:
        print(f"  {cit['loose_count']} bracketed reference(s) that are not "
              f"corpus citations: {cit['loose_examples'][:6]}")
        print(f"  ^ these look like citations but cannot be checked. A model "
              f"that cites [1]")
        print(f"    has cited; it has not cited *checkably*. Different problem, "
              f"different fix.")

    names = report["grounding"]["names"]
    nm, nh = len(names["not_in_corpus"]), len(names["in_corpus"])
    print(f"\nproper nouns {nh} in corpus, {nm} not found")
    for m in names["not_in_corpus"]:
        print(f"  NOT IN CORPUS  {m['name']!r} (x{m['count']})")
    if nm:
        print("  ^ check these by hand before concluding anything: capitalisation")
        print("    is a heuristic, and a name can be correct and absent from")
        print("    the corpus. This is a flag, not a verdict.")

    nums = report["grounding"]["numbers"]
    um, uh = len(nums["not_in_corpus"]), len(nums["in_corpus"])
    print(f"\nnumbers      {uh} in corpus, {um} not found")
    for m in nums["not_in_corpus"]:
        print(f"  NOT IN CORPUS  {m['value']!r} (x{m['count']})")

    r = report["retrieval"]
    print(f"\nretrieval    recall@{r['k']} = {r['recall_at_k']:.2f} "
          f"({r['hits']}/{r['total']})")
    for row in r["rows"]:
        mark = "ok " if row["hit_at_k"] else "MISS"
        print(f"  {mark} {row['query']}")
        if not row["hit_at_k"]:
            print(f"       expected a chunk containing {row['expect']!r}, "
                  f"got {row['top_anchor']!r}")

    g = report["gate"]
    print("\n" + "-" * 72)
    print(f"VERDICT: {g['verdict']}")
    for line in g["reasons"]:
        print(f"  - {line}")


def gate(report):
    """One line a CI could read. Fails on any broken citation or invented token."""
    reasons = []
    cit = report["citations"]
    if cit["total"] == 0 and cit["loose_count"] == 0:
        reasons.append("no citations at all -- an uncited document passes "
                       "every other check by default")
    elif cit["total"] == 0:
        reasons.append(
            f"0 resolvable citations, but {cit['loose_count']} bracketed "
            f"reference(s) such as {cit['loose_examples'][:3]} -- the model "
            f"cited in a format that cannot be checked")
    elif cit["broken"]:
        reasons.append(f"{len(cit['broken'])} citation(s) do not resolve")
    n_missing = len(report["grounding"]["names"]["not_in_corpus"])
    if n_missing:
        reasons.append(f"{n_missing} proper noun(s) absent from the corpus")
    u_missing = len(report["grounding"]["numbers"]["not_in_corpus"])
    if u_missing:
        reasons.append(f"{u_missing} number(s) absent from the corpus")
    return {"pass": not reasons,
            "verdict": "PASS" if not reasons else "FAIL",
            "reasons": reasons or ["all citations resolve; no ungrounded names "
                                   "or numbers found"]}


def evaluate_document(doc, chunks, probes=None):
    report = {
        "corpus_chunks": len(chunks),
        "corpus_files": len(sorted({c["rel"] for c in chunks})),
        "citations": check_citations(doc),
        "grounding": check_against_corpus(doc, chunks),
    }
    if probes:
        report["retrieval"] = check_retrieval(chunks, probes)
    report["gate"] = gate(report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--doc", help="a generated document to score. Omit for the "
                                 "instrument check only.")
    p.add_argument("--retrieval", action="store_true",
                   help="score the retriever against the hand-written probes")
    p.add_argument("--k", type=int, default=3, help="retrieval depth")
    p.add_argument("--probes", help=f"probe set, default {os.path.relpath(PROBE_FILE, HERE)}")
    p.add_argument("--out", help="write the full report as JSON here")
    args = p.parse_args()

    chunks = lorekit.load_chunks()
    files = sorted({c["rel"] for c in chunks})
    print(f"corpus: {len(files)} files, {len(chunks)} claims")
    for f in files:
        n = sum(1 for c in chunks if c["rel"] == f)
        print(f"  {n:>4}  {f}")

    probes = load_probes(args.probes)
    if not probes and (args.retrieval or not args.doc):
        print(f"\nno probe set at {os.path.relpath(PROBE_FILE, HERE)} -- the "
              f"retriever cannot be scored.")
        print("That file is gitignored, because a probe set names the characters "
              "and places of")
        print("whatever corpus it was written for. Write one line per probe: "
              '{"query": "...", "expect": "heading-slug-substring"}')
        if not args.doc:
            return 1

    if probes and (args.retrieval or not args.doc):
        r = check_retrieval(chunks, probes, k=args.k)
        print("\n" + "=" * 72)
        print("RETRIEVER -- instrument check, not a benchmark")
        print("=" * 72)
        print(f"recall@{r['k']} = {r['recall_at_k']:.2f} ({r['hits']}/{r['total']})")
        for row in r["rows"]:
            mark = "ok  " if row["hit_at_k"] else "MISS"
            print(f"  {mark} {row['query']}")
            if not row["hit_at_k"]:
                print(f"        expected {row['expect']!r}, got {row['top_anchor']!r}")
        if not args.doc:
            print("\nThis is the instrument check only. To score a document:")
            print("  .venv\\Scripts\\python.exe loreeval.py --doc lore/generated/<file>.md")
            return 0 if r["hits"] == r["total"] else 1

    if args.doc:
        if not os.path.isfile(args.doc):
            raise SystemExit(f"no such document: {args.doc}")
        doc = load_document(args.doc)
        report = evaluate_document(doc, chunks, probes)
        print(f"\ndocument: {args.doc}")
        print_report(report)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nwrote {args.out}")
        return 0 if report["gate"]["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
