"""One way to turn the lore corpus into retrievable chunks.

Same discipline as evalkit.py: one implementation, deterministic, shared by every
caller. If two scripts disagree about what a chunk is, a retrieval number means
nothing, which is the exact failure evalkit.py was written to stop.

Why chunk on '##' and not on a token count: the corpus format already says one
heading is one claim (lore/README.md). Cutting on that boundary means every
chunk is quotable and every chunk has a citation anchor that resolves. Cutting
on 512 tokens instead would slice claims in half and produce chunks that cannot
be cited -- and an uncitable chunk cannot be checked for faithfulness, which
defeats the purpose.

Two levels on purpose:
  load_chunks()  -- splits the corpus, no scoring. Cheap, safe, testable.
  BM25Index      -- the retriever. No I/O, so it can be built in a test.

Nothing here writes files, and importing this module runs nothing.
"""
import math
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "lore", "corpus")

# Words that are capitalised for grammar rather than because they are names.
# Without this, every sentence contributes "The", "In", "It" and the proper
# noun check is unusable. It is a stoplist, not a claim about your corpus.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "in", "on", "at",
    "to", "from", "by", "for", "with", "without", "of", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "there", "here", "he", "she", "they", "we", "you", "i", "not", "no", "yes",
    "when", "while", "after", "before", "during", "into", "over", "under",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "each", "every", "both", "all", "some", "any", "more", "most",
    "also", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "would", "could", "should", "may", "might", "must", "have", "has", "had",
    "do", "does", "did", "does", "does", "up", "down", "out", "off", "about",
    "because", "however", "although", "though", "since", "until", "whether",
    "his", "her", "their", "our", "my", "your", "me", "him", "them", "us",
}

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

# Conservative plural folding only, and deliberately not Porter. A full
# stemmer maps "knits" to "knit" and "knights" to "knight", which still do not
# match each other -- so it would not have rescued the probe that surfaced this,
# and it mangles the proper nouns a lore corpus is mostly made of. Plurals are
# where real queries genuinely diverge from real headings: someone asks about
# "the commandments" and the heading says "commandment".
_SUFFIXES = (("ies", "y"), ("ches", "ch"), ("shes", "sh"), ("sses", "ss"),
             ("xes", "x"), ("zes", "z"), ("s", ""))


def _stem(word):
    """Fold one trailing plural. Applied to corpus and query alike, so the
    two sides cannot disagree. Never strips below three characters."""
    if len(word) < 4:
        return word
    for suffix, repl in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + repl
    return word
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")
_COMMENT_RE = re.compile(r"^\s*<!--.*?-->", re.DOTALL)


def slugify(text):
    """'Knights are exempt' -> 'knights-are-exempt'. The citation anchor rule.

    Deliberately simple and deliberately total: it never raises, because a
    citation that cannot be slugified is a broken citation either way, and the
    checker wants to report that rather than crash on it.
    """
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)


def tokenize(text):
    """Lowercased, plural-folded word tokens. Keeps digits, drops punctuation."""
    return [_stem(w) for w in _WORD_RE.findall(text.lower())]


def iter_corpus_files(root=None):
    """Every .md under the corpus, sorted, skipping dotfiles and _sources.md.

    _sources.md is provenance ABOUT the corpus, not lore from it. Retrieving it
    would let a document cite its own excluded-facts list as if it were source
    material -- which is how a known-bad fact gets laundered back in.
    """
    root = root or CORPUS_DIR
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"no corpus at {root!r}. Put your Markdown in lore/corpus/ -- see "
            f"lore/README.md. (That folder is gitignored, so it will be empty "
            f"on a fresh clone.)")
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith(".") or not fn.endswith(".md"):
                continue
            if fn == "_sources.md":
                continue
            out.append(os.path.join(dirpath, fn))
    return out


def load_chunks(root=None, strict=True):
    """Split the corpus into one chunk per '##' claim.

    `strict` decides what happens when a file yields no chunks at all. It is on
    by default, and it is there because of a bug this function shipped for an
    hour: a variable was renamed mid-function, every '##' heading produced a
    chunk that was then dropped on the floor, and the function returned an empty
    list -- no exception, no warning. The caller carried on, printed
    'corpus: 6 files, 0 claims', and reported recall@3 = 0.00 as though that
    were a measurement of the retriever.

    A parser that finds nothing is almost always broken rather than being told
    about a corpus that genuinely has no '##' headings. Make that loud.
    """
    """Split the corpus into one chunk per '##' claim.

    Returns a list of dicts, sorted by (file, index) so the order is stable
    across machines and across runs -- a corpus that reorders itself between two
    evaluations makes the two sets of numbers incomparable, which is rule 2 of
    AGENTS.md applied to lore.

    Each chunk: {file, rel, heading, anchor, body, text, tokens}
      file    absolute path
      rel     path relative to the repo root, e.g. 'lore/corpus/<work>/story.md'
              -- this is what a citation points at, so it is part of a chunk's
              identity: same text under a different path is a different citation
      heading the claim, verbatim
      anchor  slugified heading, the thing citations point at
      body    the prose under the heading, comments stripped
      text    heading + body
      tokens  tokenize(text)
    """
    chunks = []
    barren = []
    for path in iter_corpus_files(root):
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        rel = os.path.relpath(path, HERE).replace("\\", "/")
        # Strip HTML comments first: _template.md and _sources.md style notes are
        # instructions to the author, not lore, and would otherwise be retrieved.
        raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
        lines = raw.splitlines()
        cur = None
        before = len(chunks)
        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                if cur:
                    chunks.append(cur)
                cur = {"file": path, "rel": rel, "heading": m.group(1),
                       "anchor": slugify(m.group(1)), "body_lines": []}
                continue
            if cur is None:
                continue  # the # title and any preamble before the first ##
            cur["body_lines"].append(line)
        if cur:
            chunks.append(cur)
        if len(chunks) == before:
            barren.append(rel)

    if not chunks:
        files = len(iter_corpus_files(root))
        msg = (f"load_chunks() found 0 claims in {files} file(s) under "
               f"{root or CORPUS_DIR!r}.")
        if strict:
            raise ValueError(
                msg + "\nEvery corpus file must use '## ' headings -- one heading "
                "per claim (see lore/README.md).\n"
                "This also fires when the parser itself is broken, which is the "
                "more likely explanation. Do not return an empty list from here: "
                "a caller will carry on and report recall 0.00 as though it were "
                "a measurement of the retriever. Pass strict=False to override.")
        raise ValueError(msg)
    if barren and strict:
        raise ValueError(
            f"{len(barren)} corpus file(s) produced no claims: {barren}\n"
            f"They are being read but nothing in them matches the '## ' heading "
            f"rule, so they are invisible to retrieval. If that is deliberate, "
            f"pass strict=False.")

    for i, c in enumerate(chunks):
        c["body"] = "\n".join(c["body_lines"]).strip()
        c["text"] = (c["heading"] + "\n" + c["body"]).strip()
        c["tokens"] = tokenize(c["text"])
        c["index"] = i
        # The exact string a citation must contain, ready to copy.
        #
        # A chunk's `rel` is 'lore/corpus/<work>/<file>.md' because that is the
        # path on disk, but a citation is written 'corpus/<work>/<file>.md' --
        # citations are resolved against HERE/lore, not HERE. Handing the model
        # `rel` and telling it to strip the leading 'lore/' works right up
        # until the model forgets, and then it emits
        # [[corpus/lore/corpus/evenicle-1/story.md#...]] and the citation does
        # not resolve. The first Kaggle run did exactly that. So the finished
        # citation is computed here, once, and the model is asked to copy it.
        #
        # `c["rel"]`, NOT the loop variable `rel`. `rel` belongs to the file
        # loop above and still holds the LAST file's path when this loop runs,
        # so using it stamped every chunk in the corpus with the same citation
        # path -- 125 of 140 pointing at world-of-eve.md. A citation is a
        # function of the chunk, never of where the chunk sits in a loop.
        c["cite"] = c["rel"][len("lore/"):] if c["rel"].startswith("lore/") else c["rel"]
        c["citation"] = f"[[{c['cite']}#{c['anchor']}]]"

    # Self-check. A chunk whose citation does not derive from its own path is
    # broken in the one way that is invisible downstream: retrieval still works,
    # the document still reads fine, and the gate simply reports "file does not
    # exist" for citations that look perfect. That bug shipped once already
    # (every chunk cited world-of-eve.md, from a loop variable holding the last
    # file's path), so it is asserted here rather than trusted.
    broken = [c["citation"] for c in chunks
              if c["cite"] not in c["rel"]
              or c["anchor"] != slugify(c["heading"])]
    if broken:
        raise AssertionError(
            f"{len(broken)} chunk(s) have a citation that does not match their "
            f"own file and heading. First: {broken[0]!r}\n"
            f"A citation is computed from c['rel'] and c['heading'] only. If a "
            f"loop variable leaks in here, every citation in the corpus silently "
            f"points at one file.")
    return chunks


class BM25Index:
    """Okapi BM25 over the chunks. Deterministic: no sampling, no seed.

    Chosen over a dense retriever deliberately. A dense retriever needs an
    embedding model, which means a GPU and a second thing whose output nobody
    has audited; BM25 is 60 lines of arithmetic you can check by reading, it
    runs in milliseconds on 140 chunks, and when it retrieves the wrong chunk
    you can see why by looking at the term counts. Get the cheap baseline
    measured before spending anything on the expensive one.
    """

    def __init__(self, chunks, k1=1.5, b=0.75):
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.N = len(chunks)
        self.df = Counter()
        self.tf = []
        self.lens = []
        for c in chunks:
            counts = Counter(c["tokens"])
            self.tf.append(counts)
            self.lens.append(len(c["tokens"]) or 1)
            for term in counts:
                self.df[term] += 1
        self.avgdl = sum(self.lens) / self.N

    def idf(self, term):
        # Robertson/Sparck-Jones IDF, floored at zero so a term appearing in
        # every chunk contributes nothing instead of a negative score.
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return max(0.0, math.log(1 + (self.N - df + 0.5) / (df + 0.5)))

    def search(self, query, k=5):
        """Top-k chunks for a query string. Returns [(chunk, score), ...]."""
        q_terms = tokenize(query)
        scored = []
        for i, c in enumerate(self.chunks):
            tf, dl = self.tf[i], self.lens[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf(term) * (f * (self.k1 + 1)) / denom
            if s > 0:
                scored.append((c, s))
        # Sort by score desc, then by rel path and heading, so ties are stable.
        scored.sort(key=lambda t: (-t[1], t[0]["rel"], t[0]["heading"]))
        return scored[:k]


def citations_in(text):
    """Extract [[corpus/path.md#anchor]] references. Returns [(raw, rel, anchor)].

    Only the corpus/ form is recognised. A citation to some other path is not a
    citation to a source, and treating it as one would let a document appear
    sourced by pointing at nothing.
    """
    pat = re.compile(r"\[\[(corpus/[^\]|#]+\.md)#([^\]|]+)(?:\|[^\]]*)?\]\]")
    return [(m.group(0), m.group(1), m.group(2).strip())
            for m in pat.finditer(text)]


def headings_in_file(path):
    """Every '##' heading in a file, slugified. Used to check citation anchors."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = re.sub(r"<!--.*?-->", "", f.read(), flags=re.DOTALL)
    except OSError:
        return None
    return {slugify(m.group(1))
            for m in (_HEADING_RE.match(l) for l in raw.splitlines()) if m}
