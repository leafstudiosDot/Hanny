# Lore corpus — the format your Markdown goes in

This file is the **format contract**, and it is the only tracked file in
`lore/`. Everything else in here is yours and is gitignored.

A clone of this repository will show you this file, `_template.md`, and three
empty directories. **That is expected.** The corpus holds other people's
copyrighted writing — wiki dumps, lore documents, book and game texts — and
that has no place in git history, no matter how private the repository is. So
the instructions ship; the material does not.

The example content below is **invented filler**, written only to show the shape
of a file. None of it is real lore about anything.

---

## Layout

One folder per work, one file per *kind* of content within it.

```
lore/corpus/
  _sources.md          provenance: which page, which revision, read when
  shared/              setting and lore that spans more than one work
  <work-name>/
    story.md           plot, chapter by chapter
    characters.md      the cast
    <system>.md        a game system, if the work has one
```

Replace `<work-name>` with your own works. Name folders after the work — short,
lowercase, hyphenated — so citations stay readable.

Why folders rather than one flat directory: a search for one work's illness
system should return that system, not a paragraph about a different work's
protagonist.

Why `shared/` rather than a copy in each folder: **duplicated lore drifts.**
Two files claiming the same fact will diverge the first time one is edited, and
a silently-wrong duplicate is worse than a missing fact, because the citation
still resolves and nothing reports a problem. Put cross-work material in
`shared/`, and cite it with its subfolder:

```markdown
[[corpus/shared/<setting>.md#<the-claim-stated-in-plain-words>]]
```

Keep `_sources.md` at the root of `corpus/` even when everything else is in
folders — provenance is about the whole corpus, not one work.

## One heading per fact, so a fact can be cited

The single most important convention. A passage carrying exactly one claim can
be quoted and verified. A passage carrying six can only be checked as a block,
and "is this supported?" stops being answerable.

```markdown
## <the claim, stated in plain words>

<one or two sentences supporting it. If the claim is disputed, say so and say
who disputes it.>
```

Put the claim in the heading, in words a reader could check without reading the
paragraph. `[the shelf broke in 1147 because of a failed binding]` is
checkable; `[History]` is not.

## Citations must resolve to something real

Cite the file **and the heading**, and make sure the heading exists:

```markdown
The shelf has been impassable since then [[corpus/shared/<setting>.md#<exact-heading-text>]].
```

The anchor has to exist in that file. A citation that resolves to a missing
anchor is worse than no citation, because it looks like evidence.

Avoid bare URLs in the corpus itself. A local file and a heading stays valid
offline, stays diffable, and cannot rot when a site reorganises.

## Style that survives retrieval

- **No front matter, no YAML, no metadata blocks.** They add tokens to every
  chunk and teach nothing.
- **Do not start a file with a title that duplicates the first heading.** It
  splits the first chunk badly.
- **Plain prose or plain lists.** Tables get mangled when a chunk cuts through
  them. If a table is genuinely the clearest form, keep it — but put a prose
  sentence directly above it restating what it says.
- **One language.** Mixed-language chunks retrieve badly and read worse.

## `lore/prompts/` and `lore/generated/`

- `lore/prompts/` — the tasks you want answered, one file per request, so a
  failed generation can be re-run without re-prompting.
- `lore/generated/` — documents the model produced, kept next to the corpus
  that produced them.

Both are gitignored for the same reason `corpus/` is.

## Before any of this is used

**Write down what "faithful" means for your corpus, because that is the
definition the evaluation will check against.** "No invented names" is checkable
by string matching. "No invented facts" is not, and pretending otherwise
produces a measurement that passes.

The first honest version is:

1. every factual claim in the document has a citation that resolves
2. every proper noun in the document appears somewhere in the corpus
3. every number, date and quantity in the document appears in the corpus

That is a real, falsifiable test, and it needs no judge model. It will not catch
a fluent misreading of a source, and it should not be described as if it does.
