"""Build the upload archives, split by who is allowed to see them.

Why a script and not a tar line in the README
----------------------------------------------
The first version of this was a hand-typed `Compress-Archive` command in the
README. It omitted `finetune.py` entirely, so a Kaggle run failed with
`No such file or directory` and the obvious conclusion -- a broken upload -- was
wrong. A file list maintained in prose is a file list that will drift. This
script owns the list, checks it against what exists, and refuses to build when
it is inconsistent.

The split
---------
Two archives instead of one, because the one archive mixed three sensitivity
tiers and "which of these can leave my machine?" was answerable only by reading
a README carefully:

  hanny-code.zip   everything shareable. Scripts, the public probe sets, the
                   baseline. This is what a public model or dataset repo would
                   contain. Safe to commit and to publish.

  hanny-data.zip   the corpus cache, which carries the private source text plus
                   the `report` block holding real folder paths and the names of
                   15 secret-bearing files, plus the gitignored real safety
                   probes. Never leaves the machine or a private Kaggle dataset.

`tar` does not read `.gitignore`, which is why the real probes reach Kaggle at
all -- the archive is built from an explicit list, not from a directory sweep.

The leak check
--------------
Building the shareable archive runs a scan for the exact strings that must not
appear in it: every folder path and every source filename recorded in the
private corpus report. AGENTS.md says folder-level detail belongs in the
gitignored report and must never reach a tracked file. That rule is easy to
state and easy to break by pasting a path into a comment, so it is enforced
here rather than trusted.

Usage:
    .venv\\Scripts\\python.exe pack.py                # build both
    .venv\\Scripts\\python.exe pack.py --check        # verify, build nothing
    .venv\\Scripts\\python.exe pack.py --which code   # only the shareable one
"""
import argparse
import json
import os
import re
import sys
import zipfile

import utf8console

utf8console.enable()

HERE = os.path.dirname(os.path.abspath(__file__))

# --- the two manifests ------------------------------------------------------

# Shareable. Any change here is a decision about what the world sees, so the
# list is explicit rather than a glob over the repo.
CODE_FILES = [
    "finetune.py",
    "sft.py",
    "chat.py",
    "qwenkit.py",
    "data.py",
    "evalkit.py",
    "chateval.py",
    "utf8console.py",
    "model.py",
    "baseline.py",
    "gpu_check.py",
    "baseline.json",
    "requirements.txt",
    "README.md",
    "DATASETS.md",
    "PUBLISHING.md",
    "evalsets/chat_probe.jsonl",
    "evalsets/safety_probe_public.jsonl",
    "evalsets/README.md",
    # The chat track. lorewrite.py imports lorekit, so all three travel
    # together; listing two of them would import-fail on Kaggle with a
    # ModuleNotFoundError that reads like a code bug. This is the second time
    # this list has silently dropped a script the notebook actually runs, so
    # pack.py aborts when a listed file is missing -- but it cannot abort on a
    # file nobody remembered to list. See check_code_files().
    "lorekit.py",
    "loreeval.py",
    "lorewrite.py",
]

# Private. The corpus is found by fingerprint, not by name, because the
# fingerprint changes whenever the corpus does and a stale hardcoded name is how
# you train on the wrong data without noticing.
# Private files: the corpus cache, the real safety probes, and the lore corpus.
#
# Lore is OPTIONAL and that is deliberate. The corpus cache and the probes are
# required because the Kaggle cells cannot run without them. The lore corpus is
# not: a clone with no lore folder must still be able to build a code archive,
# and making lore required would turn "I have no lore yet" into "pack.py
# crashes", which is the wrong way round. Entries are (folder, pattern, required).
#
# Lore goes in the PRIVATE archive only. It is other people's copyrighted
# writing; hanny-code.zip stays publishable, and a lore file in it would be a
# leak the post-build scan cannot see, because that scan only reads what it is
# about to ship.
DATA_GLOBS = [
    ("upload", "corpus-*.json", True),
    ("evalsets", "safety_probe.jsonl", True),
    ("lore/corpus", "**/*.md", False),
    ("lore/probes", "*.jsonl", False),
]

# Strings that must never appear in the shareable archive. Built from the real
# corpus report at run time, not hardcoded -- see private_path_tokens().
MIN_TOKEN_LEN = 6  # short fragments collide with ordinary English

# Folder names that are also ordinary words are useless as leak signals: a
# machine that has `scripts/`, `upload/` and `project/` on disk would flag every
# script in the repo. A segment only becomes a token when it is *distinctive*,
# which is what these tests look for.
COMMON_SEGMENTS = {
    "scripts", "script", "upload", "uploads", "project", "projects", "public",
    "private", "action", "actions", "parser", "parsers", "versions", "version",
    "search", "reports", "report", "sessions", "session", "studio", "studios",
    "source", "sources", "assets", "images", "static", "shared", "common",
    "tools", "tool", "build", "builds", "dist", "output", "outputs", "temp",
    "docs", "doc", "tests", "test", "utils", "util", "data", "code", "main",
    "index", "types", "config", "configs", "core", "base", "lib", "libs",
    "packages", "services", "service", "models", "model", "utils", "examples",
    "example", "sample", "samples", "default", "other", "misc", "local",
    "build", "venv", "env", "git", "cache", "logs", "log", "tmp", "target",
    "obj", "bin", "src", "include", "vendor", "node_modules", "coverage",
}


def _is_distinctive(segment):
    """True when a path segment is unlikely to appear as an ordinary word.

    Signals, in order of reliability: a digit, an underscore, a dot (i.e. a
    filename), or mixed case. `leafstudiosDot` is caught by the last one;
    `scripts` is caught by the stoplist.
    """
    low = segment.lower()
    if low in COMMON_SEGMENTS:
        return False
    # A leading dot is a convention, not a secret: `.build`, `.venv` and `.git`
    # are every project's, and flagging them would train you to ignore the scan.
    bare = low.lstrip(".")
    if bare in COMMON_SEGMENTS:
        return False
    if any(ch.isdigit() for ch in segment):
        return True
    if "_" in segment or "." in segment:
        return True
    # camelCase / PascalCase: a lowercase word has no uppercase at all
    if any(ch.isupper() for ch in segment):
        return True
    return False



def find_private_files():
    """Resolve DATA_GLOBS to concrete files. Returns [(arcname, abspath)].

    A missing REQUIRED glob is fatal and says how to create it. A missing
    optional glob is reported and skipped -- see the comment on DATA_GLOBS for
    why lore is optional and the corpus cache is not.
    """
    import glob
    found = []
    skipped = []
    for folder, pattern, required in DATA_GLOBS:
        hits = sorted(glob.glob(os.path.join(HERE, folder, pattern),
                                recursive=True))
        if not hits:
            if required:
                raise SystemExit(
                    f"no file matching {folder}/{pattern}\n"
                    f"  Run:  .venv\\Scripts\\python.exe data.py --export-plain upload")
            skipped.append(f"{folder}/{pattern}")
            continue
        for path in hits:
            arc = os.path.relpath(path, HERE).replace("\\", "/")
            found.append((arc, path))
    if skipped:
        print("optional inputs not present, skipping: " + ", ".join(skipped))
        print("  (a clone with no lore corpus still builds both archives)")
    return found


def private_path_tokens():
    """Every real folder path and source filename recorded in the corpus report.

    Returns a set of lowercase tokens. Empty when no corpus cache is present --
    the leak check then degrades to the static rules below rather than failing,
    so a fresh clone can still build a code archive.
    """
    import gzip
    import glob

    caches = glob.glob(os.path.join(HERE, "cache", "corpus-*.json.gz"))
    if not caches:
        return set(), None
    path = max(caches, key=os.path.getmtime)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        blob = json.load(f)
    report = blob.get("report", {})

    tokens = set()

    def add_path(value):
        """Add a full multi-segment path verbatim (slashes included).

        This is the highest-signal check and has no false positives worth
        worrying about, so it is always added regardless of segment contents.
        """
        if not isinstance(value, str) or not value.strip():
            return
        raw = value.strip().replace("\\", "/")
        norm = raw.lower()
        if len(norm) >= MIN_TOKEN_LEN:
            tokens.add(norm)
        # Split the ORIGINAL, not the lowercased copy. Distinctiveness is judged
        # on `leafstudiosDot`, which has an uppercase letter; lowercasing first
        # turns it into `leafstudiosdot`, which looks like an ordinary word and
        # gets dropped. That bug made the scan report clean while the name was
        # still sitting in the README.
        for part in raw.split("/"):
            if len(part) >= MIN_TOKEN_LEN and _is_distinctive(part):
                tokens.add(part.lower())

    def add_filename(value):
        """Add a filename unconditionally: secret_files entries are the whole point."""
        if not isinstance(value, str):
            return
        base = value.replace("\\", "/").rstrip("/").split("/")[-1].strip().lower()
        if len(base) >= MIN_TOKEN_LEN:
            tokens.add(base)

    # sources is keyed by folder path; secret_files holds [path, reason] pairs.
    for folder in (report.get("sources") or {}):
        add_path(folder)
    for entry in (report.get("secret_files") or []):
        if isinstance(entry, (list, tuple)) and entry:
            add_path(entry[0])
            add_filename(entry[0])
        else:
            add_path(entry)
    for folder in (report.get("missing_dirs") or []):
        add_path(folder)
    return tokens, os.path.basename(path)


# Phrases from the gitignored probes. A shareable archive that contains one of
# these is a publish accident, so they are checked by name too rather than
# trusting the token scan to notice.
FORBIDDEN_SUBSTRINGS = (
    "13-year-old", "keylogger", "pipe bomb", "methamphetamine",
    "self-harm without", "hurt myself", "poison someone",
    "private medical records",
)

# Stand-in for a tracked allowance list, which would defeat the purpose: the
# token being allowed would then live in a tracked file itself. See
# pack.local.json.
LOCAL_CONFIG = "pack.local.json"


def load_local_allow():
    """Read accepted tokens from pack.local.json, which is gitignored.

    An allowance has to live somewhere, and nowhere else works. Putting it in
    pack.py would put the allowed string into a tracked file, which is the very
    thing the scan exists to prevent. Putting it on the command line means
    re-typing it for every build, and a check that fails on a decision already
    made is a check people stop running.

    Returns (tokens, reason_by_token). Missing file is normal -- a fresh clone
    has none.
    """
    path = os.path.join(HERE, LOCAL_CONFIG)
    if not os.path.exists(path):
        return [], {}
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: {LOCAL_CONFIG} could not be read ({e}); ignoring it.")
        return [], {}
    allow = blob.get("allow") or []
    reasons = {}
    for entry in allow:
        if isinstance(entry, str):
            reasons[entry.lower()] = "recorded as deliberate"
        elif isinstance(entry, dict) and entry.get("token"):
            reasons[str(entry["token"]).lower()] = entry.get("reason", "no reason given")
    return sorted(reasons), reasons


def save_local_allow_example():
    """Print the shape of pack.local.json. Not written automatically."""
    return {
        "allow": [
            {"token": "<string you deliberately publish>",
             "reason": "<why it is fine: e.g. copyright attribution>"},
        ],
    }


# Local-only tools, deliberately not in the Kaggle archive. pack.py holds the
# FORBIDDEN_SUBSTRINGS list it scans for, so shipping it makes its own scanner
# trip; upload.py imports pack, so it needs the local tree to run at all.
LOCAL_ONLY = {"pack.py", "upload.py"}


def check_code_files():
    """Every listed shareable file must exist, and every top-level script should be listed."""
    missing = [f for f in CODE_FILES if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        raise SystemExit(
            "pack.py lists files that do not exist:\n  "
            + "\n  ".join(missing)
            + "\nEither they were deleted or CODE_FILES is out of date. Fix the "
              "manifest rather than deleting the check -- that is exactly the "
              "bug that dropped finetune.py from the first upload command.")
    listed = set(CODE_FILES) | LOCAL_ONLY
    unlisted = sorted(
        name for name in os.listdir(HERE)
        if name.endswith(".py") and name not in listed
        and os.path.isfile(os.path.join(HERE, name)))
    if unlisted:
        print("  warning: top-level Python files are in neither CODE_FILES nor "
              "LOCAL_ONLY: " + ", ".join(unlisted))
        print("           they will not exist in a Kaggle notebook built from the archive.")
    return CODE_FILES


def scan_for_leaks(names, tokens, allow=()):
    """Refuse to build the shareable archive if it carries private strings.

    Reads the candidate files as text. Binary-ish files are skipped rather than
    guessed at: a decode error means the file is not source, and the real risk
    is a path pasted into a comment.

    `allow` holds tokens the user explicitly accepted via --allow. That exists
    because a hard failure on legitimate content trains people to switch the
    check off, which is worse than no check at all. A copyright line naming the
    holder is ordinary attribution rather than a corpus leak, and whether that
    name belongs in a public repo is the owner's decision -- not a substring
    match's.
    """
    allow = {a.lower() for a in allow}
    problems = []
    for name in names:
        path = os.path.join(HERE, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().lower()
        except (UnicodeDecodeError, OSError):
            continue
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in text and bad not in allow:
                line = text[:text.index(bad)].count("\n") + 1
                problems.append(f"{name}:{line} contains forbidden phrase {bad!r}")
        for tok in tokens:
            if tok in text and tok not in allow:
                line = text[:text.index(tok)].count("\n") + 1
                problems.append(f"{name}:{line} contains private path fragment {tok!r}")
    return problems


# Model weights never belong in either archive.
#
# The leak scan cannot see them: scan_for_leaks() skips any file that raises
# UnicodeDecodeError, and a safetensors blob always does. So an adapter built
# from the user's private code would pass the leak check that exists precisely
# to catch private material -- a check that passes for the wrong reason, which
# is worse than no check.
#
# There is also a workflow reason. hanny-data.zip is the *input* archive: the
# corpus, the probes, the lore. Putting a training *output* in it makes the
# input depend on the result of training on that input, and every adapter
# update forces a re-upload of the whole thing. Ship the adapter as its own
# private Dataset, or attach the previous notebook's output via Input > Add
# Input, which needs no upload at all.
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".ckpt", ".bin")


def find_weights(names):
    """Every archive entry that is model weights.

    `names` may be plain arcname strings (CODE_FILES is a list of paths) or
    (arcname, abspath) pairs (DATA_GLOBS resolves to pairs), so normalise first
    rather than assuming -- this function blew up on the first call otherwise.
    """
    arcs = []
    for entry in names:
        arc = entry[0] if isinstance(entry, (tuple, list)) else entry
        if arc.lower().endswith(WEIGHT_SUFFIXES):
            arcs.append(arc)
    return arcs


def write_archive(out_name, entries, kind):
    """Write a zip from [(arcname, abspath)], with a manifest describing it.

    The manifest name carries the kind. Both archives unpack into the same
    /kaggle/working, and a shared MANIFEST.json meant the second unpack silently
    overwrote the first -- so afterwards you could not tell which files had come
    from the private archive.

    No absolute path is recorded anywhere. An earlier version wrote
    `built_from: HERE`, which put the full local path -- parent folder name
    included -- into the very archive whose purpose is to not contain that.
    """
    out = os.path.join(HERE, out_name)
    entries = list(entries)
    manifest = {
        "kind": kind,
        "archive": out_name,
        "files": sorted(arc for arc, _ in entries),
        "note": ("safe to publish" if kind == "code" else
                 "PRIVATE -- contains the corpus cache with real folder paths, "
                 "and the gitignored safety probes. Do not commit or publish."),
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, path in entries:
            z.write(path, arc)
        z.writestr(f"MANIFEST.{kind}.json", json.dumps(manifest, indent=2))
    return out, manifest


def verify_built(out, kind, tokens, allow=()):
    """Re-open the finished archive and scan its actual bytes.

    The pre-build scan only reads the listed source files, so anything the
    packager itself generates is invisible to it -- which is exactly how
    `built_from: HERE` shipped the local path inside the publishable archive.
    This runs on the output bytes, so generated content is covered too.
    """
    allow = {a.lower() for a in allow}
    problems = []
    with zipfile.ZipFile(out) as z:
        for name in z.namelist():
            try:
                blob = z.read(name).decode("utf-8").lower()
            except UnicodeDecodeError:
                continue  # binary payload (the corpus); no text to scan
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad in blob and bad not in allow:
                    problems.append(f"{name} contains forbidden phrase {bad!r}")
            for tok in tokens:
                if tok in blob and tok not in allow:
                    problems.append(f"{name} contains private path fragment {tok!r}")
    return problems


def restore_artifacts(zip_path, force=False):
    """Unpack a Kaggle artifacts.zip into the repo, refusing silent collisions.

    `Expand-Archive -Force` is the obvious way to do this and the wrong one. This
    repo can hold two adapters both called 'm4' -- finetune.py is not
    bit-reproducible, and re-running overwrites the folder without asking -- so
    -Force would swap one set of weights for another with no record of which is
    which, and chateval.py would later hash whichever survived.

    So: refuse to overwrite by default, and always report the provenance
    (loss_after, kernels) of whatever was restored. An adapter that arrives
    without its result.json is restored but flagged, because its provenance is
    then unknown and an unknown-provenance adapter is not evidence of anything.
    """
    import zipfile

    if not os.path.isfile(zip_path):
        raise SystemExit(f"no such file: {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        # Only these prefixes are ever restored. A zip that carries lore/corpus
        # would overwrite the local corpus with whatever was in the session, and
        # the corpus is hand-edited -- that must never happen from an import.
        ok_prefixes = ("checkpoints/", "lore/generated/")
        rejected = [n for n in names
                    if not n.startswith(ok_prefixes)]
        if rejected:
            print(f"refusing {len(rejected)} entr(y/ies) outside "
                  f"checkpoints/ and lore/generated/:")
            for n in rejected[:10]:
                print(f"  - {n}")
            print("An artifacts.zip should only ever carry a training run and the "
                  "documents it produced.\nIf that is wrong, unpack it by hand "
                  "and look at what you are overwriting.")
            raise SystemExit(1)

        targets = [os.path.join(HERE, n.replace("/", os.sep)) for n in names]
        clashes = [os.path.relpath(t, HERE).replace("\\", "/")
                   for t in targets if os.path.exists(t)]
        if clashes and not force:
            print(f"refusing to overwrite {len(clashes)} existing file(s):")
            for c in clashes[:10]:
                print(f"  - {c}")
            print("\nTwo adapters can share the name 'm4'. Overwriting one with "
                  "the other\ndestroys the record of which run produced which "
                  "weights.\n\n"
                  "Move the old one aside first, or re-run with --force if you "
                  "are certain:")
            print(f"  .venv\\Scripts\\python.exe pack.py --restore {zip_path} --force")
            raise SystemExit(1)
        if clashes:
            print(f"--force: overwriting {len(clashes)} existing file(s)\n")

        print(f"{os.path.basename(zip_path)}: {len(names)} file(s)\n")
        for n in sorted(names):
            print(f"  + {n}")
        z.extractall(HERE)

    # Report provenance of every restored run, because that is the only thing
    # that distinguishes two adapters with the same folder name.
    runs = sorted({n.split("/")[1] for n in names
                   if n.startswith("checkpoints/") and n.count("/") >= 2})
    if not runs:
        print("\nno checkpoints in the bundle -- nothing to report")
        return
    print("\nprovenance of what was restored:")
    for run in runs:
        meta = os.path.join(HERE, "checkpoints", run, "result.json")
        if os.path.exists(meta):
            try:
                with open(meta, encoding="utf-8") as f:
                    j = json.load(f)
                print(f"  {run:<12} loss_after={j.get('loss_after')}  "
                      f"seconds={j.get('seconds')}  "
                      f"kernels={j.get('kernels', 'not recorded')}")
                print(f"  {'':<12} use: --adapter checkpoints/{run}/adapter-best")
            except (OSError, ValueError) as e:
                print(f"  {run:<12} result.json unreadable ({e}) -- treat as "
                      f"suspect")
        else:
            print(f"  {run:<12} NO result.json -- these weights have no "
                  f"recorded provenance.")
            print(f"  {'':<12} usable, but chateval.py should hash them before "
                  f"any number is quoted.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--which", choices=["code", "data", "both"], default="both")
    p.add_argument("--check", action="store_true",
                   help="run every check and build nothing")
    p.add_argument("--skip-leak-scan", action="store_true",
                   help="build the code archive without scanning. Only for a "
                        "clone with no corpus cache and no network.")
    p.add_argument("--allow", action="append", default=[], metavar="TOKEN",
                   help="accept one flagged string as deliberate. Repeatable. "
                        "Use for legitimate attribution such as a copyright "
                        "line; do not use it to silence a real corpus leak.")
    p.add_argument("--restore", metavar="ZIP",
                   help="unpack a Kaggle artifacts.zip (checkpoints + generated "
                        "documents) into the repo, refusing to overwrite "
                        "existing files. See restore_artifacts().")
    p.add_argument("--force", action="store_true",
                   help="with --restore, overwrite files that already exist")
    args = p.parse_args()

    if args.restore:
        restore_artifacts(args.restore, force=args.force)
        return 0

    local_allow, reasons = load_local_allow()
    allow = sorted({a.lower() for a in (list(args.allow) + local_allow)})
    if allow:
        print("accepted as deliberate (not a leak):")
        for tok in allow:
            origin = "command line" if tok in {a.lower() for a in args.allow} else LOCAL_CONFIG
            print(f"  - {tok!r}  [{origin}] {reasons.get(tok, '')}")
        print()

    print(f"Hanny packager\n  repo: {HERE}\n")
    code_files = check_code_files()
    private_files = find_private_files()
    tokens, corpus_name = private_path_tokens()

    print(f"shareable files : {len(code_files)}")
    print(f"private files   : {len(private_files)}")
    if corpus_name:
        print(f"leak scan against: {corpus_name} ({len(tokens)} path tokens)")
    else:
        print("leak scan against: (no corpus cache found -- static rules only)")

    # Refuse model weights before anything is built, not after. See
    # WEIGHT_SUFFIXES: the leak scan cannot detect them, so this has to be an
    # explicit rule rather than something the scan happens to catch.
    stray = find_weights(code_files) + find_weights(private_files)
    if stray:
        print("REFUSING TO BUILD: model weights in an archive\n")
        for arc in stray:
            print(f"  - {arc}")
        print("\nA LoRA adapter is learned from your private source code. The leak")
        print("scan reads files as text and skips anything that fails to decode, so")
        print("it would pass weights straight through -- a check that passes for the")
        print("wrong reason.\n")
        print("Ship it separately instead:")
        print("  - a private Kaggle Dataset of its own, or")
        print("  - 'Save output' on the notebook, then Input > Add Input > the")
        print("    notebook, which needs no upload at all.")
        raise SystemExit(1)

    if not args.skip_leak_scan:
        problems = scan_for_leaks(code_files, tokens, allow)
        if problems:
            print("\nLEAK CHECK FAILED -- not building the shareable archive:\n")
            for prob in problems:
                print(f"  - {prob}")
            print("\nEach of these is a string from your private corpus that "
                  "would ship in a public archive. Fix it in the file, or record "
                  "it as deliberate in pack.local.json:\n")
            print(json.dumps(save_local_allow_example(), indent=2))
            print("\nUse that only for mentions you mean to publish, such as a "
                  "copyright line naming the holder. Do not use it to silence a "
                  "real corpus leak -- that is the one case where being wrong "
                  "publishes someone's private source tree.")
            raise SystemExit(1)
        print("leak scan       : clean")

    if args.check:
        print("\n--check only, nothing written")
        return

    built = []
    if args.which in ("code", "both"):
        code_entries = [(f, os.path.join(HERE, f)) for f in code_files]
        out, man = write_archive("hanny-code.zip", code_entries, "code")
        size = os.path.getsize(out)
        print(f"\nwrote {os.path.basename(out)}  ({size:,} bytes, {len(code_entries)} files)")
        if not args.skip_leak_scan:
            problems = verify_built(out, "code", tokens, allow)
            if problems:
                print("  POST-BUILD SCAN FAILED -- deleting it rather than shipping:")
                for prob in problems:
                    print(f"    - {prob}")
                os.remove(out)
                raise SystemExit(1)
            print("  post-build scan: clean")
        print("  -> safe to publish")
        built.append(out)

    if args.which in ("data", "both"):
        out, man = write_archive("hanny-data.zip", private_files, "data")
        size = os.path.getsize(out)
        print(f"\nwrote {os.path.basename(out)}  ({size:,} bytes, {len(private_files)} files)")
        print("  -> PRIVATE. Upload as a private Kaggle dataset. Never commit.")
        built.append(out)

    print("\nKaggle: attach ONE dataset containing both zips. Cell 2 unzips each "
          "into /kaggle/working:")
    print("  !unzip -q /kaggle/input/<user>/<dataset>/hanny-code.zip -d /kaggle/working")
    print("  !unzip -q /kaggle/input/<user>/<dataset>/hanny-data.zip -d /kaggle/working")
    print("\nOr attach them as two separate datasets if you prefer to keep the "
          "private one independently rotatable.")


if __name__ == "__main__":
    main()
