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
    "data.py",
    "evalkit.py",
    "chateval.py",
    "utf8console.py",
    "model.py",
    "baseline.py",
    "baseline.json",
    "requirements.txt",
    "README.md",
    "evalsets/chat_probe.jsonl",
    "evalsets/safety_probe_public.jsonl",
    "evalsets/README.md",
]

# Private. The corpus is found by fingerprint, not by name, because the
# fingerprint changes whenever the corpus does and a stale hardcoded name is how
# you train on the wrong data without noticing.
DATA_GLOBS = [
    ("upload", "corpus-*.json"),
    ("evalsets", "safety_probe.jsonl"),
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
    """Resolve DATA_GLOBS to concrete files. Returns [(arcname, abspath)]."""
    import glob
    found = []
    for folder, pattern in DATA_GLOBS:
        hits = sorted(glob.glob(os.path.join(HERE, folder, pattern)))
        if not hits:
            raise SystemExit(
                f"no file matching {folder}/{pattern}\n"
                f"  Run:  .venv\\Scripts\\python.exe data.py --export-plain upload")
        for path in hits:
            arc = os.path.relpath(path, HERE).replace("\\", "/")
            found.append((arc, path))
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


def check_code_files():
    """Every listed shareable file must exist. Missing means the manifest is wrong."""
    missing = [f for f in CODE_FILES if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        raise SystemExit(
            "pack.py lists files that do not exist:\n  "
            + "\n  ".join(missing)
            + "\nEither they were deleted or CODE_FILES is out of date. Fix the "
              "manifest rather than deleting the check -- that is exactly the "
              "bug that dropped finetune.py from the first upload command.")
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
    args = p.parse_args()

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
