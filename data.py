"""Hanny data loader: collects source code from your folders for training.

It skips dependency and build folders, minified or generated files, duplicates, and any file
that looks like it contains a secret (API keys, tokens, private keys, hardcoded passwords).
A file with a possible secret is skipped ENTIRELY, so it never reaches the model.

The secret check is best-effort pattern matching, not a guarantee. Before publishing any model
trained on private code, also scan the source with a dedicated tool such as gitleaks or
trufflehog, and rotate any key that was ever committed.

The result is cached in cache/ next to this file, so the second run starts in seconds. The cache
is rebuilt automatically when your files, folder list or these rules change.

Preview without training:   python data.py            (add --refresh-data to rebuild)
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import zlib
from collections import Counter

DATA_VERSION = 2   # bump when the filtering rules change so old caches are not reused

# What to learn from
CODE_EXTENSIONS = {
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".html", ".css", ".scss",
    ".py", ".java", ".kt", ".cs", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    ".rs", ".go", ".php", ".sh", ".sql", ".yaml", ".yml", ".json",
}  # files named Dockerfile or Dockerfile.* are included too

# What to never read
SKIP_DIRS = {
    "node_modules", ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "dist", "build",
    "out", "bin", "obj", ".next", ".nuxt", ".cache", "target", "vendor", "coverage", ".idea",
    ".vscode", ".gradle", "site-packages", "bower_components", ".terraform", "pods",
    "deriveddata", "$recycle.bin", "system volume information", "windows", "program files",
    "program files (x86)", "appdata",
    # usually other people's code that was copied or cloned into a project
    "emsdk", "third_party", "thirdparty", "3rdparty", "extern", "external", "deps",
}
SKIP_FILE_NAMES = {"package-lock.json", "pnpm-lock.yaml", "npm-shrinkwrap.json", "composer.lock"}
SKIP_FILE_SUFFIXES = (".min.js", ".min.css", ".map", ".bundle.js", ".d.ts")
# Files whose NAME suggests secrets (skipped even if the content looks harmless)
SENSITIVE_NAME = re.compile(
    r"(secret|credential|service[-_]?account|google-services|keystore|"
    r"appsettings\.(prod|production|staging|development|local))", re.I)

MAX_FILE_BYTES = 200_000     # bigger files are usually data dumps or generated code
MAX_LINE_CHARS = 1_000       # a longer line means minified or generated
MIN_FILE_CHARS = 50

# Secret patterns (a match skips the whole file)
SECRET_PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{20,})")),
    ("Slack token", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}")),
    ("Stripe key", re.compile(r"\b[rs]k_(?:live|test)_[0-9A-Za-z]{16,}")),
    ("OpenAI/Anthropic-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("credentials inside a URL",
     re.compile(r"""[A-Za-z][A-Za-z0-9+.\-]*://[^\s/:@'"<>]+:[^\s/@'"<>]+@""")),
    ("hardcoded secret assignment", re.compile(
        r"""(?i)(?:pass(?:word|wd)?|pwd|secret|api[_-]?key|apikey|access[_-]?key|auth[_-]?token|"""
        r"""access[_-]?token|client[_-]?secret|private[_-]?key|token)\w*["']?\s*[:=]\s*["'][^"'\s]{6,}["']""")),
]
# Unquoted values only make sense to check in YAML and Dockerfiles
YAML_UNQUOTED = re.compile(
    r"""(?im)^\s*-?\s*[\w.\-]*(?:password|passwd|secret|api[_-]?key|token|private[_-]?key)[\w.\-]*"""
    r"""\s*:\s*[^\s"'$<\[{(#|>][^\s#]{3,}\s*(?:#.*)?$""")
DOCKER_ENV = re.compile(
    r"""(?im)^\s*(?:ENV|ARG)\s+[\w.\-]*(?:password|passwd|secret|api[_-]?key|token)[\w.\-]*[ \t=]+[^\s$"'][^\s]*""")


def find_secret(text, filename):
    """Return a short label if the text looks like it contains a secret, else None."""
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return label
    low = filename.lower()
    if low.endswith((".yaml", ".yml")) and YAML_UNQUOTED.search(text):
        return "unquoted secret in YAML"
    if low.startswith("dockerfile") and DOCKER_ENV.search(text):
        return "secret in Dockerfile ENV/ARG"
    return None


def wanted(filename):
    low = filename.lower()
    if low in SKIP_FILE_NAMES or low.endswith(SKIP_FILE_SUFFIXES):
        return False
    if low.startswith("dockerfile"):
        return True
    return os.path.splitext(low)[1] in CODE_EXTENSIONS


def ext_key(filename):
    return "Dockerfile" if filename.lower().startswith("dockerfile") else os.path.splitext(filename)[1].lower()


# Folder list
def load_dirs(path):
    """Read data_dirs.txt. Returns (folders_to_include, folders_to_exclude).
    One folder per line. A line starting with ! excludes that folder. # starts a comment."""
    include, exclude = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            is_exclude = line.startswith("!")
            line = line.lstrip("!").strip().strip('"').strip("'")
            if line:
                (exclude if is_exclude else include).append(os.path.expanduser(line))
    return include, exclude


def load_or_explain(dirs_file):
    if not os.path.exists(dirs_file):
        sys.exit(f"Missing {dirs_file}. Create it with one folder per line, for example:\n"
                 r"  C:\Users\you\projects\my-app" + "\n" + r"  !C:\Users\you\projects\my-app\third-party-libs")
    include, exclude = load_dirs(dirs_file)
    if not include:
        sys.exit(f"{dirs_file} has no folders yet. Add one folder per line (lines starting with # are ignored).")
    return include, exclude


def _norm(path):
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def _is_excluded(path, excluded_norm):
    p = _norm(path)
    return any(p == e or p.startswith(e + os.sep) for e in excluded_norm)


def _new_report():
    return {"skipped": Counter(), "secret_files": [], "kept_ext": Counter(), "sources": Counter(),
            "train_files": 0, "val_files": 0, "train_chars": 0, "val_chars": 0, "missing_dirs": []}


def list_candidates(dirs, excludes, report):
    """Walk the folders (without reading files) and return [(root, path), ...]."""
    excluded = [_norm(e) for e in excludes]
    out, seen = [], set()
    for root in dirs:
        if not os.path.isdir(root):
            report["missing_dirs"].append(root)
            continue
        if _is_excluded(root, excluded):
            continue
        for cur, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs
                          if d.lower() not in SKIP_DIRS and not _is_excluded(os.path.join(cur, d), excluded)]
            for name in files:
                if wanted(name):
                    path = os.path.join(cur, name)
                    key = _norm(path)
                    if key not in seen:
                        seen.add(key)
                        out.append((root, path))
    return out


def _source_key(root, path):
    """The first two folder levels under the root: a copy-pasteable path for a ! exclude line."""
    parts = os.path.relpath(os.path.dirname(path), root).split(os.sep)
    parts = [] if parts == ["."] else parts[:2]
    return os.path.join(root, *parts)


def _build(candidates, report, max_chars, seed):
    """Read, filter and split the files. Returns (train_text, val_text)."""
    paths = list(candidates)
    random.Random(seed).shuffle(paths)   # so the character cap doesn't favor one folder
    seen, train_parts, val_parts, total = set(), [], [], 0
    for root, path in paths:
        name = os.path.basename(path)
        if total >= max_chars:
            report["skipped"]["not read (character cap reached)"] += 1
            continue
        if SENSITIVE_NAME.search(name):
            report["skipped"]["sensitive file name"] += 1
            continue
        try:
            if os.path.getsize(path) > MAX_FILE_BYTES:
                report["skipped"]["too large"] += 1
                continue
            with open(path, encoding="utf-8-sig") as f:   # strict: non-UTF-8 files are skipped
                text = f.read()
        except UnicodeDecodeError:
            report["skipped"]["not UTF-8 (binary?)"] += 1
            continue
        except OSError:
            report["skipped"]["unreadable"] += 1
            continue
        if len(text) < MIN_FILE_CHARS or "\x00" in text:
            report["skipped"]["empty or binary"] += 1
            continue
        if max(len(line) for line in text.splitlines()) > MAX_LINE_CHARS:
            report["skipped"]["minified or generated"] += 1
            continue
        label = find_secret(text, name)
        if label:
            report["skipped"]["possible secret"] += 1
            report["secret_files"].append((path, label))   # path and reason only, never the content
            continue
        if text in seen:
            report["skipped"]["duplicate content"] += 1
            continue
        seen.add(text)
        report["kept_ext"][ext_key(name)] += 1
        report["sources"][_source_key(root, path)] += len(text)
        is_val = zlib.crc32(os.path.abspath(path).lower().encode("utf-8", "surrogatepass")) % 10 == 0
        (val_parts if is_val else train_parts).append(text)
        report["val_files" if is_val else "train_files"] += 1
        total += len(text)
    train_text, val_text = "\n\n".join(train_parts), "\n\n".join(val_parts)
    report["train_chars"], report["val_chars"] = len(train_text), len(val_text)
    return train_text, val_text


# Cache
def _fingerprint(candidates):
    """Hash of every candidate file's path, size and modified time (no file contents read)."""
    h = hashlib.sha1()
    for root, path in sorted(candidates):
        try:
            st = os.stat(path)
        except OSError:
            continue
        h.update(f"{root}|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8", "surrogatepass"))
    return h.hexdigest()


def _cache_key(candidates, excludes, max_chars, seed):
    settings = json.dumps({
        "version": DATA_VERSION, "max_chars": max_chars, "seed": seed,
        "extensions": sorted(CODE_EXTENSIONS), "skip_dirs": sorted(SKIP_DIRS),
        "max_file_bytes": MAX_FILE_BYTES, "excludes": sorted(_norm(e) for e in excludes),
    }, sort_keys=True)
    h = hashlib.sha1(settings.encode("utf-8"))
    h.update(_fingerprint(candidates).encode("ascii"))
    return h.hexdigest()[:16]


def _report_to_json(report):
    out = dict(report)
    for k in ("skipped", "kept_ext", "sources"):
        out[k] = dict(report[k])
    out["secret_files"] = [list(x) for x in report["secret_files"]]
    return out


def _report_from_json(data):
    report = _new_report()
    report.update(data)
    for k in ("skipped", "kept_ext", "sources"):
        report[k] = Counter(data[k])
    report["secret_files"] = [tuple(x) for x in data["secret_files"]]
    return report


def _save_cache(path, train_text, val_text, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as f:
        json.dump({"train": train_text, "val": val_text, "report": _report_to_json(report)}, f, ensure_ascii=False)
    os.replace(tmp, path)
    for name in os.listdir(os.path.dirname(path)):   # keep only the newest cache
        full = os.path.join(os.path.dirname(path), name)
        if name.startswith("corpus-") and name.endswith(".json.gz") and full != path:
            os.remove(full)


def get_corpus(dirs, excludes=(), max_chars=30_000_000, seed=1337, cache_dir=None, refresh=False):
    """Return (train_text, val_text, report). Uses the cache when nothing has changed."""
    report = _new_report()
    candidates = list_candidates(dirs, excludes, report)
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"corpus-{_cache_key(candidates, excludes, max_chars, seed)}.json.gz")
        if os.path.exists(cache_path) and not refresh:
            try:
                with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                cached = _report_from_json(data["report"])
                cached["missing_dirs"] = report["missing_dirs"]
                print("loaded the corpus from cache (add --refresh-data to rebuild it)")
                return data["train"], data["val"], cached
            except (OSError, ValueError, KeyError):
                print("cache could not be read, rebuilding it")
    print(f"reading {len(candidates):,} candidate files (first run, or something changed)...")
    train_text, val_text = _build(candidates, report, max_chars, seed)
    if cache_path:
        _save_cache(cache_path, train_text, val_text, report)
    return train_text, val_text, report


# Reporting
def format_report(report, max_secret_lines=15, max_sources=10):
    lines = [f"train: {report['train_files']:,} files, {report['train_chars']:,} characters",
             f"val:   {report['val_files']:,} files, {report['val_chars']:,} characters (whole files held out)"]
    if report["kept_ext"]:
        lines.append("kept by type: " + ", ".join(f"{k} {v}" for k, v in report["kept_ext"].most_common()))
    if report["skipped"]:
        lines.append("skipped: " + ", ".join(f"{k} {v}" for k, v in report["skipped"].most_common()))
    total = sum(report["sources"].values())
    if total:
        lines.append("biggest sources (to leave one out, add a line to data_dirs.txt starting with !):")
        for src, n in report["sources"].most_common(max_sources):
            lines.append(f"  {n / total:5.1%}  {src}")
    for d in report["missing_dirs"]:
        lines.append(f"WARNING: folder not found: {d}")
    secrets = report["secret_files"]
    if secrets:
        lines.append(f"files skipped for a possible secret ({len(secrets)}), showing {min(len(secrets), max_secret_lines)}:")
        lines += [f"  {p}  [{label}]" for p, label in secrets[:max_secret_lines]]
    return "\n".join(lines)


def full_report_text(report):
    """Like format_report but lists every source and every secret-skipped file (paths only)."""
    return format_report(report, max_secret_lines=len(report["secret_files"]) or 1,
                         max_sources=len(report["sources"]) or 1)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs-file", default=os.path.join(here, "data_dirs.txt"))
    ap.add_argument("--max-chars", type=int, default=30_000_000)
    ap.add_argument("--refresh-data", action="store_true")
    a = ap.parse_args()
    include, exclude = load_or_explain(a.dirs_file)
    _, _, rep = get_corpus(include, exclude, a.max_chars, cache_dir=os.path.join(here, "cache"), refresh=a.refresh_data)
    print(format_report(rep))