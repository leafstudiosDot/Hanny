"""Hanny data loader: collects source code from your folders for training.

It skips dependency and build folders, minified or generated files, duplicates, and any file
that looks like it contains a secret (API keys, tokens, private keys, hardcoded passwords).
A file with a possible secret is skipped ENTIRELY, so it never reaches the model.

The secret check is best-effort pattern matching, not a guarantee. Before publishing any model
trained on private code, also scan the source with a dedicated tool such as gitleaks or
trufflehog, and rotate any key that was ever committed.

Check what it would do, without training:   python data.py
"""
import os
import random
import re
import sys
import zlib
from collections import Counter

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


def load_dirs(path):
    """Read folders from a text file: one per line, blank lines and # comments ignored."""
    dirs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip().strip('"').strip("'")
            if line:
                dirs.append(os.path.expanduser(line))
    return dirs


def collect_corpus(dirs, max_chars=30_000_000, seed=1337):
    """Return (train_text, val_text, report). Validation = whole held-out files (about 1 in 10)."""
    report = {"skipped": Counter(), "secret_files": [], "kept_ext": Counter(),
              "train_files": 0, "val_files": 0, "missing_dirs": []}
    candidates = []
    for root in dirs:
        if not os.path.isdir(root):
            report["missing_dirs"].append(root)
            continue
        for cur, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d.lower() not in SKIP_DIRS]
            for name in files:
                if wanted(name):
                    candidates.append(os.path.join(cur, name))
    random.Random(seed).shuffle(candidates)   # so the char cap doesn't favor one folder

    seen, train_parts, val_parts, total = set(), [], [], 0
    for path in candidates:
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
        is_val = zlib.crc32(os.path.abspath(path).lower().encode("utf-8", "surrogatepass")) % 10 == 0
        (val_parts if is_val else train_parts).append(text)
        report["val_files" if is_val else "train_files"] += 1
        total += len(text)

    train_text, val_text = "\n\n".join(train_parts), "\n\n".join(val_parts)
    report["train_chars"], report["val_chars"] = len(train_text), len(val_text)
    return train_text, val_text, report


def format_report(report, max_secret_lines=15):
    lines = [f"train: {report['train_files']:,} files, {report['train_chars']:,} characters",
             f"val:   {report['val_files']:,} files, {report['val_chars']:,} characters (whole files held out)"]
    if report["kept_ext"]:
        lines.append("kept by type: " + ", ".join(f"{k} {v}" for k, v in report["kept_ext"].most_common()))
    if report["skipped"]:
        lines.append("skipped: " + ", ".join(f"{k} {v}" for k, v in report["skipped"].most_common()))
    for d in report["missing_dirs"]:
        lines.append(f"WARNING: folder not found: {d}")
    secrets = report["secret_files"]
    if secrets:
        lines.append(f"files skipped for a possible secret ({len(secrets)}), showing {min(len(secrets), max_secret_lines)}:")
        lines += [f"  {p}  [{label}]" for p, label in secrets[:max_secret_lines]]
    return "\n".join(lines)


def full_report_text(report):
    """Like format_report but lists every secret-skipped file (paths and reasons only)."""
    return format_report(report, max_secret_lines=len(report["secret_files"]) or 1)


def load_or_explain(dirs_file):
    if not os.path.exists(dirs_file):
        sys.exit(f"Missing {dirs_file}. Create it with one folder per line, for example:\n"
                 r"  C:\Users\you\projects\my-app" + "\n" + r"  D:\old-projects")
    dirs = load_dirs(dirs_file)
    if not dirs:
        sys.exit(f"{dirs_file} has no folders yet. Add one folder per line (lines starting with # are ignored).")
    return dirs


if __name__ == "__main__":
    dirs_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_dirs.txt")
    train_text, val_text, report = collect_corpus(load_or_explain(dirs_file))
    print(format_report(report))