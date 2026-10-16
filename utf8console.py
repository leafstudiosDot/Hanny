"""Make printing non-ASCII text safe on a Windows console. Import and call enable().

Windows consoles default to a legacy codepage (cp932 on this machine, cp1252 in
much of the West, cp936 in Chinese) depending on the system locale. Those
encodings cannot represent box-drawing characters, '+/-', or U+FFFD. When
stdout cannot encode a character, Python raises UnicodeEncodeError -- and it
raises it at *flush* time, which means the script can run to completion, save
every file, and still exit with code 1 at the very last print. That failure
looks like a crash in code that actually succeeded.

This lives in its own module because three entry points now print non-ASCII
(model.py, baseline.py, and whatever comes next) and copying a try/except into
each one is how the bug comes back.
"""
import sys


def enable() -> bool:
    """Switch stdout/stderr to UTF-8 with replacement. Returns True on success."""
    ok = False
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # Python < 3.7, or the stream was replaced/redirected
        try:
            reconfigure(encoding="utf-8", errors="replace")
            ok = True
        except (ValueError, OSError):
            pass
    return ok


if __name__ == "__main__":
    print("utf8 enabled:", enable(), "| stdout:", sys.stdout.encoding)