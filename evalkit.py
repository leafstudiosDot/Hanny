"""One fixed evaluation method, shared by every script that scores a model.

M2 established that a loss number is only trustworthy if every run is scored on
exactly the same windows. That guarantee is easy to state and easy to break by
accident: two files that each build "evenly spaced windows" will drift the
moment one of them is edited.

So there is exactly one implementation of the method here, and baseline.py and
finetune.py both call it. If you want to change how evaluation works, change it
in this file -- never in a caller.

The method:
  - tokenize the held-out split once
  - choose window start positions evenly across the whole split (np.linspace),
    which is deterministic and needs no seed
  - always score exactly those windows
  - report mean +- standard error over windows, so you can see the noise floor
"""
import numpy as np
import torch

MIN_WINDOWS = 2  # a standard error needs at least two samples; ddof=1 is NaN at n=1


def fixed_windows(ids, n_windows, seq_len):
    """Build the fixed evaluation windows. Returns an int64 array (n, seq_len).

    Identical inputs must give identical windows -- that is the whole point.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < seq_len + 1:
        raise ValueError(f"only {len(ids)} tokens available, need seq_len {seq_len}")
    if n_windows < MIN_WINDOWS:
        raise ValueError(f"n_windows must be >= {MIN_WINDOWS}, got {n_windows}")
    n = min(n_windows, len(ids) - seq_len - 1)
    starts = np.linspace(0, len(ids) - seq_len - 1, n).astype(np.int64)
    return np.stack([ids[s:s + seq_len] for s in starts])


def score(model, windows, batch, device):
    """Mean loss over the fixed windows plus the standard error of that mean.

    Every window holds exactly seq_len tokens, so the batch mean IS the mean of
    its window means -- replicating it across the batch is exact, not a fudge.
    """
    per_window = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            xb = torch.as_tensor(windows[i:i + batch], dtype=torch.long, device=device)
            m = model(xb, labels=xb).loss.item()
            per_window.extend([m] * len(xb))
    a = np.asarray(per_window, dtype=np.float64)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")
    return mean, se


def window_geometry(n_tokens, n_windows, seq_len):
    """Describe how the fixed windows sit inside the split.

    Returns (coverage, overlap, spacing):
      coverage  - fraction of the split actually scored (0.0-1.0). Anything
                  well under 1.0 means the rest of the split is unseen.
      overlap   - fraction of each window that covers the previous one. 0.0
                  means the windows are disjoint, so they do not double-count.
      spacing   - tokens between consecutive window starts.
    """
    n = min(n_windows, n_tokens - seq_len - 1)
    if n < 2:
        return min(1.0, (n * seq_len) / n_tokens), 0.0, float(seq_len)
    spacing = (n_tokens - seq_len - 1) / (n - 1)
    overlap = max(0.0, min(1.0, (seq_len - spacing) / seq_len))
    coverage = min(1.0, (n * seq_len) / n_tokens)
    return coverage, overlap, spacing