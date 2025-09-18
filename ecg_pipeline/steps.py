from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .index_map import Interval, normalize_intervals, invert_to_kept, IndexMap, clamp_intervals, merge_intervals

# -----------------------------
# Core removal + map builder
# -----------------------------
def remove_intervals_and_build_map(x: np.ndarray, removed: List[Interval]) -> Tuple[np.ndarray, IndexMap]:
    n = len(x)
    removed = normalize_intervals(removed, n)
    kept = invert_to_kept(removed, n)
    if not kept:
        return np.array([], dtype=x.dtype), IndexMap.from_kept_ranges([])
    parts = [x[s:e+1] for (s, e) in kept]
    y = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()
    idx_map = IndexMap.from_kept_ranges(kept)
    return y, idx_map

# -----------------------------
# Step 1: Remove gaps
# -----------------------------
def remove_gaps(x: np.ndarray, gaps: List[Interval]) -> Tuple[np.ndarray, IndexMap]:
    """Remove gaps from x and return (signal_without_gaps, IndexMap)."""
    if not gaps:
        # Identity map
        kept = [(0, len(x)-1)] if len(x) > 0 else []
        return x.copy(), IndexMap.from_kept_ranges(kept)
    return remove_intervals_and_build_map(x, gaps)

# -----------------------------
# Step 2: Filter
# -----------------------------
def filter_ecg(x: np.ndarray, fs: int, method: str, ftype: str, lowcut: float | None, highcut: float | None, order: int) -> np.ndarray:
    """Filter with NeuroKit2 if requested/available, else SciPy Butterworth. Length preserved."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return x.copy()

    if method == "neurokit":
        try:
            import neurokit2 as nk  # type: ignore
            # neurokit2.ecg_clean supports methods like 'biosppy', 'neurokit', 'pantompkins1985'
            # but we want a band/high-pass butter-like. We'll fall back to nk.signal_filter with parameters.
            if ftype == "highpass":
                y = nk.signal_filter(x, sampling_rate=fs, highcut=None, lowcut=lowcut, method="butterworth", order=order)
            else:
                y = nk.signal_filter(x, sampling_rate=fs, highcut=highcut, lowcut=lowcut, method="butterworth", order=order)
            return np.asarray(y, dtype=float)
        except Exception:
            method = "butter"  # fallback

    # SciPy fallback
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        if ftype == "highpass":
            if lowcut is None or lowcut <= 0:
                return x.copy()
            Wn = lowcut / nyq
            b, a = butter(order, Wn, btype="highpass", analog=False)
        else:
            # bandpass
            if lowcut is None and highcut is None:
                return x.copy()
            if lowcut is None:
                lowcut = 0.3
            if highcut is None:
                highcut = min(40.0, nyq * 0.9)
            Wn = [lowcut / nyq, highcut / nyq]
            b, a = butter(order, Wn, btype="bandpass", analog=False)
        y = filtfilt(b, a, x)
        return np.asarray(y, dtype=float)
    except Exception:
        # Last resort: identity
        return x.copy()

# -----------------------------
# Step 3: Outliers (mock detection)
# -----------------------------
def detect_outliers(x: np.ndarray, z_thresh: float, min_len: int, merge_gap: int) -> List[Interval]:
    if len(x) == 0:
        return []
    # z-score
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-12
    z = np.abs((x - mu) / sd)
    hits = np.where(z > z_thresh)[0].tolist()
    if not hits:
        return []
    # expand each hit to [i-min_len//2, i+min_len//2]
    half = max(1, min_len // 2)
    spans = [(max(0, i - half), min(len(x) - 1, i + half)) for i in hits]
    spans = merge_intervals(spans, merge_gap=merge_gap)
    return spans

# -----------------------------
# Step 4: R-dropouts (mock detection)
# -----------------------------
def detect_rdropouts(x: np.ndarray, win: int, var_thresh: float, merge_gap: int) -> List[Interval]:
    if len(x) == 0:
        return []
    win = max(2, int(win))
    # rolling variance (simple)
    kernel = np.ones(win) / win
    x2 = x * x
    mean = np.convolve(x, kernel, mode="same")
    mean2 = np.convolve(x2, kernel, mode="same")
    var = mean2 - mean * mean
    hits = np.where(var < var_thresh)[0].tolist()
    if not hits:
        return []
    spans = []
    start = hits[0]
    prev = hits[0]
    for i in hits[1:]:
        if i == prev + 1:
            prev = i
        else:
            spans.append((max(0, start - win//2), min(len(x)-1, prev + win//2)))
            start = prev = i
    spans.append((max(0, start - win//2), min(len(x)-1, prev + win//2)))
    spans = merge_intervals(spans, merge_gap=merge_gap)
    return spans

# -----------------------------
# Step 5: Motions (mock detection)
# -----------------------------
def detect_motions(x: np.ndarray, win: int, std_thresh: float, merge_gap: int) -> List[Interval]:
    if len(x) == 0:
        return []
    win = max(2, int(win))
    kernel = np.ones(win) / win
    # robust std via rolling absolute deviation approx
    mean = np.convolve(x, kernel, mode="same")
    dev = np.abs(x - mean)
    rstd = np.convolve(dev, kernel, mode="same")  # approx rolling std
    hits = np.where(rstd > std_thresh)[0].tolist()
    if not hits:
        return []
    spans = []
    start = hits[0]
    prev = hits[0]
    for i in hits[1:]:
        if i == prev + 1:
            prev = i
        else:
            spans.append((max(0, start - win//2), min(len(x)-1, prev + win//2)))
            start = prev = i
    spans.append((max(0, start - win//2), min(len(x)-1, prev + win//2)))
    spans = merge_intervals(spans, merge_gap=merge_gap)
    return spans