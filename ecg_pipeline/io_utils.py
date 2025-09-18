from __future__ import annotations
from typing import List, Tuple, Union
import json
import numpy as np
from pathlib import Path

from .index_map import Interval, normalize_intervals

def load_ecg_npy(path: Union[str, Path]) -> np.ndarray:
    x = np.load(str(path))
    return x.astype(float, copy=False)

def parse_intervals_from_lines(lines: List[str]) -> List[Interval]:
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            continue
        parts = ln.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                a = int(float(parts[0]))
                b = int(float(parts[1]))
                if a > b:
                    a, b = b, a
                out.append((a, b))
            except Exception:
                pass
    return out

def load_intervals(path: Union[str, Path], length: int | None = None) -> List[Interval]:
    p = Path(path)
    if p.suffix.lower() in {".json", ".js"}:
        data = json.loads(p.read_text(encoding="utf-8"))
        spans = [(int(a), int(b)) for (a, b) in data]
    else:
        # try plaintext
        spans = parse_intervals_from_lines(p.read_text(encoding="utf-8").splitlines())
    if length is not None:
        spans = normalize_intervals(spans, length)
    return spans

def save_intervals_json(spans: List[Interval], path: Union[str, Path]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([[int(a), int(b)] for (a, b) in spans], f, ensure_ascii=False, indent=2)

def save_numpy_array(x: np.ndarray, path: Union[str, Path]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), x)