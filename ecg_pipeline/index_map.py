from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

Interval = Tuple[int, int]

def normalize_intervals(intervals: List[Interval], length: int | None = None) -> List[Interval]:
    """Sort, clamp (if length given), and merge overlapping intervals."""
    if not intervals:
        return []
    xs = []
    for a, b in intervals:
        s, e = (int(a), int(b))
        if s > e:
            s, e = e, s
        xs.append((s, e))
    xs.sort(key=lambda x: x[0])
    # clamp
    if length is not None:
        xs = [(max(0, s), min(length-1, e)) for (s, e) in xs if s < length and e >= 0]
        if not xs:
            return []
    # merge
    merged = []
    cs, ce = xs[0]
    for s, e in xs[1:]:
        if s <= ce + 1:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def merge_intervals(intervals: List[Interval], merge_gap: int = 0) -> List[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce + merge_gap + 1:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out

def clamp_intervals(intervals: List[Interval], length: int) -> List[Interval]:
    return normalize_intervals(intervals, length)

def invert_to_kept(removed: List[Interval], length: int) -> List[Interval]:
    """From removed intervals build kept intervals covering [0, length-1] not in removed."""
    removed = normalize_intervals(removed, length)
    if not removed:
        return [(0, length-1)]
    kept = []
    cur = 0
    for s, e in removed:
        if cur < s:
            kept.append((cur, s-1))
        cur = e + 1
    if cur < length:
        kept.append((cur, length-1))
    return kept

@dataclass
class IndexMap:
    """Mapping between input and output after removing intervals.
    Stores kept spans: tuples of (in_start, in_end, out_start, out_end).
    """
    kept: List[Tuple[int, int, int, int]]

    @staticmethod
    def from_kept_ranges(kept_in: List[Interval]) -> "IndexMap":
        kept_out = []
        out_pos = 0
        for (s, e) in kept_in:
            L = e - s + 1
            kept_out.append((s, e, out_pos, out_pos + L - 1))
            out_pos += L
        return IndexMap(kept_out)

    @staticmethod
    def from_removed_ranges(removed_in: List[Interval], in_len: int) -> "IndexMap":
        kept_in = invert_to_kept(removed_in, in_len)
        return IndexMap.from_kept_ranges(kept_in)

    def project_intervals_input_to_output(self, intervals_in: List[Interval]) -> List[Interval]:
        """Map intervals defined on input indices to output indices."""
        if not intervals_in:
            return []
        res = []
        for (s, e) in intervals_in:
            cur = s
            while cur <= e:
                for (si, ei, so, eo) in self.kept:
                    if si <= cur <= ei:
                        # map this chunk within [si, ei]
                        off = cur - si
                        chunk_end = min(e, ei)
                        out_s = so + off
                        out_e = so + (chunk_end - si)
                        res.append((out_s, out_e))
                        cur = chunk_end + 1
                        break
                else:
                    # current point lies in removed region -> skip to next kept
                    next_kept = None
                    for (si2, ei2, so2, eo2) in self.kept:
                        if si2 > cur:
                            next_kept = si2
                            break
                    if next_kept is None:
                        cur = e + 1
                    else:
                        cur = next_kept
        return normalize_intervals(res)

    def project_intervals_output_to_input(self, intervals_out: List[Interval]) -> List[Interval]:
        """Map intervals defined on output indices back to input indices."""
        if not intervals_out:
            return []
        res = []
        for (s, e) in intervals_out:
            cur = s
            while cur <= e:
                for (si, ei, so, eo) in self.kept:
                    if so <= cur <= eo:
                        off = cur - so
                        chunk_end = min(e, eo)
                        in_s = si + off
                        in_e = si + (chunk_end - so)
                        res.append((in_s, in_e))
                        cur = chunk_end + 1
                        break
                else:
                    # out of any kept region (shouldn't happen)
                    cur = e + 1
        return normalize_intervals(res)

    def compose(self, other: "IndexMap") -> "IndexMap":
        """Return a map equivalent to applying self then other.
        self: A -> B; other: B -> C; result: A -> C
        """
        composed_kept = []
        for (si, ei, so, eo) in self.kept:
            # Map this kept span through 'other' by projecting its output span [so, eo] to C, then back to input A with si as base
            # We'll sample endpoints and rebuild a contiguous mapping where possible.
            mapped = other.project_intervals_output_to_input([(so, eo)])
            # 'mapped' is on B's input => actually B's input == A's output; we need direct A->C map.
            # Alternative robust approach: Create mapping by lengths.
            # Project [so, eo] to C directly:
            to_c = other.project_intervals_input_to_output([(so, eo)])
            off = 0
            for (cs, ce) in to_c:
                L = ce - cs + 1
                # input span for this chunk in A:
                in_s = si + off
                in_e = in_s + L - 1
                composed_kept.append((in_s, in_e, cs, ce))
                off += L
        # normalize composed_kept by merging adjacent ones if contiguous
        composed_kept.sort(key=lambda t: (t[0], t[2]))
        merged = []
        for seg in composed_kept:
            if not merged:
                merged.append(seg)
            else:
                (pi_s, pi_e, po_s, po_e) = merged[-1]
                (ci_s, ci_e, co_s, co_e) = seg
                if pi_e + 1 == ci_s and po_e + 1 == co_s:
                    merged[-1] = (pi_s, ci_e, po_s, co_e)
                else:
                    merged.append(seg)
        return IndexMap(merged)