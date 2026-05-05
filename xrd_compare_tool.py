"""XRD comparison utilities: parsing, normalization, peak detection, matching."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------- parsing ----------

_TWOTHETA_HINTS = ("2theta", "two_theta", "twotheta", "2θ", "angle", "deg")
_INTENSITY_HINTS = ("intensity", "counts", "int", "i_obs", "y")


def _pick_column(columns: Iterable[str], hints: tuple[str, ...]) -> str | None:
    norm = {c: re.sub(r"[^a-z0-9]", "", c.lower()) for c in columns}
    for c, n in norm.items():
        for h in hints:
            if re.sub(r"[^a-z0-9]", "", h) in n:
                return c
    return None


def load_xrd_csv(file_or_buffer, name: str = "pattern") -> "XRDPattern":
    """Load a Fiji-exported CSV and auto-detect 2theta/intensity columns."""
    # Try common separators
    raw = file_or_buffer.read() if hasattr(file_or_buffer, "read") else open(file_or_buffer, "rb").read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw

    df = None
    for sep in (",", "\t", ";", r"\s+"):
        try:
            cand = pd.read_csv(io.StringIO(text), sep=sep, engine="python", comment="#")
            if cand.shape[1] >= 2:
                df = cand
                break
        except Exception:
            continue
    if df is None or df.shape[1] < 2:
        raise ValueError(f"Could not parse {name}: need at least 2 columns")

    tt_col = _pick_column(df.columns, _TWOTHETA_HINTS)
    in_col = _pick_column(df.columns, _INTENSITY_HINTS)

    # Fall back to first two numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if tt_col is None and numeric_cols:
        tt_col = numeric_cols[0]
    if in_col is None and len(numeric_cols) > 1:
        in_col = numeric_cols[1] if numeric_cols[1] != tt_col else (numeric_cols[2] if len(numeric_cols) > 2 else None)

    if tt_col is None or in_col is None:
        raise ValueError(f"Could not detect 2theta/intensity columns in {name}; found {list(df.columns)}")

    sub = df[[tt_col, in_col]].dropna()
    sub = sub.apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub.sort_values(tt_col).reset_index(drop=True)

    return XRDPattern(
        name=name,
        two_theta=sub[tt_col].to_numpy(dtype=float),
        intensity=sub[in_col].to_numpy(dtype=float),
        source_columns=(tt_col, in_col),
    )


# ---------- pattern container ----------

@dataclass
class XRDPattern:
    name: str
    two_theta: np.ndarray
    intensity: np.ndarray
    source_columns: tuple[str, str] = ("", "")
    peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def normalized(self) -> np.ndarray:
        y = self.intensity.astype(float)
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-12:
            return np.zeros_like(y)
        return (y - lo) / (hi - lo)


# ---------- peak detection ----------

def detect_peaks(pattern: XRDPattern, prominence: float = 0.05, window: int = 5) -> np.ndarray:
    """Return indices of detected peaks on the normalized intensity."""
    y = pattern.normalized()
    distance = max(1, int(window))
    idx, _ = find_peaks(y, prominence=prominence, distance=distance)
    pattern.peaks = idx
    return idx


# ---------- comparison ----------

@dataclass
class CompareResult:
    matched: list[tuple[float, float]]   # (sample_2theta, ref_2theta)
    new_in_sample: list[float]
    missing_from_sample: list[float]
    match_score: float
    novelty_score: float
    reference_name: str


def compare_patterns(
    sample: XRDPattern,
    reference: XRDPattern,
    tolerance_deg: float = 0.2,
    prominence: float = 0.05,
    window: int = 5,
) -> CompareResult:
    s_idx = detect_peaks(sample, prominence=prominence, window=window)
    r_idx = detect_peaks(reference, prominence=prominence, window=window)

    s_pos = sample.two_theta[s_idx]
    r_pos = reference.two_theta[r_idx]

    used_ref: set[int] = set()
    matched: list[tuple[float, float]] = []
    new_peaks: list[float] = []

    for sp in s_pos:
        if len(r_pos) == 0:
            new_peaks.append(float(sp))
            continue
        diffs = np.abs(r_pos - sp)
        # find closest unused ref
        order = np.argsort(diffs)
        chosen = None
        for j in order:
            if j in used_ref:
                continue
            if diffs[j] <= tolerance_deg:
                chosen = int(j)
            break
        if chosen is not None:
            used_ref.add(chosen)
            matched.append((float(sp), float(r_pos[chosen])))
        else:
            new_peaks.append(float(sp))

    missing = [float(r_pos[j]) for j in range(len(r_pos)) if j not in used_ref]

    n_ref = max(1, len(r_pos))
    n_sample = max(1, len(s_pos))
    match_score = len(matched) / n_ref
    novelty_score = len(new_peaks) / n_sample

    return CompareResult(
        matched=matched,
        new_in_sample=new_peaks,
        missing_from_sample=missing,
        match_score=float(match_score),
        novelty_score=float(novelty_score),
        reference_name=reference.name,
    )


def comparison_to_dataframe(result: CompareResult) -> pd.DataFrame:
    rows = []
    for sp, rp in result.matched:
        rows.append({"category": "matched", "sample_2theta": sp, "reference_2theta": rp, "delta": sp - rp})
    for sp in result.new_in_sample:
        rows.append({"category": "new_in_sample", "sample_2theta": sp, "reference_2theta": None, "delta": None})
    for rp in result.missing_from_sample:
        rows.append({"category": "missing_from_sample", "sample_2theta": None, "reference_2theta": rp, "delta": None})
    return pd.DataFrame(rows)
