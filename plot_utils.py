"""Publication-style matplotlib plots + PDF report builder for XRD patterns."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from xrd_compare_tool import CompareResult, XRDPattern, detect_peaks

# ---- scientific-ish rcParams ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 1.1,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.dpi": 180,
})


def _style(ax, title: str = "") -> None:
    ax.set_xlabel(r"2$\theta$ (deg)")
    ax.set_ylabel("Intensity (counts)")
    ax.grid(True, which="major", ls=":", alpha=0.35)
    if title:
        ax.set_title(title)


def plot_pattern_mpl(pattern: XRDPattern, peak_idx=None, title: str | None = None,
                     annotate_peaks: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pattern.two_theta, pattern.intensity,
            color="#1f3b73", lw=1.2, label=pattern.name)
    if peak_idx is not None and len(peak_idx):
        ax.plot(pattern.two_theta[peak_idx], pattern.intensity[peak_idx],
                "x", color="#c0392b", ms=7, mew=1.3, label="peaks")
        if annotate_peaks:
            ymax = float(np.max(pattern.intensity))
            for i in peak_idx:
                ax.annotate(f"{pattern.two_theta[i]:.2f}",
                            (pattern.two_theta[i], pattern.intensity[i]),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=6.5, rotation=90,
                            color="#555")
            ax.set_ylim(top=ymax * 1.18)
    _style(ax, title or pattern.name)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _norm01(y: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi - lo < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def plot_pair_overlay_mpl(
    sample: XRDPattern, ref: XRDPattern, res: CompareResult, title: str = "",
) -> plt.Figure:
    """One sample vs one reference, true overlay (normalized 0-1)."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(ref.two_theta, _norm01(ref.intensity),
            color="#1f3b73", lw=1.1, alpha=0.85, label=f"ref: {ref.name}")
    ax.plot(sample.two_theta, _norm01(sample.intensity),
            color="#b71c1c", lw=1.4, label=f"sample: {sample.name}", zorder=5)

    if res.matched:
        xs = sorted({m[0] for m in res.matched})
        ax.plot(xs, [1.04] * len(xs), marker="v", ls="", color="#2ca02c",
                ms=7, mec="black", mew=0.4, label="matched")
    if res.missing_from_sample:
        ax.plot(sorted(set(res.missing_from_sample)),
                [1.08] * len(set(res.missing_from_sample)),
                marker="v", ls="", color="#ff7f0e", ms=7, mec="black", mew=0.4,
                label="missing in sample")
    if res.new_in_sample:
        ax.plot(sorted(set(res.new_in_sample)),
                [1.12] * len(set(res.new_in_sample)),
                marker="v", ls="", color="#c0392b", ms=7, mec="black", mew=0.4,
                label="new in sample")
        for sp in res.new_in_sample:
            ax.axvline(sp, color="#c0392b", ls="--", lw=0.5, alpha=0.4, zorder=1)

    ax.set_xlabel(r"2$\theta$ (deg)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_ylim(-0.03, 1.18)
    ax.grid(True, ls=":", alpha=0.35)
    ax.set_title(title or f"{sample.name}  vs  {ref.name}", pad=10)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_stacked_overlay_mpl(
    sample: XRDPattern,
    refs_results: dict[str, tuple[XRDPattern, CompareResult]],
    title: str = "",
) -> plt.Figure:
    """Offset-stacked: each reference on its own baseline, sample on top."""
    n = len(refs_results)
    fig, ax = plt.subplots(figsize=(11, 4.5 + 1.0 * n))
    step = 1.20
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, n)))

    offset = 0.0
    for (name, (ref, res)), c in zip(refs_results.items(), colors):
        y = _norm01(ref.intensity) + offset
        ax.plot(ref.two_theta, y, color=c, lw=1.0, label=f"ref: {name}")
        ax.text(ref.two_theta.min(), offset + 1.02, name,
                fontsize=8, color=c, va="bottom")
        if res.matched:
            xs = [m[1] for m in res.matched]
            ax.plot(xs, [offset + 1.04] * len(xs), marker="v", ls="",
                    color="#2ca02c", ms=5, mec="black", mew=0.3)
        offset += step

    y_sample = _norm01(sample.intensity) + offset
    ax.plot(sample.two_theta, y_sample, color="#b71c1c", lw=1.4,
            label=f"sample: {sample.name}", zorder=5)
    ax.text(sample.two_theta.min(), offset + 1.02, sample.name,
            fontsize=9, color="#b71c1c", va="bottom", weight="bold")

    ax.set_xlabel(r"2$\theta$ (deg)")
    ax.set_ylabel("Intensity (normalized, offset)")
    ax.set_yticks([])
    ax.grid(True, axis="x", ls=":", alpha=0.35)
    ax.set_title(title or f"Stacked: {sample.name} vs references", pad=12)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_overlay_mpl(
    sample: XRDPattern,
    refs_results: dict[str, tuple[XRDPattern, CompareResult]],
    title: str = "",
) -> plt.Figure:
    """True overlay: sample and all references on the same normalized axes."""
    n = len(refs_results)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    colors_ref = plt.cm.tab10(np.linspace(0, 1, max(1, n)))

    # References
    for (name, (ref, res)), c in zip(refs_results.items(), colors_ref):
        ax.plot(ref.two_theta, _norm01(ref.intensity),
                color=c, lw=1.0, alpha=0.75, label=f"ref: {name}")

    # Sample on top
    ax.plot(sample.two_theta, _norm01(sample.intensity),
            color="#b71c1c", lw=1.5, label=f"sample: {sample.name}", zorder=5)

    # Peak markers just above y=1
    y_match, y_miss, y_new = 1.04, 1.08, 1.12
    all_matched_sample_x = []
    all_missing_x = []
    for name, (ref, res) in refs_results.items():
        all_matched_sample_x.extend([m[0] for m in res.matched])
        all_missing_x.extend(res.missing_from_sample)

    if all_matched_sample_x:
        ax.plot(sorted(set(all_matched_sample_x)),
                [y_match] * len(set(all_matched_sample_x)),
                marker="v", ls="", color="#2ca02c", ms=7, mec="black", mew=0.4,
                label="matched")
    if all_missing_x:
        ax.plot(sorted(set(all_missing_x)),
                [y_miss] * len(set(all_missing_x)),
                marker="v", ls="", color="#ff7f0e", ms=7, mec="black", mew=0.4,
                label="missing in sample")

    if refs_results:
        new_sets = [set(np.round(r.new_in_sample, 3)) for _, r in refs_results.values()]
        truly_new = sorted(set.intersection(*new_sets)) if new_sets else []
        for sp in truly_new:
            ax.axvline(sp, color="#c0392b", ls="--", lw=0.6, alpha=0.45, zorder=1)
        if truly_new:
            ax.plot(truly_new, [y_new] * len(truly_new),
                    marker="v", ls="", color="#c0392b", ms=7, mec="black", mew=0.4,
                    label="new in sample")

    ax.set_xlabel(r"2$\theta$ (deg)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_ylim(-0.03, 1.18)
    ax.grid(True, which="major", ls=":", alpha=0.35)
    ax.set_title(title or f"Sample {sample.name} vs references", pad=12)

    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    return buf.getvalue()


def build_pdf_report(
    pdf_path: str | Path,
    samples: dict[str, XRDPattern],
    references: dict[str, XRDPattern],
    results: dict[str, dict[str, CompareResult]],
    raw_images: dict[str, str] | None = None,
    params: dict | None = None,
) -> Path:
    pdf_path = Path(pdf_path)
    params = params or {}
    raw_images = raw_images or {}

    with PdfPages(pdf_path) as pdf:
        # ---------- Cover ----------
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.text(0.5, 0.92, "XRD Comparison Report", ha="center",
                 fontsize=22, weight="bold")
        fig.text(0.5, 0.88, f"{len(samples)} sample(s) · {len(references)} reference(s)",
                 ha="center", fontsize=12, color="#555")
        info = [
            f"Peak prominence : {params.get('prominence', '—')}",
            f"Window (pts)    : {params.get('window', '—')}",
            f"Match tolerance : {params.get('tolerance', '—')}°",
        ]
        fig.text(0.1, 0.82, "\n".join(info), fontsize=10, family="monospace")

        rows = []
        for sname, per_ref in results.items():
            for rname, res in per_ref.items():
                rows.append([
                    sname[:22], rname[:22],
                    f"{res.match_score:.2f}", f"{res.novelty_score:.2f}",
                    str(len(res.matched)), str(len(res.new_in_sample)),
                    str(len(res.missing_from_sample)),
                ])
        if rows:
            ax = fig.add_axes([0.08, 0.15, 0.84, 0.6])
            ax.axis("off")
            table = ax.table(
                cellText=rows,
                colLabels=["sample", "reference", "match", "novelty",
                           "matched", "new", "missing"],
                loc="upper center", cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.4)
            ax.set_title("Score summary", pad=10)
        pdf.savefig(fig); plt.close(fig)

        # ---------- Individual digitized patterns ----------
        all_patterns = {**{f"sample · {k}": v for k, v in samples.items()},
                        **{f"ref · {k}": v for k, v in references.items()}}
        for label, pat in all_patterns.items():
            idx = pat.peaks if len(pat.peaks) else None
            fig = plot_pattern_mpl(pat, idx, title=label)
            pdf.savefig(fig); plt.close(fig)

        # ---------- Per-sample overlays + per-pair + stacked + raw images ----------
        for sname, per_ref in results.items():
            sample = samples[sname]
            refs_res = {rn: (references[rn], r) for rn, r in per_ref.items()}

            # Combined overlay (all refs on one axis)
            fig = plot_overlay_mpl(sample, refs_res,
                                   title=f"Combined overlay — {sname}")
            pdf.savefig(fig); plt.close(fig)

            # Stacked overlay
            fig = plot_stacked_overlay_mpl(sample, refs_res,
                                           title=f"Stacked overlay — {sname}")
            pdf.savefig(fig); plt.close(fig)

            # Per-pair overlays
            for rname, res in per_ref.items():
                fig = plot_pair_overlay_mpl(
                    sample, references[rname], res,
                    title=f"{sname}  vs  {rname}",
                )
                pdf.savefig(fig); plt.close(fig)

            # Raw reference images (one per page)
            for rname in per_ref.keys():
                key_candidates = [rname, Path(rname).stem,
                                  Path(rname).stem.replace("_extracted", "")]
                img_path = None
                for k in key_candidates:
                    if k in raw_images:
                        img_path = raw_images[k]
                        break
                if img_path is None:
                    continue
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    fig, ax = plt.subplots(figsize=(8.27, 9))
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(f"Raw reference image — {rname}", pad=10)
                    pdf.savefig(fig); plt.close(fig)
                except Exception:
                    continue

    return pdf_path
