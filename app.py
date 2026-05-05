"""Streamlit XRD comparison app — 5 pages."""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from xrd_compare_tool import (
    XRDPattern,
    compare_patterns,
    comparison_to_dataframe,
    detect_peaks,
    load_xrd_csv,
)
from fiji_runner import find_fiji, launch_fiji_with_macro
from plot_utils import (
    build_pdf_report,
    fig_to_png_bytes,
    plot_overlay_mpl,
    plot_pair_overlay_mpl,
    plot_pattern_mpl,
    plot_stacked_overlay_mpl,
)

st.set_page_config(page_title="XRD Compare", layout="wide")

# ---------- session state ----------
ss = st.session_state
ss.setdefault("samples", {})             # name -> XRDPattern
ss.setdefault("references", {})          # name -> XRDPattern
ss.setdefault("raw_ref_images", {})      # stem -> temp path (for PDF report)
ss.setdefault("params", {"prominence": 0.05, "window": 5, "tolerance": 0.2})
ss.setdefault("fiji_path", find_fiji() or "")
ss.setdefault("extracted_csvs", {})
_DEFAULT_MACRO = str(Path(__file__).parent / "xrd_batch_extractor_v4_fixed (1).ijm")
ss.setdefault("macro_path", _DEFAULT_MACRO)
_DEFAULT_BASE = Path.cwd() / "xrd_csvs"
ss.setdefault("ref_output_dir", str(_DEFAULT_BASE / "references"))
ss.setdefault("sample_output_dir", str(_DEFAULT_BASE / "samples"))
ss.setdefault("results", {})             # sample_name -> {ref_name: CompareResult}


# ---------- plotly (real units, no forced normalization) ----------
def plotly_pattern(pattern: XRDPattern, peak_idx=None, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pattern.two_theta, y=pattern.intensity,
        mode="lines", name=pattern.name, line=dict(width=1.4, color="#1f3b73"),
    ))
    if peak_idx is not None and len(peak_idx):
        fig.add_trace(go.Scatter(
            x=pattern.two_theta[peak_idx], y=pattern.intensity[peak_idx],
            mode="markers", name="peaks",
            marker=dict(size=9, color="#c0392b", symbol="x"),
        ))
    fig.update_layout(
        title=title or pattern.name,
        xaxis_title="2θ (deg)", yaxis_title="Intensity (counts)",
        height=460, margin=dict(l=50, r=20, t=50, b=45),
        template="plotly_white",
    )
    fig.update_xaxes(ticks="inside", mirror=True, showline=True, linecolor="black")
    fig.update_yaxes(ticks="inside", mirror=True, showline=True, linecolor="black")
    return fig


# ==========================================================================
def _load_csvs_from(folder: str) -> dict[str, XRDPattern]:
    d = Path(folder or "")
    if not d.is_dir():
        return {}
    csvs = sorted(d.glob("*_extracted.csv")) or [
        p for p in sorted(d.glob("*.csv")) if "_peaks" not in p.name.lower()
    ]
    out = {}
    for p in csvs:
        try:
            with open(p, "rb") as fh:
                out[p.name] = load_xrd_csv(fh, name=p.name)
        except Exception as e:
            st.error(f"{p.name}: {e}")
    return out


def _csv_uploader_fallback():
    """Cloud fallback: upload CSVs directly instead of running Fiji."""
    st.info(
        "**Fiji is not available in this environment** (cloud deployment). "
        "Extract CSVs locally using Fiji, then upload them below — "
        "or use **1 · Upload** to load them."
    )
    st.divider()

    st.subheader("Step 1 — Upload Reference CSVs")
    ref_files = st.file_uploader(
        "Reference CSVs (`*_extracted.csv`)", type=["csv", "txt", "tsv"],
        accept_multiple_files=True, key="extract_refs_up",
    )
    if ref_files:
        ss.references = {}
        for f in ref_files:
            try:
                ss.references[f.name] = load_xrd_csv(f, name=f.name)
            except Exception as e:
                st.error(f"{f.name}: {e}")
        if ss.references:
            st.success(f"Loaded {len(ss.references)} reference(s).")

    st.divider()
    st.subheader("Step 2 — Upload Sample CSVs")
    sam_files = st.file_uploader(
        "Sample CSVs (`*_extracted.csv`)", type=["csv", "txt", "tsv"],
        accept_multiple_files=True, key="extract_sams_up",
    )
    if sam_files:
        ss.samples = {}
        for f in sam_files:
            try:
                ss.samples[f.name] = load_xrd_csv(f, name=f.name)
            except Exception as e:
                st.error(f"{f.name}: {e}")
        if ss.samples:
            st.success(f"Loaded {len(ss.samples)} sample(s).")

    st.divider()
    st.subheader("Current data loaded")
    st.write(f"**Samples:** {len(ss.samples)}  ·  **References:** {len(ss.references)}")
    if ss.samples and ss.references:
        st.info("Ready — proceed to **2 · Analyze** or **3 · Compare**.")


def page_extract():
    st.header("0 · Extract CSVs from images with Fiji")
    st.caption(
        "Run Fiji **twice** — once for your reference images, once for your sample "
        "images — saving each batch into its own folder. Then load both folders here."
    )

    detected = find_fiji(ss.fiji_path or None)

    if not detected:
        _csv_uploader_fallback()
        return

    st.success(f"Fiji detected: `{detected}`")
    ss.fiji_path = st.text_input("Fiji executable path", value=ss.fiji_path)
    ss.macro_path = st.text_input("Macro (.ijm) path", value=ss.macro_path)

    st.divider()

    # ---------------- REFERENCES ----------------
    st.subheader("Step 1 — References")
    st.caption(
        "Click **Launch Fiji for REFERENCES**, run the macro in Batch mode, and save "
        "the reference CSVs into a dedicated folder (e.g. `…\\xrd_csvs\\references`)."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🚀 Launch Fiji for REFERENCES", type="primary", key="launch_ref"):
            try:
                launch_fiji_with_macro(
                    ss.macro_path,
                    fiji_path=ss.fiji_path or None,
                    output_dir=ss.ref_output_dir,
                )
                st.success(
                    f"Fiji launched. CSVs will be saved automatically to:\n\n`{ss.ref_output_dir}`\n\n"
                    "Use **Batch folder** mode in the dialog. The output folder prompt is skipped."
                )
            except Exception as e:
                st.error(f"Launch failed: {e}")
    with c2:
        ss.ref_output_dir = st.text_input(
            "References output folder", value=ss.ref_output_dir,
            placeholder=r"C:\Users\Admin\Desktop\xrd_csvs\references",
        )
    if st.button("📥 Load reference CSVs"):
        refs = _load_csvs_from(ss.ref_output_dir)
        if refs:
            ss.references = refs
            st.success(f"Loaded {len(refs)} reference(s) from folder.")
        else:
            st.error("No CSVs found in that folder.")

    st.divider()

    # ---------------- SAMPLES ----------------
    st.subheader("Step 2 — Samples")
    st.caption(
        "Now click **Launch Fiji for SAMPLES**, run the macro again, and save the "
        "sample CSVs into a **different** folder (e.g. `…\\xrd_csvs\\samples`)."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🚀 Launch Fiji for SAMPLES", type="primary", key="launch_sam"):
            try:
                launch_fiji_with_macro(
                    ss.macro_path,
                    fiji_path=ss.fiji_path or None,
                    output_dir=ss.sample_output_dir,
                )
                st.success(
                    f"Fiji launched. CSVs will be saved automatically to:\n\n`{ss.sample_output_dir}`\n\n"
                    "Use **Batch folder** mode in the dialog. The output folder prompt is skipped."
                )
            except Exception as e:
                st.error(f"Launch failed: {e}")
    with c2:
        ss.sample_output_dir = st.text_input(
            "Samples output folder", value=ss.sample_output_dir,
            placeholder=r"C:\Users\Admin\Desktop\xrd_csvs\samples",
        )
    if st.button("📥 Load sample CSVs"):
        sams = _load_csvs_from(ss.sample_output_dir)
        if sams:
            ss.samples = sams
            st.success(f"Loaded {len(sams)} sample(s) from folder.")
        else:
            st.error("No CSVs found in that folder.")

    st.divider()

    # ---------------- Status ----------------
    st.subheader("Current data loaded")
    st.write(f"**Samples:** {len(ss.samples)}  ·  **References:** {len(ss.references)}")
    if ss.samples:
        st.write("Samples:", list(ss.samples.keys()))
    if ss.references:
        st.write("References:", list(ss.references.keys()))
    if ss.samples and ss.references:
        st.info("Ready — proceed to **2 · Analyze** or **3 · Compare**.")


# ==========================================================================
def page_upload():
    st.header("1 · Upload CSVs & raw reference images")
    st.caption("Multiple samples and multiple references supported.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Samples (1+)")
        s_files = st.file_uploader(
            "Sample CSVs", type=["csv", "txt", "tsv"],
            accept_multiple_files=True, key="samples_up",
        )
        if s_files:
            ss.samples = {}
            for f in s_files:
                try:
                    pat = load_xrd_csv(f, name=f.name)
                    ss.samples[f.name] = pat
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            st.success(f"Loaded {len(ss.samples)} sample(s).")

    with c2:
        st.subheader("References (1+)")
        r_files = st.file_uploader(
            "Reference CSVs", type=["csv", "txt", "tsv"],
            accept_multiple_files=True, key="refs_up",
        )
        if r_files:
            ss.references = {}
            for f in r_files:
                try:
                    pat = load_xrd_csv(f, name=f.name)
                    ss.references[f.name] = pat
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            st.success(f"Loaded {len(ss.references)} reference(s).")

    st.divider()
    st.subheader("Raw reference images (optional, included in report)")
    img_files = st.file_uploader(
        "Reference images (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True, key="raw_imgs",
    )
    if img_files:
        ss.raw_ref_images = {}
        tmp = Path(tempfile.mkdtemp(prefix="xrd_imgs_"))
        for f in img_files:
            p = tmp / f.name
            p.write_bytes(f.getvalue())
            stem = Path(f.name).stem
            ss.raw_ref_images[stem] = str(p)
            ss.raw_ref_images[f.name] = str(p)
            ss.raw_ref_images[stem + "_extracted"] = str(p)
        st.success(f"Loaded {len(img_files)} raw image(s). They will be embedded in the PDF report.")

    if ss.samples or ss.references:
        st.divider()
        rows = []
        for p in ss.samples.values():
            rows.append({"role": "sample", "name": p.name, "points": len(p.two_theta),
                         "2θ range": f"{p.two_theta.min():.2f}–{p.two_theta.max():.2f}",
                         "I range": f"{p.intensity.min():.0f}–{p.intensity.max():.0f}"})
        for p in ss.references.values():
            rows.append({"role": "reference", "name": p.name, "points": len(p.two_theta),
                         "2θ range": f"{p.two_theta.min():.2f}–{p.two_theta.max():.2f}",
                         "I range": f"{p.intensity.min():.0f}–{p.intensity.max():.0f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ==========================================================================
def page_analyze():
    st.header("2 · Analyze individual patterns")
    patterns: dict[str, XRDPattern] = {}
    for n, p in ss.samples.items():
        patterns[f"[sample] {n}"] = p
    for n, p in ss.references.items():
        patterns[f"[ref] {n}"] = p
    if not patterns:
        st.info("Upload data on page 1 first.")
        return

    choice = st.selectbox("Pattern", list(patterns.keys()))
    pat = patterns[choice]

    c1, c2, c3 = st.columns(3)
    prom = c1.slider("Prominence (fraction)", 0.001, 0.5, ss.params["prominence"], 0.001)
    win = c2.slider("Min distance (pts)", 1, 50, ss.params["window"])
    tol = c3.slider("Match tolerance (°2θ)", 0.01, 1.0, ss.params["tolerance"], 0.01)
    ss.params.update({"prominence": prom, "window": win, "tolerance": tol})

    idx = detect_peaks(pat, prominence=prom, window=win)
    st.plotly_chart(plotly_pattern(pat, idx, title=choice), use_container_width=True)
    st.write(f"**{len(idx)} peaks detected**")
    if len(idx):
        st.dataframe(
            pd.DataFrame({"2θ (deg)": pat.two_theta[idx],
                          "I (counts)": pat.intensity[idx]}),
            use_container_width=True, height=260,
        )


# ==========================================================================
def _run_all_comparisons():
    p = ss.params
    results: dict[str, dict] = {}
    for sn, sample in ss.samples.items():
        results[sn] = {}
        for rn, ref in ss.references.items():
            results[sn][rn] = compare_patterns(
                sample, ref,
                tolerance_deg=p["tolerance"],
                prominence=p["prominence"],
                window=p["window"],
            )
    ss.results = results


def page_compare():
    st.header("3 · Compare samples vs references")
    if not ss.samples or not ss.references:
        st.info("Need ≥1 sample and ≥1 reference (page 1).")
        return

    _run_all_comparisons()

    rows = []
    for sn, per_ref in ss.results.items():
        for rn, r in per_ref.items():
            rows.append({
                "sample": sn, "reference": rn,
                "matched": len(r.matched),
                "new_in_sample": len(r.new_in_sample),
                "missing_from_sample": len(r.missing_from_sample),
                "match_score": round(r.match_score, 3),
                "novelty_score": round(r.novelty_score, 3),
            })
    st.subheader("Score summary (all sample × reference pairs)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    sel_sample = st.selectbox("Sample", list(ss.samples.keys()))
    sample = ss.samples[sel_sample]
    refs_res = {rn: (ss.references[rn], r) for rn, r in ss.results[sel_sample].items()}

    st.subheader("Individual digitized patterns")
    cols = st.columns(2)
    all_pats = [("sample", sample)] + [("ref", ss.references[rn]) for rn in refs_res]
    for i, (role, pat) in enumerate(all_pats):
        with cols[i % 2]:
            idx = detect_peaks(pat, ss.params["prominence"], ss.params["window"])
            st.pyplot(plot_pattern_mpl(pat, idx, title=f"[{role}] {pat.name}"),
                      use_container_width=True)

    st.subheader("Combined overlay (all references on one axis)")
    st.pyplot(plot_overlay_mpl(sample, refs_res,
                               title=f"Combined — {sel_sample}"),
              use_container_width=True)

    st.subheader("Stacked overlay")
    st.pyplot(plot_stacked_overlay_mpl(sample, refs_res,
                                       title=f"Stacked — {sel_sample}"),
              use_container_width=True)

    st.subheader("Per-pair overlays (sample vs each reference)")
    for rn, (ref, res) in refs_res.items():
        st.pyplot(plot_pair_overlay_mpl(sample, ref, res,
                                        title=f"{sel_sample}  vs  {rn}"),
                  use_container_width=True)
        with st.expander(f"Peak table — {rn} (match {res.match_score:.2f} · novelty {res.novelty_score:.2f})"):
            st.dataframe(comparison_to_dataframe(res), use_container_width=True)


# ==========================================================================
def page_report():
    st.header("4 · Report & download")
    if not ss.results:
        st.info("Run a comparison on page 3 first.")
        return

    # Summary
    rows = []
    for sn, per_ref in ss.results.items():
        for rn, r in per_ref.items():
            rows.append({
                "sample": sn, "reference": rn,
                "match_score": round(r.match_score, 3),
                "novelty_score": round(r.novelty_score, 3),
                "matched": len(r.matched),
                "new": len(r.new_in_sample),
                "missing": len(r.missing_from_sample),
            })
    summary = pd.DataFrame(rows)
    st.dataframe(summary, use_container_width=True)

    # Build ZIP bundle: all digitized PNGs + all overlays + peak CSVs + PDF + summary
    st.subheader("Downloads")
    tmpdir = Path(tempfile.mkdtemp(prefix="xrd_report_"))
    pdf_path = tmpdir / "xrd_report.pdf"

    build_pdf_report(
        pdf_path,
        samples=ss.samples,
        references=ss.references,
        results=ss.results,
        raw_images=ss.raw_ref_images,
        params=ss.params,
    )

    # PDF direct download
    with open(pdf_path, "rb") as fh:
        st.download_button("⬇ Download PDF report", fh.read(),
                           file_name="xrd_report.pdf", mime="application/pdf")

    # Build ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.csv", summary.to_csv(index=False))

        # Digitized individual plots
        for n, pat in {**ss.samples, **ss.references}.items():
            idx = pat.peaks if len(pat.peaks) else detect_peaks(
                pat, ss.params["prominence"], ss.params["window"])
            fig = plot_pattern_mpl(pat, idx, title=n)
            z.writestr(f"digitized/{Path(n).stem}.png", fig_to_png_bytes(fig))
            import matplotlib.pyplot as plt
            plt.close(fig)

        # Per-sample overlays (combined + stacked + per-pair) + peak CSVs
        import matplotlib.pyplot as plt
        for sn, per_ref in ss.results.items():
            sample = ss.samples[sn]
            refs_res = {rn: (ss.references[rn], r) for rn, r in per_ref.items()}
            stem = Path(sn).stem

            fig = plot_overlay_mpl(sample, refs_res, title=f"Combined — {sn}")
            z.writestr(f"overlays_combined/{stem}.png", fig_to_png_bytes(fig))
            plt.close(fig)

            fig = plot_stacked_overlay_mpl(sample, refs_res, title=f"Stacked — {sn}")
            z.writestr(f"overlays_stacked/{stem}.png", fig_to_png_bytes(fig))
            plt.close(fig)

            for rn, r in per_ref.items():
                fig = plot_pair_overlay_mpl(sample, ss.references[rn], r,
                                            title=f"{sn}  vs  {rn}")
                z.writestr(
                    f"overlays_pairs/{stem}__vs__{Path(rn).stem}.png",
                    fig_to_png_bytes(fig),
                )
                plt.close(fig)
                df = comparison_to_dataframe(r)
                z.writestr(f"peaks/{stem}__vs__{Path(rn).stem}.csv",
                           df.to_csv(index=False))

        # Raw reference images
        seen = set()
        for key, p in ss.raw_ref_images.items():
            if p in seen:
                continue
            seen.add(p)
            try:
                z.write(p, arcname=f"raw_references/{Path(p).name}")
            except OSError:
                pass

        # Include PDF in ZIP too
        z.write(pdf_path, arcname="xrd_report.pdf")

    st.download_button("⬇ Download All (ZIP)", buf.getvalue(),
                       file_name="xrd_report.zip", mime="application/zip")


# ---------- nav ----------
PAGES = {
    "0 · Extract (Fiji)": page_extract,
    "1 · Upload":         page_upload,
    "2 · Analyze":        page_analyze,
    "3 · Compare":        page_compare,
    "4 · Report":         page_report,
}
choice = st.sidebar.radio("Pages", list(PAGES.keys()))
st.sidebar.divider()
st.sidebar.caption("XRD Compare — Fiji → analysis → PDF")
PAGES[choice]()
