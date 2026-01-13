import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

# Prefer mplcairo for correct complex-script shaping (e.g., Devanagari) when available.
# This must run BEFORE importing pyplot.
if os.environ.get("GLOBAL_XAI_DISABLE_MPLCAIRO", "0") != "1":
    try:
        import mplcairo  # noqa: F401

        matplotlib.use("module://mplcairo.agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


COLORS = {
    "real": "#2E86AB",
    "fake": "#A23B72",
    "accent": "#F18F01",
    "neutral": "#6C757D",
}


def _configure_matplotlib_fonts(preferred_devanagari_font: str = "") -> None:
    # Embed TrueType fonts in PDF/PS (better for publication).
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    candidates = [
        preferred_devanagari_font.strip(),
        "Noto Sans Devanagari",
        "Noto Serif Devanagari",
        "Nirmala UI",  # Windows
        "Mangal",      # Windows
        "Lohit Devanagari",
    ]
    candidates = [c for c in candidates if c]

    chosen = next((c for c in candidates if c in available), "")
    if chosen:
        # Use a font list so Latin/Serif still looks publication-like while Hindi renders.
        matplotlib.rcParams["font.family"] = ["DejaVu Serif", chosen]
        matplotlib.rcParams["font.serif"] = ["DejaVu Serif", chosen]
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        # Avoid repeated warnings from Matplotlib about Devanagari support.
        warnings.filterwarnings("once", message=r"Matplotlib currently does not support Devanagari.*")
    else:
        # Fall back to DejaVu; Hindi may render poorly. Tell user via a single warning.
        warnings.warn(
            "No Devanagari-capable font found (e.g., 'Nirmala UI', 'Mangal', 'Noto Sans Devanagari'). "
            "Hindi labels may not render correctly. Install a Devanagari font and re-run.",
            RuntimeWarning,
        )


def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: dict, path: list, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")


def _sorted_band_keys(band_stats: dict) -> list:
    def _key(x: str):
        try:
            lo = float(x.split("Hz")[0].split("-")[0])
            return lo
        except Exception:
            return float("inf")

    return sorted(list(band_stats.keys()), key=_key)


def _plot_temporal(results: dict, title: str) -> Optional[plt.Figure]:
    tp = results.get("temporal_patterns") or {}
    real = tp.get("real") or {}
    fake = tp.get("fake") or {}

    real_mu = real.get("avg_scores_by_position")
    fake_mu = fake.get("avg_scores_by_position")
    if not real_mu or not fake_mu:
        return None

    real_mu = np.asarray(real_mu, dtype=float)
    fake_mu = np.asarray(fake_mu, dtype=float)
    real_sd = np.asarray(real.get("std_scores_by_position") or np.zeros_like(real_mu), dtype=float)
    fake_sd = np.asarray(fake.get("std_scores_by_position") or np.zeros_like(fake_mu), dtype=float)

    x_real = np.linspace(0.0, 1.0, real_mu.size)
    x_fake = np.linspace(0.0, 1.0, fake_mu.size)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x_real, real_mu, color=COLORS["real"], lw=2.2, label=f"Real (n={real.get('n_samples', 'NA')})")
    ax.fill_between(x_real, real_mu - real_sd, real_mu + real_sd, color=COLORS["real"], alpha=0.18, linewidth=0)

    ax.plot(x_fake, fake_mu, color=COLORS["fake"], lw=2.2, label=f"Fake (n={fake.get('n_samples', 'NA')})")
    ax.fill_between(x_fake, fake_mu - fake_sd, fake_mu + fake_sd, color=COLORS["fake"], alpha=0.18, linewidth=0)

    ax.set_xlabel("Normalized time (0=start, 1=end)")
    ax.set_ylabel("Mean suspiciousness score")
    ax.set_title(f"Temporal Suspiciousness Profile\n{title}")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    return fig


def _plot_frequency_band_energy(results: dict, title: str) -> Optional[plt.Figure]:
    fa = results.get("frequency_analysis") or {}
    bs = fa.get("band_statistics") or {}
    real_bs = bs.get("real") or {}
    fake_bs = bs.get("fake") or {}
    if not real_bs or not fake_bs:
        return None

    bands = _sorted_band_keys(real_bs)
    rows = []
    for b in bands:
        r = real_bs.get(b) or {}
        f = fake_bs.get(b) or {}
        rows.append((b, "real", float(r.get("mean_energy", np.nan))))
        rows.append((b, "fake", float(f.get("mean_energy", np.nan))))

    import pandas as pd

    df = pd.DataFrame(rows, columns=["band", "label", "mean_energy"]).dropna()

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    sns.barplot(
        data=df,
        x="band",
        y="mean_energy",
        hue="label",
        palette={"real": COLORS["real"], "fake": COLORS["fake"]},
        ax=ax,
    )
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Mean band energy")
    ax.set_title(f"Frequency Band Energy (Mean)\n{title}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
    return fig


def _plot_suspicious_bands(results: dict, title: str) -> Optional[plt.Figure]:
    fa = results.get("frequency_analysis") or {}
    sb = fa.get("suspicious_bands")
    if not sb:
        return None

    import pandas as pd

    df = pd.DataFrame(sb)
    if "freq_range" not in df.columns or "mean_variance" not in df.columns:
        return None

    df = df.copy()
    df["mean_variance"] = pd.to_numeric(df["mean_variance"], errors="coerce")
    df = df.dropna(subset=["mean_variance"])  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    sns.barplot(data=df, x="freq_range", y="mean_variance", color=COLORS["accent"], ax=ax)
    ax.set_xlabel("Frequency range")
    ax.set_ylabel("Mean variance")
    ax.set_title(f"Suspicious Frequency Bands (Low Variance)\n{title}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    return fig


def _plot_spectrogram_difference(results: dict, title: str) -> Optional[plt.Figure]:
    ch = _safe_get(results, ["frequency_analysis", "comparison_heatmap"], {}) or {}
    diff = ch.get("difference_map")
    if diff is None:
        return None

    arr = np.asarray(diff, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None

    vmax = float(np.nanpercentile(np.abs(arr), 99.0)) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Spectrogram Difference Heatmap (Fake - Real)\n{title}")
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Mel bins")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Difference")
    return fig


def _plot_expert_shap(results: dict, title: str) -> Optional[plt.Figure]:
    es = results.get("expert_shap_analysis") or {}
    real = es.get("real") or {}
    fake = es.get("fake") or {}

    r_mu = real.get("expert_contribution_mean")
    f_mu = fake.get("expert_contribution_mean")
    if not isinstance(r_mu, dict) or not isinstance(f_mu, dict) or not r_mu or not f_mu:
        return None

    r_sd = real.get("expert_contribution_std") or {}
    f_sd = fake.get("expert_contribution_std") or {}

    experts = sorted(set(list(r_mu.keys()) + list(f_mu.keys())))
    r_vals = np.array([float(r_mu.get(k, 0.0)) for k in experts], dtype=float)
    f_vals = np.array([float(f_mu.get(k, 0.0)) for k in experts], dtype=float)
    r_err = np.array([float(r_sd.get(k, 0.0)) for k in experts], dtype=float)
    f_err = np.array([float(f_sd.get(k, 0.0)) for k in experts], dtype=float)

    x = np.arange(len(experts))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - w / 2, r_vals, width=w, color=COLORS["real"], yerr=r_err, capsize=4, label="Real")
    ax.bar(x + w / 2, f_vals, width=w, color=COLORS["fake"], yerr=f_err, capsize=4, label="Fake")
    ax.axhline(0.0, color=COLORS["neutral"], lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(experts, rotation=15)
    ax.set_ylabel("Mean expert contribution")
    ax.set_title(f"Expert Contributions (SHAP over Experts)\n{title}")
    ax.legend(title="")
    ax.grid(True, axis="y", alpha=0.25)
    return fig


def _plot_ig(results: dict, title: str) -> Optional[plt.Figure]:
    ig = results.get("integrated_gradients_analysis") or {}
    real = ig.get("real") or {}
    fake = ig.get("fake") or {}
    r_mu = real.get("combined_mean_curve")
    f_mu = fake.get("combined_mean_curve")
    if not r_mu or not f_mu:
        return None

    r_mu = np.asarray(r_mu, dtype=float)
    f_mu = np.asarray(f_mu, dtype=float)
    r_sd = np.asarray(real.get("combined_std_curve") or np.zeros_like(r_mu), dtype=float)
    f_sd = np.asarray(fake.get("combined_std_curve") or np.zeros_like(f_mu), dtype=float)

    x_r = np.arange(1, r_mu.size + 1)
    x_f = np.arange(1, f_mu.size + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    ax.plot(x_r, r_mu, color=COLORS["real"], lw=2.2, label=f"Real (n={real.get('n_samples', 'NA')})")
    ax.fill_between(x_r, r_mu - r_sd, r_mu + r_sd, color=COLORS["real"], alpha=0.18, linewidth=0)
    ax.plot(x_f, f_mu, color=COLORS["fake"], lw=2.2, label=f"Fake (n={fake.get('n_samples', 'NA')})")
    ax.fill_between(x_f, f_mu - f_sd, f_mu + f_sd, color=COLORS["fake"], alpha=0.18, linewidth=0)
    ax.set_xlabel("Feature-dimension bin (1..50; NOT time)")
    ax.set_ylabel("IG magnitude (pooled features)")
    ax.set_title("Integrated Gradients (Feature Attribution; Mean ± Std)")
    ax.legend(title="")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    r_ratio = real.get("second_half_to_first_half_ratio_mean")
    f_ratio = fake.get("second_half_to_first_half_ratio_mean")
    labels = ["Real", "Fake"]
    vals = [float(r_ratio) if r_ratio is not None else np.nan, float(f_ratio) if f_ratio is not None else np.nan]
    ax2.bar(labels, vals, color=[COLORS["real"], COLORS["fake"]])
    ax2.set_ylabel("Std-half / Mean-half ratio (feature dims)")
    ax2.set_title("Concentration across pooled feature halves")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Integrated Gradients Summary\n{title}")
    fig.tight_layout()
    return fig


def _plot_high_risk_words(results: dict, title: str, top_k: int = 20) -> Optional[plt.Figure]:
    lp = results.get("linguistic_patterns") or {}
    hrw = lp.get("high_risk_words")
    if not hrw:
        return None

    words = [str(x.get("word", "")) for x in hrw][:top_k]
    scores = [float(x.get("avg_score_when_present", np.nan)) for x in hrw][:top_k]
    occ = [int(x.get("occurrences", 0)) for x in hrw][:top_k]

    y = np.arange(len(words))
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.barh(y, scores, color=COLORS["accent"], alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Avg suspiciousness when present")
    ax.set_title(f"Top High-Risk Words (n≥3 occurrences)\n{title}")
    ax.grid(True, axis="x", alpha=0.25)

    for i, (s, c) in enumerate(zip(scores, occ)):
        if np.isfinite(s):
            ax.text(s + 0.002, i, f"n={c}", va="center", fontsize=9, color=COLORS["neutral"])

    return fig


def _plot_syllable_complexity(results: dict, title: str) -> Optional[plt.Figure]:
    lp = results.get("linguistic_patterns") or {}
    sc = lp.get("syllable_complexity") or {}
    real = sc.get("real") or {}
    fake = sc.get("fake") or {}
    if not real or not fake:
        return None

    r_avg = float(real.get("avg_syllables_per_word", np.nan))
    f_avg = float(fake.get("avg_syllables_per_word", np.nan))
    r_4p = float(real.get("words_4plus_syllables", np.nan))
    f_4p = float(fake.get("words_4plus_syllables", np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    axes[0].bar(["Real", "Fake"], [r_avg, f_avg], color=[COLORS["real"], COLORS["fake"]])
    axes[0].set_ylabel("Avg syllables per word")
    axes[0].set_title("Syllable complexity")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(["Real", "Fake"], [r_4p, f_4p], color=[COLORS["real"], COLORS["fake"]])
    axes[1].set_ylabel("Count of 4+ syllable words")
    axes[1].set_title("Rare complex words")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Linguistic Complexity Summary\n{title}")
    fig.tight_layout()
    return fig


def _plot_crossref_summary(results: dict, title: str) -> Optional[plt.Figure]:
    cr = results.get("cross_referenced_insights")
    if not isinstance(cr, list) or not cr:
        return None

    align = None
    glob = None
    for item in cr:
        if isinstance(item, dict) and item.get("pattern_type") == "alignment_statistics":
            align = item
        if isinstance(item, dict) and item.get("pattern_type") == "global_summary":
            glob = item

    if align is None and glob is None:
        return None

    lines = []
    if glob is not None:
        lines.append(str(glob.get("description", "")))
        trw = glob.get("top_risk_words")
        if isinstance(trw, list) and trw:
            lines.append("Top risk words: " + ", ".join([str(x) for x in trw[:8]]))

    if align is not None:
        desc = align.get("description")
        if desc:
            lines.append(str(desc))
        tf = align.get("temporal_freq_alignment")
        tl = align.get("temporal_linguistic_alignment")
        at = align.get("all_three_alignment")
        if tf is not None and tl is not None and at is not None:
            lines.append(f"Temporal-Freq alignment: {float(tf)*100:.1f}%")
            lines.append(f"Temporal-Linguistic alignment: {float(tl)*100:.1f}%")
            lines.append(f"All-three alignment: {float(at)*100:.1f}%")

    text = "\n".join([x for x in lines if x])
    if not text.strip():
        return None

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.axis("off")
    ax.set_title(f"Cross-Referenced Insights Summary\n{title}", pad=14)
    ax.text(0.02, 0.86, text, va="top", ha="left", fontsize=11)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", default="global_xai_results.json")
    parser.add_argument("--out-dir", default="figures_pub")
    parser.add_argument("--title", default="")
    parser.add_argument("--devanagari-font", default="")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    out_dir = Path(args.out_dir)

    sns.set_theme(style="whitegrid")
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = 450
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["axes.labelsize"] = 11
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["legend.fontsize"] = 9

    _configure_matplotlib_fonts(args.devanagari_font)

    results = _load_json(results_path)

    title = args.title.strip() or results_path.parent.name

    figs = []
    fig = _plot_temporal(results, title)
    if fig is not None:
        figs.append(("01_temporal_profile", fig))

    fig = _plot_frequency_band_energy(results, title)
    if fig is not None:
        figs.append(("02_frequency_band_energy", fig))

    fig = _plot_suspicious_bands(results, title)
    if fig is not None:
        figs.append(("03_suspicious_frequency_bands", fig))

    fig = _plot_spectrogram_difference(results, title)
    if fig is not None:
        figs.append(("04_spectrogram_difference", fig))

    fig = _plot_expert_shap(results, title)
    if fig is not None:
        figs.append(("05_expert_shap", fig))

    fig = _plot_ig(results, title)
    if fig is not None:
        figs.append(("06_integrated_gradients", fig))

    fig = _plot_high_risk_words(results, title)
    if fig is not None:
        figs.append(("07_high_risk_words", fig))

    fig = _plot_syllable_complexity(results, title)
    if fig is not None:
        figs.append(("08_syllable_complexity", fig))

    fig = _plot_crossref_summary(results, title)
    if fig is not None:
        figs.append(("09_crossref_summary", fig))

    combined_pdf = out_dir / "all_figures.pdf"
    with PdfPages(combined_pdf) as pdf:
        for stem, f in figs:
            _save_figure(f, out_dir, stem)
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)

    print(f"Saved {len(figs)} figures to: {out_dir}")


if __name__ == "__main__":
    main()
