import os
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Root export directory (relative to backend)
EXPORT_DIR = Path(os.environ.get("XAI_EXPORT_DIR", "xai_exports"))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_get_temporal(basic_xai: Dict[str, Any]) -> Dict[str, Any]:
    th = basic_xai.get("temporal_heatmap") or {}
    timestamps = th.get("timestamps") or []
    scores = th.get("scores") or []
    return {
        "timestamps": np.asarray(timestamps, dtype=float) if timestamps else np.array([], dtype=float),
        "scores": np.asarray(scores, dtype=float) if scores else np.array([], dtype=float),
        "mean_score": float(th.get("mean_score", 0.0)),
        "std_score": float(th.get("std_score", 0.0)),
        "consistency_index": float(th.get("consistency_index", 0.0)),
    }


def _safe_get_shap(advanced_xai: Dict[str, Any]) -> Dict[str, Any]:
    shap = advanced_xai.get("shap_approximation") or {}
    feat_imp = np.asarray(shap.get("feature_importance") or [], dtype=float)
    feat_imp_full = np.asarray(shap.get("feature_importance_full") or [], dtype=float)
    top_features = shap.get("top_features") or []
    expert_contrib = shap.get("expert_contributions") or {}
    baseline_pred = shap.get("baseline_pred")
    actual_pred = shap.get("actual_pred")
    coalition_preds = shap.get("coalition_preds") or {}
    return {
        "feature_importance": feat_imp,
        "feature_importance_full": feat_imp_full,
        "top_features": top_features,
        "expert_contributions": expert_contrib,
        "baseline_pred": float(baseline_pred) if baseline_pred is not None else None,
        "actual_pred": float(actual_pred) if actual_pred is not None else None,
        "coalition_preds": coalition_preds,
    }


def _safe_get_integrated_gradients(advanced_xai: Dict[str, Any]) -> Dict[str, Any]:
    ig = advanced_xai.get("integrated_gradients") or {}
    feat_attr = np.asarray(ig.get("feature_attribution") or [], dtype=float)
    per_expert = ig.get("attribution_per_expert") or {}
    per_expert_out: Dict[str, np.ndarray] = {}
    for k, v in per_expert.items():
        try:
            per_expert_out[str(k)] = np.asarray(v or [], dtype=float)
        except Exception:
            continue
    return {"feature_attribution": feat_attr, "attribution_per_expert": per_expert_out}


def _safe_get_expert_agreement(basic_xai: Dict[str, Any]) -> Dict[str, Any]:
    ea = basic_xai.get("expert_agreement") or {}
    experts = ea.get("experts") or {}
    out = []
    for name, info in experts.items():
        try:
            out.append(
                {
                    "name": str(name),
                    "prob_fake": float(info.get("prob_fake", 0.0)),
                    "gate_weight": float(info.get("gate_weight", 0.0)),
                }
            )
        except Exception:
            continue
    return {
        "experts": out,
        "agreement_score": float(ea.get("agreement_score", 0.0)),
        "interpretation": str(ea.get("interpretation", "")),
    }


def _safe_get_breathing(basic_xai: Dict[str, Any]) -> Dict[str, Any]:
    bp = basic_xai.get("breathing_patterns") or {}
    pauses = bp.get("pauses") or []
    out = []
    for p in pauses:
        try:
            out.append(
                {
                    "start": float(p.get("start", 0.0)),
                    "end": float(p.get("end", 0.0)),
                    "duration": float(p.get("duration", 0.0)),
                }
            )
        except Exception:
            continue
    return {
        "pauses": out,
        "regularity_score": float(bp.get("regularity_score", 0.0)),
        "mean_pause_duration": float(bp.get("mean_pause_duration", 0.0)),
        "std_pause_duration": float(bp.get("std_pause_duration", 0.0)),
        "interpretation": str(bp.get("interpretation", "")),
    }


def _safe_get_frequency(basic_xai: Dict[str, Any]) -> Dict[str, Any]:
    fc = basic_xai.get("frequency_contribution") or {}
    mel = np.asarray(fc.get("mel_spectrogram") or [], dtype=float)
    freq_bins = np.asarray(fc.get("freq_bins") or [], dtype=float)
    time_bins = np.asarray(fc.get("time_bins") or [], dtype=float)
    suspicious_in = fc.get("suspicious_bands") or []
    suspicious_out = []
    for b in suspicious_in:
        try:
            suspicious_out.append(
                {
                    "freq_hz": float(b.get("freq_hz", 0.0)),
                    "uniformity": float(b.get("uniformity", 0.0)),
                    "freq_range": str(b.get("freq_range", "")),
                }
            )
        except Exception:
            continue
    return {
        "mel_spectrogram": mel,
        "freq_bins": freq_bins,
        "time_bins": time_bins,
        "suspicious_bands": suspicious_out,
        "high_freq_energy": float(fc.get("high_freq_energy", 0.0)),
        "formant_consistency": float(fc.get("formant_consistency", 0.0)),
    }


def _plot_temporal_line(ax, timestamps: np.ndarray, scores: np.ndarray, title: str):
    if timestamps.size == 0 or scores.size == 0:
        ax.text(0.5, 0.5, "No temporal data", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.plot(timestamps, scores, color="#2563eb", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fake probability")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.0, 1.0)


def _plot_waveform_style(ax, timestamps: np.ndarray, scores: np.ndarray, title: str):
    if timestamps.size == 0 or scores.size == 0:
        ax.text(0.5, 0.5, "No temporal data", ha="center", va="center")
        ax.set_axis_off()
        return
    duration = float(timestamps[-1]) if timestamps.size > 0 else 1.0
    samples = 250

    # Match frontend ImprovedWaveformCard: derive waveform from temporal scores
    avg_score = float(scores.mean())
    deltas = scores - avg_score
    max_delta = float(np.max(np.abs(deltas)) if deltas.size > 0 else 1.0)
    if max_delta < 1e-4:
        max_delta = 1e-4

    t = np.linspace(0.0, duration, samples)
    # Map sample positions into score indices and interpolate
    wave = np.zeros_like(t)
    if scores.size > 1:
        positions = (np.arange(samples) / (samples - 1)) * (scores.size - 1)
        idx0 = np.floor(positions).astype(int)
        idx1 = np.minimum(idx0 + 1, scores.size - 1)
        frac = positions - idx0
        s_vals = scores[idx0] * (1.0 - frac) + scores[idx1] * frac
    else:
        s_vals = np.full_like(t, scores[0] if scores.size == 1 else avg_score, dtype=float)

    deltas_interp = s_vals - avg_score
    # Flip vertical orientation to match frontend SVG rendering
    wave = -np.clip(deltas_interp / max_delta, -1.0, 1.0)

    # High-confidence regions: scores above mean
    mean_score = avg_score
    high_mask = scores > mean_score

    ax.plot(t, wave, color="#60a5fa", linewidth=1.5)

    # Highlight windows between timestamps where score > mean
    for i in range(len(timestamps) - 1):
        if not high_mask[i]:
            continue
        start_t = timestamps[i]
        end_t = timestamps[i + 1]
        m = (t >= start_t) & (t <= end_t)
        if not m.any():
            continue
        ax.plot(t[m], wave[m], color="#ef4444", linewidth=2.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Waveform (normalized from scores)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def _plot_temporal_consistency(ax, timestamps: np.ndarray, scores: np.ndarray, mean_score: float, std_score: float, consistency_index: float, title: str):
    if timestamps.size == 0 or scores.size == 0:
        ax.text(0.5, 0.5, "No temporal data", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.bar(timestamps, scores, width=max(timestamps[-1] / max(len(timestamps), 1), 0.05), color="#22c55e", alpha=0.8)
    ax.axhline(mean_score, color="#f97316", linestyle="--", linewidth=1.5, label=f"Mean={mean_score:.2f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fake probability")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    text = f"std={std_score:.3f}\nconsistency_index={consistency_index:.3f}"
    ax.text(0.99, 0.01, text, ha="right", va="bottom", transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))


def _plot_shap_top_features(ax, shap_data: Dict[str, Any], title: str):
    feat_imp_full: np.ndarray = shap_data.get("feature_importance_full")
    feat_imp_viz: np.ndarray = shap_data.get("feature_importance")
    expert_contrib: Dict[str, float] = shap_data.get("expert_contributions") or {}

    # Prefer full-resolution attribution if available.
    base = None
    if isinstance(feat_imp_full, np.ndarray) and feat_imp_full.size > 0:
        base = feat_imp_full
    elif isinstance(feat_imp_viz, np.ndarray) and feat_imp_viz.size > 0:
        base = feat_imp_viz

    if base is None and not expert_contrib:
        ax.text(0.5, 0.5, "No SHAP data", ha="center", va="center")
        ax.set_axis_off()
        return

    # Determine top features.
    top_features = shap_data.get("top_features") or []
    if top_features:
        idx = [int(d.get("index", 0)) for d in top_features]
        vals = np.asarray([float(d.get("importance", 0.0)) for d in top_features], dtype=float)
    elif base is not None and base.size > 0:
        k = int(min(20, int(base.size)))
        top_idx = np.argsort(base)[-k:][::-1]
        idx = [int(i) for i in top_idx]
        vals = np.asarray([float(base[i]) for i in top_idx], dtype=float)
    else:
        idx = []
        vals = np.asarray([], dtype=float)

    if vals.size == 0:
        if expert_contrib:
            names = list(expert_contrib.keys())

            def _sort_key(n: str):
                nl = str(n).lower()
                if "hubert" in nl:
                    return (0, nl)
                if "wav2vec" in nl or "wav2vec2" in nl:
                    return (1, nl)
                return (2, nl)

            names = sorted(names, key=_sort_key)
            phi = np.asarray([float(expert_contrib.get(k, 0.0)) for k in names], dtype=float)

            baseline_pred = shap_data.get("baseline_pred")
            actual_pred = shap_data.get("actual_pred")
            baseline = float(baseline_pred) if baseline_pred is not None else 0.0
            actual = float(actual_pred) if actual_pred is not None else float(baseline + float(phi.sum()))

            ax.axhline(
                baseline,
                color="#6b7280",
                linestyle="--",
                linewidth=1.6,
                label=f"Baseline ({baseline:.3f})",
            )
            ax.axhline(
                actual,
                color="#111827",
                linestyle="-",
                linewidth=2.2,
                label=f"Actual ({actual:.3f})",
            )

            x = np.arange(int(phi.size))
            curr = baseline
            path_vals = [baseline]
            for i, (name, contrib) in enumerate(zip(names, phi)):
                start = float(curr)
                end = float(curr + float(contrib))
                color = "#ef4444" if float(contrib) >= 0.0 else "#22c55e"
                ax.bar(i, end - start, bottom=start, width=0.6, color=color, alpha=0.85)

                delta = abs(end - start)
                pad = max(0.015, delta * 0.35)
                y_text = end + pad if float(contrib) >= 0.0 else end - pad
                va = "bottom" if float(contrib) >= 0.0 else "top"
                ax.annotate(
                    "Marginal\n" + f"{float(contrib):+.3f}",
                    xy=(i, end),
                    xytext=(i, y_text),
                    ha="center",
                    va=va,
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                )

                curr = end
                path_vals.append(curr)

            path_vals.append(actual)
            y_min = float(min(path_vals))
            y_max = float(max(path_vals))
            pad = max(0.02, (y_max - y_min) * 0.18)
            ax.set_ylim(y_min - pad, y_max + pad)

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in names], rotation=0, ha="center", fontsize=9)
            ax.set_ylabel("P(fake)")
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.2)
            ax.legend(loc="upper left", fontsize=8)
            return

        ax.text(0.5, 0.5, "No SHAP data", ha="center", va="center")
        ax.set_axis_off()
        return

    if float(vals.max()) > 0.0:
        vals_norm = (vals / float(vals.max())) * 100.0
    else:
        vals_norm = vals

    x = np.arange(int(vals_norm.size))
    ax.bar(x, vals_norm, color="#3b82f6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idx], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Relative importance (%)")
    ax.set_xlabel("Feature dimension index")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)

    if expert_contrib:
        try:
            contrib_txt = ", ".join([f"{k}={float(v):.3f}" for k, v in expert_contrib.items()])
        except Exception:
            contrib_txt = ""
        if contrib_txt:
            ax.text(
                0.99,
                0.01,
                f"expert_contrib: {contrib_txt}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )


def _plot_true_frequency_band_energy(ax, freq_data: Dict[str, Any], title: str):
    mel = freq_data.get("mel_spectrogram")
    freq_bins = freq_data.get("freq_bins")
    if not isinstance(mel, np.ndarray) or mel.size == 0 or not isinstance(freq_bins, np.ndarray) or freq_bins.size == 0:
        ax.text(0.5, 0.5, "No frequency data", ha="center", va="center")
        ax.set_axis_off()
        return

    # mel is in dB (ref=np.max). Convert to a relative linear scale for aggregation.
    mel_lin = np.power(10.0, mel / 10.0)
    per_bin_energy = mel_lin.mean(axis=1)  # average over time

    # Use paper-friendly Hz bands.
    edges = np.array([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, float(freq_bins.max()) + 1.0], dtype=float)
    labels = [
        "0-250", "250-500", "500-1k", "1-1.5k", "1.5-2k", "2-3k",
        "3-4k", "4-6k", "6-8k", "8-12k", "12-16k", "16k+",
    ]

    band_vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freq_bins >= lo) & (freq_bins < hi)
        if not np.any(m):
            band_vals.append(0.0)
        else:
            band_vals.append(float(per_bin_energy[m].mean()))
    band_vals = np.asarray(band_vals, dtype=float)
    if float(band_vals.max()) > 0.0:
        band_vals = (band_vals / float(band_vals.max())) * 100.0

    x = np.arange(len(labels))
    ax.bar(x, band_vals, color="#f97316", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Relative band energy (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)


def _plot_integrated_gradients_features(ax, ig_data: Dict[str, Any], title: str):
    feat_attr: np.ndarray = ig_data.get("feature_attribution")
    if not isinstance(feat_attr, np.ndarray) or feat_attr.size == 0:
        ax.text(0.5, 0.5, "No Integrated Gradients data", ha="center", va="center")
        ax.set_axis_off()
        return

    per_expert: Dict[str, np.ndarray] = ig_data.get("attribution_per_expert") or {}
    plotted = []

    colors = ["#3b82f6", "#f97316", "#22c55e", "#ef4444", "#a78bfa"]
    for i, (name, arr) in enumerate(per_expert.items()):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue
        xs = np.arange(int(arr.size))
        ax.plot(xs, arr, linewidth=1.8, alpha=0.9, label=f"{name}", color=colors[i % len(colors)])
        plotted.append(arr)

    x = np.arange(int(feat_attr.size))
    ax.plot(x, feat_attr, color="#7c3aed", linewidth=2.2, alpha=0.35, linestyle="--", label="combined (viz)")
    plotted.append(feat_attr)

    if plotted:
        all_vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in plotted if v is not None])
        if all_vals.size > 0:
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))
            pad = max(0.02, (vmax - vmin) * 0.15)
            ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

            mean_v = float(np.mean(all_vals))
            std_v = float(np.std(all_vals))
            ax.text(
                0.01,
                0.99,
                f"mean={mean_v:.3f}  std={std_v:.3f}\npooled embedding dims (NOT time)",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

    ax.set_xlabel("Feature-dimension bin (downsampled)")
    ax.set_ylabel("Relative IG magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    if int(feat_attr.size) == 50:
        boundary = 25
        ax.axvline(boundary - 0.5, color="#64748b", linewidth=1.2, linestyle=":", alpha=0.9)
        ax.text(
            0.02,
            0.02,
            "bins 0-24: mean(768)\nbins 25-49: std(768)",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
    ax.legend(loc="upper right", fontsize=7, ncol=1)


def _plot_expert_agreement(ax, ea: Dict[str, Any], title: str):
    experts = ea.get("experts") or []
    if not experts:
        ax.text(0.5, 0.5, "No expert agreement data", ha="center", va="center")
        ax.set_axis_off()
        return

    names = [e.get("name", "") for e in experts]
    prob_fake = np.asarray([float(e.get("prob_fake", 0.0)) for e in experts], dtype=float)
    gate_w = np.asarray([float(e.get("gate_weight", 0.0)) for e in experts], dtype=float)

    x = np.arange(len(names))
    ax.bar(x - 0.2, prob_fake, width=0.4, color="#ef4444", alpha=0.8, label="P(fake)")
    ax.bar(x + 0.2, gate_w, width=0.4, color="#2563eb", alpha=0.8, label="Gate weight")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    agreement_score = float(ea.get("agreement_score", 0.0))
    ax.text(
        0.99,
        0.01,
        f"agreement_score={agreement_score:.3f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )


def _plot_breathing_pauses(ax, breathing: Dict[str, Any], title: str):
    pauses = breathing.get("pauses") or []
    if not pauses:
        ax.text(0.5, 0.5, "No pauses detected", ha="center", va="center")
        ax.set_axis_off()
        return

    starts = np.asarray([float(p.get("start", 0.0)) for p in pauses], dtype=float)
    durations = np.asarray([float(p.get("duration", 0.0)) for p in pauses], dtype=float)

    y = np.zeros_like(starts)
    ax.scatter(starts, y, s=np.clip(durations * 400.0, 20.0, 300.0), color="#22c55e", alpha=0.85)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)

    regularity_score = float(breathing.get("regularity_score", 0.0))
    mean_pause = float(breathing.get("mean_pause_duration", 0.0))
    std_pause = float(breathing.get("std_pause_duration", 0.0))
    ax.text(
        0.99,
        0.01,
        f"regularity={regularity_score:.3f}\nmean={mean_pause:.3f}s\nstd={std_pause:.3f}s",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )


def _plot_mel_spectrogram(ax, freq_data: Dict[str, Any], breathing: Dict[str, Any], title: str):
    mel = freq_data.get("mel_spectrogram")
    if not isinstance(mel, np.ndarray) or mel.size == 0:
        ax.text(0.5, 0.5, "No mel-spectrogram data", ha="center", va="center")
        ax.set_axis_off()
        return

    freq_bins = freq_data.get("freq_bins")
    time_bins = freq_data.get("time_bins")

    def _edges_from_centers(centers: np.ndarray, clamp_min: Optional[float] = None) -> np.ndarray:
        centers = np.asarray(centers, dtype=float)
        if centers.size == 1:
            w = 1.0
            left = centers[0] - w / 2.0
            right = centers[0] + w / 2.0
            if clamp_min is not None:
                left = max(clamp_min, left)
            return np.asarray([left, right], dtype=float)

        edges = np.zeros(centers.size + 1, dtype=float)
        edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
        edges[0] = centers[0] - (edges[1] - centers[0])
        edges[-1] = centers[-1] + (centers[-1] - edges[-2])
        if clamp_min is not None:
            edges[0] = max(clamp_min, edges[0])
        return edges

    x0, x1 = 0.0, float(mel.shape[1])
    y0, y1 = 0.0, float(mel.shape[0])
    x_label = "Time (downsampled bins)"
    y_label = "Mel bins"

    img = None
    if (
        isinstance(time_bins, np.ndarray)
        and isinstance(freq_bins, np.ndarray)
        and time_bins.size == mel.shape[1]
        and freq_bins.size == mel.shape[0]
        and time_bins.size > 0
        and freq_bins.size > 0
    ):
        t_edges = _edges_from_centers(time_bins)
        f_edges = _edges_from_centers(freq_bins, clamp_min=0.0)
        x0, x1 = float(t_edges[0]), float(t_edges[-1])
        y0, y1 = float(f_edges[0]), float(f_edges[-1])
        x_label = "Time (s)"
        y_label = "Frequency (Hz)"
        img = ax.pcolormesh(t_edges, f_edges, mel, shading="auto", cmap="magma")
    else:
        img = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    high_freq_energy = float(freq_data.get("high_freq_energy", 0.0))
    if y_label == "Frequency (Hz)" and y1 >= 8000.0 and high_freq_energy <= -60.0:
        ax.axhline(8000.0, color="#f97316", linewidth=1.8)
        ax.text(
            x0 + (x1 - x0) * 0.01,
            8000.0 + (y1 - y0) * 0.02,
            "Missing HF content (8kHz+)",
            color="#f97316",
            fontsize=8,
            fontweight="bold",
        )

    suspicious_bands = freq_data.get("suspicious_bands") or []
    if y_label == "Frequency (Hz)" and isinstance(freq_bins, np.ndarray) and freq_bins.size > 0 and suspicious_bands:
        for b in suspicious_bands:
            try:
                freq_hz = float(b.get("freq_hz", 0.0))
            except Exception:
                continue
            idx = int(np.argmin(np.abs(freq_bins - freq_hz)))
            try:
                f_edges = _edges_from_centers(freq_bins, clamp_min=0.0)
                lo = float(f_edges[idx])
                hi = float(f_edges[idx + 1])
            except Exception:
                lo = float(freq_bins[idx - 1]) if idx > 0 else float(freq_bins[0])
                hi = float(freq_bins[idx])
            ax.axhspan(lo, hi, color="#fbbf24", alpha=0.22)

    pauses = (breathing or {}).get("pauses") or []
    regularity_score = float((breathing or {}).get("regularity_score", 0.0))
    if x_label == "Time (s)" and pauses and regularity_score > 0.7:
        for p in pauses:
            try:
                start = float(p.get("start", 0.0))
                end = float(p.get("end", 0.0))
            except Exception:
                continue
            if end <= x0 or start >= x1:
                continue
            start = max(x0, start)
            end = min(x1, end)
            rect = patches.Rectangle(
                (start, y0),
                end - start,
                y1 - y0,
                fill=False,
                edgecolor="#ef4444",
                linewidth=2.0,
                alpha=0.9,
            )
            ax.add_patch(rect)

        ax.text(
            0.01,
            0.01,
            "⚠ Robotic pause pattern",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            fontsize=9,
            color="#ef4444",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )
    formant_consistency = float(freq_data.get("formant_consistency", 0.0))
    ax.text(
        0.99,
        0.01,
        f"high_freq_energy={high_freq_energy:.3g}\nformant_consistency={formant_consistency:.3g}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )


def generate_xai_plots(stem: str, basic_xai: Dict[str, Any], advanced_xai: Dict[str, Any]) -> Path:
    """Generate 4 XAI figures (PNG) using existing /xai results and return a zip path."""
    export_root = EXPORT_DIR / stem
    export_root.mkdir(parents=True, exist_ok=True)

    temporal = _safe_get_temporal(basic_xai)
    shap_data = _safe_get_shap(advanced_xai)
    ig_data = _safe_get_integrated_gradients(advanced_xai)
    expert_agreement = _safe_get_expert_agreement(basic_xai)
    breathing = _safe_get_breathing(basic_xai)
    freq_data = _safe_get_frequency(basic_xai)

    # 1. Temporal line (IntegratedGradientsMinimal-style)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    _plot_temporal_line(ax1, temporal["timestamps"], temporal["scores"], "Temporal confidence over time")
    fig1.tight_layout()
    fig1_path = export_root / f"{stem}_temporal_line.png"
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    # 2. Waveform-style temporal view
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    _plot_waveform_style(ax2, temporal["timestamps"], temporal["scores"], "Waveform-style temporal confidence")
    fig2.tight_layout()
    fig2_path = export_root / f"{stem}_waveform_style.png"
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    # 3. Temporal consistency / variation
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    _plot_temporal_consistency(
        ax3,
        temporal["timestamps"],
        temporal["scores"],
        temporal["mean_score"],
        temporal["std_score"],
        temporal["consistency_index"],
        "Temporal consistency / variation",
    )
    fig3.tight_layout()
    fig3_path = export_root / f"{stem}_temporal_consistency.png"
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)

    # 4. SHAP feature attribution (best-use for this model: top feature dimensions)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    _plot_shap_top_features(ax4, shap_data, "SHAP (expert Shapley contributions)")
    fig4.tight_layout()
    fig4_path = export_root / f"{stem}_shap_frequency.png"
    fig4.savefig(fig4_path, dpi=200)
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(7, 4))
    _plot_expert_agreement(ax5, expert_agreement, "Expert agreement")
    fig5.tight_layout()
    fig5_path = export_root / f"{stem}_expert_agreement.png"
    fig5.savefig(fig5_path, dpi=200)
    plt.close(fig5)

    fig6, ax6 = plt.subplots(figsize=(8, 3))
    _plot_breathing_pauses(ax6, breathing, "Breathing / pause timeline")
    fig6.tight_layout()
    fig6_path = export_root / f"{stem}_breathing_pauses.png"
    fig6.savefig(fig6_path, dpi=200)
    plt.close(fig6)

    fig7, ax7 = plt.subplots(figsize=(8, 4))
    _plot_mel_spectrogram(ax7, freq_data, breathing, "Mel spectrogram (model-independent)")
    fig7.tight_layout()
    fig7_path = export_root / f"{stem}_mel_spectrogram.png"
    fig7.savefig(fig7_path, dpi=200)
    plt.close(fig7)

    include_true_freq_bands = os.environ.get("XAI_EXPORT_INCLUDE_TRUE_FREQ_BANDS", "0").strip().lower() in {"1", "true", "yes"}
    if include_true_freq_bands:
        fig7b, ax7b = plt.subplots(figsize=(8, 4))
        _plot_true_frequency_band_energy(ax7b, freq_data, "True frequency-band energy (Hz) from mel")
        fig7b.tight_layout()
        fig7b_path = export_root / f"{stem}_true_frequency_bands.png"
        fig7b.savefig(fig7b_path, dpi=200)
        plt.close(fig7b)
    else:
        fig7b_path = None

    fig8, ax8 = plt.subplots(figsize=(8, 4))
    _plot_integrated_gradients_features(ax8, ig_data, "Integrated Gradients (feature attribution)")
    fig8.tight_layout()
    fig8_path = export_root / f"{stem}_integrated_gradients_features.png"
    fig8.savefig(fig8_path, dpi=200)
    plt.close(fig8)

    # Zip all PNGs for download
    import zipfile

    zip_path = EXPORT_DIR / f"{stem}_xai_plots.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        files = [fig1_path, fig2_path, fig3_path, fig4_path, fig5_path, fig6_path, fig7_path, fig8_path]
        if fig7b_path is not None:
            files.append(fig7b_path)
        for p in files:
            if p is not None and p.exists():
                zf.write(p, arcname=p.name)

    return zip_path
