from __future__ import annotations

import os
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parent
META_DIR = REPO_ROOT / "metadata"
LOCAL_PTM_DIR = REPO_ROOT / "ptm"

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13

COLORS = {
    "real": "#2E86AB",
    "fake": "#A23B72",
    "accent": "#F18F01",
    "neutral": "#6C757D",
}

MIN_NPY_BYTES = 1024


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _norm(s: str) -> str:
    return str(s).replace("\\", "/")


def map_csv_vec_to_local_clean(path_str: str) -> str:
    """Map old CSV PTM vec path -> local repo ptm/<ptm>/clean/..."""
    s = _norm(path_str)

    needle = "/processed/features/ptm/"
    i = s.lower().find(needle)
    if i != -1:
        tail = s[i + len(needle) :]
        parts = tail.split("/")
        if len(parts) >= 2:
            ptm = parts[0]
            rest = "/".join(parts[1:])
            if rest.startswith("raw/"):
                rest = "clean/" + rest[len("raw/") :]
            elif rest.startswith("clean/"):
                rest = rest
            else:
                rest = "clean/" + rest
            return str((LOCAL_PTM_DIR / ptm / Path(rest)).resolve())

    # Already local?
    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    return str((REPO_ROOT / p).resolve())


def load_vec(path: str) -> Optional[np.ndarray]:
    try:
        path2 = map_csv_vec_to_local_clean(path)
        if not os.path.isfile(path2):
            return None
        if os.path.getsize(path2) < MIN_NPY_BYTES:
            return None
        v = np.load(path2, mmap_mode="r")
        if v.dtype != np.float32:
            v = v.astype(np.float32, copy=False)
        if v.ndim != 1 or v.shape[0] <= 0:
            return None
        return v
    except Exception:
        return None


def amp_ctx(device: str, enabled: bool):
    if enabled and device == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def to_int_labels(series: pd.Series) -> np.ndarray:
    s = series.copy()
    if s.dtype == object:
        low = s.astype(str).str.lower()
        mapped = np.where(low.isin(["fake", "1"]), 1, np.where(low.isin(["real", "0"]), 0, np.nan))
        out = pd.Series(mapped)
    else:
        out = s
    return out.fillna(0).astype(np.int64).values


def resolve_ptm_columns(csv_path: str, ptm_list: List[str]) -> Dict[str, str]:
    df_head = pd.read_csv(csv_path, nrows=200)
    cols = list(df_head.columns)

    def norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "")

    npy_like = []
    for c in cols:
        if df_head[c].dtype == object:
            vals = df_head[c].dropna().astype(str)
            if not vals.empty and (vals.str.endswith(".npy").mean() > 0.7):
                npy_like.append(c)

    col_map: Dict[str, str] = {}
    for ptm in ptm_list:
        target = norm(ptm)
        cands = [c for c in npy_like if target in norm(c)]
        if len(cands) == 1:
            col_map[ptm] = cands[0]
        elif len(cands) > 1:
            cands.sort(key=len)
            col_map[ptm] = cands[0]
        else:
            fallback = [c for c in npy_like if c not in col_map.values()]
            if not fallback:
                raise ValueError(f"Could not find .npy column for PTM '{ptm}' in {csv_path}.")
            col_map[ptm] = fallback[0]

    return col_map


class PTMDataset(Dataset):
    def __init__(self, csv_path: str, ptm_list: List[str], ptm_columns: Dict[str, str]):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.ptms = ptm_list
        self.ptm_cols = ptm_columns
        if "label" not in self.df.columns:
            raise ValueError(f"Missing 'label' in {csv_path}")
        self.labels = to_int_labels(self.df["label"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs = {}
        for ptm in self.ptms:
            p = row[self.ptm_cols[ptm]]
            v = load_vec(p)
            if v is None:
                return None
            xs[ptm] = torch.from_numpy(v)
        y = int(self.labels[idx])
        return {"x": xs, "y": torch.tensor(y, dtype=torch.long)}


def collate_fn(batch_list):
    batch_list = [b for b in batch_list if b is not None]
    if len(batch_list) == 0:
        return {"x": {}, "y": torch.empty(0, dtype=torch.long)}
    ptm_names = list(batch_list[0]["x"].keys())
    xs = {ptm: torch.stack([b["x"][ptm] for b in batch_list], dim=0) for ptm in ptm_names}
    y = torch.stack([b["y"] for b in batch_list], dim=0)
    return {"x": xs, "y": y}


class SE1D(nn.Module):
    def __init__(self, C: int, reduction: int = 16):
        super().__init__()
        r = max(1, C // reduction)
        self.fc1 = nn.Linear(C, r)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(r, C)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.fc2(self.act(self.fc1(x)))
        w = self.sig(w)
        return x * w


class ImprovedExpert(nn.Module):
    def __init__(self, in_dim=1536, bottleneck=768, drop=0.3, use_batchnorm=True, use_se=False):
        super().__init__()
        self.use_se = use_se

        if use_batchnorm:
            self.pre = nn.Sequential(
                nn.BatchNorm1d(in_dim),
                nn.Linear(in_dim, bottleneck),
                nn.GELU(),
                nn.Dropout(drop),
            )
        else:
            self.pre = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, bottleneck),
                nn.GELU(),
                nn.Dropout(drop),
            )

        if use_se:
            self.se = SE1D(bottleneck, reduction=16)

        self.mid = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.head = nn.Linear(bottleneck, 2)

    def forward(self, x):
        h = self.pre(x)
        if self.use_se:
            h = self.se(h)
        h2 = self.mid(h)
        h = h + h2
        return self.head(h)


class TinyGate(nn.Module):
    def __init__(self, in_dim_concat, hidden=64, drop=0.15, n_experts=2, simple=False):
        super().__init__()
        if simple:
            self.net = nn.Sequential(nn.Linear(in_dim_concat, n_experts))
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim_concat),
                nn.Linear(in_dim_concat, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden, n_experts),
            )

    def forward(self, x_concat):
        return torch.softmax(self.net(x_concat), dim=1)


class MoEModel(nn.Module):
    def __init__(
        self,
        ptms: List[str],
        in_dim_each=1536,
        expert_bottleneck=768,
        expert_drop=0.3,
        gate_hidden=64,
        gate_drop=0.15,
        use_batchnorm=True,
        use_se=False,
        simple_gate=False,
        stochastic_depth=0.6,
        use_fusion=False,
        fusion_dropout=0.5,
    ):
        super().__init__()
        self.ptms = ptms
        self.stochastic_depth = stochastic_depth
        self.use_fusion = use_fusion

        self.experts = nn.ModuleDict(
            {ptm: ImprovedExpert(in_dim_each, expert_bottleneck, expert_drop, use_batchnorm, use_se) for ptm in ptms}
        )
        self.gate = TinyGate(in_dim_each * len(ptms), gate_hidden, gate_drop, len(ptms), simple_gate)

        if use_fusion:
            self.fusion = nn.Sequential(nn.Linear(2, 2), nn.Dropout(fusion_dropout))

    def forward(self, x_dict: Dict[str, torch.Tensor]):
        ptm_order = self.ptms
        xs = [x_dict[ptm] for ptm in ptm_order]
        x_concat = torch.cat(xs, dim=1)
        gate_w = self.gate(x_concat)

        if self.training and self.stochastic_depth < 1.0:
            keep_prob = torch.full((gate_w.shape[0], len(self.ptms)), self.stochastic_depth, device=gate_w.device)
            mask = torch.bernoulli(keep_prob)
            gate_w = gate_w * mask
            gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-8)

        expert_logits = torch.stack([self.experts[p](x) for p, x in zip(ptm_order, xs)], dim=1)
        final_logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)

        if self.use_fusion:
            final_logits = self.fusion(final_logits)

        return final_logits, expert_logits, gate_w


def eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if len(scores) == 0 or len(labels) == 0:
        return 0.5, 0.5
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    P = (labels == 1).sum()
    N = (labels == 0).sum()
    if P == 0 or N == 0:
        return 0.5, 0.5
    tp = fp = 0
    fn = P
    best_diff = 1.0
    eer = 1.0
    thr_at_eer = scores[0]
    prev_s = np.inf
    for i in range(len(scores)):
        s, y = scores[i], labels[i]
        if s != prev_s:
            fpr = fp / N
            fnr = fn / P
            diff = abs(fpr - fnr)
            if diff < best_diff:
                best_diff = diff
                eer = (fpr + fnr) / 2.0
                thr_at_eer = prev_s
            prev_s = s
        if y == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
    fpr = fp / N
    fnr = fn / P
    diff = abs(fpr - fnr)
    if diff < best_diff:
        eer = (fpr + fnr) / 2.0
        thr_at_eer = scores[-1]
    return float(eer), float(thr_at_eer)


@torch.no_grad()
def run_inference_collect(model, loader, device):
    model.eval()
    all_scores, all_gate, all_labels = [], [], []

    for batch in tqdm(loader, desc="infer", unit="batch"):
        if batch is None or batch["y"].numel() == 0:
            continue
        xs = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y = batch["y"].to(device, non_blocking=True)

        with amp_ctx(device, enabled=True):
            logits, _, gate_w = model(xs)
            scores = torch.softmax(logits, dim=1)[:, 1]

        all_scores.append(scores.detach().cpu().numpy())
        all_gate.append(gate_w.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    if not all_scores:
        return {
            "scores": np.zeros((0,), dtype=np.float32),
            "gate_weights": np.zeros((0, 0), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
        }

    return {
        "scores": np.concatenate(all_scores),
        "gate_weights": np.concatenate(all_gate),
        "labels": np.concatenate(all_labels).astype(np.int32),
    }


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, threshold: float, save_path: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Percentage"},
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
        vmin=0,
        vmax=1,
    )

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"n={cm[i, j]}", ha="center", va="center", color="gray", fontsize=8)

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(f"Confusion Matrix\n(thr={threshold:.6f})", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, threshold: float, save_path: Path):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLORS["accent"], lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Classifier")

    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    ax.plot(fpr[eer_idx], tpr[eer_idx], "ro", markersize=10, label=f"EER = {eer:.4f}", zorder=5)

    calib_idx = np.argmin(np.abs(thresholds - threshold))
    ax.plot(
        fpr[calib_idx],
        tpr[calib_idx],
        "go",
        markersize=10,
        label=f"Calibrated Thr = {threshold:.4f}",
        zorder=5,
    )

    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curve (MoE v5 on Test Set)", fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_expert_utilization_heatmap(gate_weights: np.ndarray, labels: np.ndarray, ptm_names: List[str], save_path: Path):
    real_gates = gate_weights[labels == 0]
    fake_gates = gate_weights[labels == 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    data_sets = [
        (real_gates, "Real Samples"),
        (fake_gates, "Fake Samples"),
        (gate_weights, "All Samples"),
    ]

    for ax, (data, title) in zip(axes, data_sets):
        mean_weights = data.mean(axis=0)
        std_weights = data.std(axis=0)
        heatmap_data = np.array([mean_weights, std_weights])

        im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(ptm_names)))
        ax.set_xticklabels(ptm_names, rotation=45, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Mean", "Std Dev"])

        for i in range(2):
            for j in range(len(ptm_names)):
                ax.text(
                    j,
                    i,
                    f"{heatmap_data[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_title(f"{title}\n(n={len(data):,})", fontweight="bold", pad=10)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Gate Weight", rotation=270, labelpad=15)

    plt.suptitle("Expert Utilization Analysis (Gate Weights)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_score_distribution(labels: np.ndarray, scores: np.ndarray, threshold: float, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    ax.hist(
        real_scores,
        bins=50,
        alpha=0.7,
        color=COLORS["real"],
        label=f"Real (n={len(real_scores)})",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        fake_scores,
        bins=50,
        alpha=0.7,
        color=COLORS["fake"],
        label=f"Fake (n={len(fake_scores)})",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axvline(threshold, color="green", linestyle="--", lw=2.5, label=f"Calibrated Thr ({threshold:.4f})")

    ax.set_xlabel("Model Score (P(fake))", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Score Distribution by True Label (MoE v5)", fontweight="bold", pad=15)
    ax.legend(loc="upper center", framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def audit_features(csv_path: str, ptms: List[str], ptm_cols: Dict[str, str], limit_show: int = 5):
    df = pd.read_csv(csv_path, usecols=list(set(["label"] + [ptm_cols[p] for p in ptms])))
    missing = {ptm: 0 for ptm in ptms}
    samples = {ptm: [] for ptm in ptms}

    for ptm in ptms:
        col = ptm_cols[ptm]
        for p in df[col].astype(str).tolist():
            lp = map_csv_vec_to_local_clean(p)
            ok = os.path.isfile(lp) and (os.path.getsize(lp) >= MIN_NPY_BYTES)
            if not ok:
                missing[ptm] += 1
                if len(samples[ptm]) < limit_show:
                    samples[ptm].append((p, lp))

    total_rows = len(df)
    total_missing = sum(missing.values())
    print(f"[audit] {Path(csv_path).name}: rows={total_rows:,}")
    for ptm in ptms:
        print(f"  {ptm}: missing={missing[ptm]:,}")
        for orig, loc in samples[ptm]:
            print(f"    - orig: {orig}")
            print(f"      loc : {loc}")

    if total_missing > 0:
        raise RuntimeError("Missing feature files detected. Fix paths / rerun feature extraction before evaluation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(REPO_ROOT / "models" / "moe_ptm2_v5_aggressive_best.pt"))
    ap.add_argument("--test-real-csv", type=str, default=str(META_DIR / "fs_test_real.labeled.csv"))
    ap.add_argument("--test-fake-csv", type=str, default=str(META_DIR / "fs_test_fake_mms.labeled.csv"))
    ap.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "figures_clean_profile"))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--use-mms-threshold", action="store_true")
    ap.add_argument("--force-calib", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})

    ptms = cfg.get("ptms", ["wav2vec2-base", "hubert-base"])
    ptm_dim = int(cfg.get("ptm_dim", 1536))

    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    PTM_COLUMNS = resolve_ptm_columns(args.test_real_csv, ptms)
    print("[ptm columns]", " | ".join([f"{k} -> {v}" for k, v in PTM_COLUMNS.items()]))

    audit_features(args.test_real_csv, ptms, PTM_COLUMNS)
    audit_features(args.test_fake_csv, ptms, PTM_COLUMNS)

    test_real_ds = PTMDataset(args.test_real_csv, ptms, PTM_COLUMNS)
    test_fake_ds = PTMDataset(args.test_fake_csv, ptms, PTM_COLUMNS)

    test_real_loader = DataLoader(
        test_real_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    test_fake_loader = DataLoader(
        test_fake_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    model = MoEModel(
        ptms=ptms,
        in_dim_each=ptm_dim,
        expert_bottleneck=int(cfg.get("expert_bottleneck", 768)),
        expert_drop=float(cfg.get("expert_dropout", 0.3)),
        gate_hidden=int(cfg.get("gate_hidden", 64)),
        gate_drop=float(cfg.get("gate_dropout", 0.15)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        use_se=bool(cfg.get("use_se", False)),
        simple_gate=bool(cfg.get("simple_gate", False)),
        stochastic_depth=float(cfg.get("stochastic_depth", 0.6)),
        use_fusion=bool(cfg.get("use_fusion", True)),
        fusion_dropout=float(cfg.get("fusion_dropout", 0.5)),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    out_real = run_inference_collect(model, test_real_loader, device)
    out_fake = run_inference_collect(model, test_fake_loader, device)

    test_scores = np.concatenate([out_real["scores"], out_fake["scores"]])
    test_labels = np.concatenate([out_real["labels"], out_fake["labels"]])
    gate_weights = np.concatenate([out_real["gate_weights"], out_fake["gate_weights"]], axis=0)

    expected_total = len(test_real_ds) + len(test_fake_ds)
    if len(test_labels) != expected_total:
        print(f"[warn] Some samples were dropped due to missing features. got={len(test_labels)} expected={expected_total}")

    test_eer, thr_eer = eer_from_scores(test_scores, test_labels)
    acc05 = float(((test_scores >= 0.5).astype(np.int32) == test_labels).mean())
    acc_eer = float(((test_scores >= thr_eer).astype(np.int32) == test_labels).mean())
    err05 = 1.0 - acc05
    err_eerthr = 1.0 - acc_eer

    print("\n[RESULTS]")
    print(f" Test EER     : {test_eer:.4f} (thr@eer={thr_eer:.6f})")
    print(f" Acc@0.5      : {acc05:.4f} | Error@0.5     : {err05:.4f}")
    print(f" Acc@thrEER   : {acc_eer:.4f} | Error@thrEER : {err_eerthr:.4f}")

    calib_out = {}

    do_calib = bool(cfg.get("calibrate", True)) or args.force_calib
    calib_frac = float(cfg.get("calib_frac", 0.2))

    if do_calib and len(test_scores) > 0:
        rng = np.random.default_rng(seed)
        pos_idx = np.where(test_labels == 1)[0]
        neg_idx = np.where(test_labels == 0)[0]
        n_pos = max(1, int(len(pos_idx) * calib_frac))
        n_neg = max(1, int(len(neg_idx) * calib_frac))
        dev_idx = set(rng.choice(pos_idx, n_pos, replace=False).tolist() + rng.choice(neg_idx, n_neg, replace=False).tolist())
        hold_idx = [i for i in range(len(test_labels)) if i not in dev_idx]

        dev_scores = test_scores[list(dev_idx)]
        dev_labels = test_labels[list(dev_idx)]
        hold_scores = test_scores[hold_idx]
        hold_labels = test_labels[hold_idx]

        uniq = np.unique(dev_scores)
        grid = np.linspace(0.0, 1.0, 401)
        thr_grid = np.unique(np.concatenate([uniq, grid]))

        best_thr = 0.5
        best_acc = -1.0
        for t in thr_grid:
            a = float(((dev_scores >= t).astype(np.int32) == dev_labels).mean())
            if a > best_acc:
                best_acc, best_thr = a, t

        acc_hold = float(((hold_scores >= best_thr).astype(np.int32) == hold_labels).mean())

        err_hold = 1.0 - acc_hold
        print("\n[MMS calibration (on combined test)]")
        print(f"  dev_frac={calib_frac:.2f} | dev_n={len(dev_scores)} | holdout_n={len(hold_scores)}")
        print(f"  Calibrated thr={best_thr:.6f} (dev best acc={best_acc:.4f})")
        print(f"  Holdout Acc   ={acc_hold:.4f} | Holdout Error ={err_hold:.4f}")

        calib_out = {
            "dev_frac": float(calib_frac),
            "dev_n": int(len(dev_scores)),
            "holdout_n": int(len(hold_scores)),
            "calibrated_thr": float(best_thr),
            "dev_best_acc": float(best_acc),
            "holdout_acc": float(acc_hold),
            "holdout_error": float(err_hold),
        }

    thr_used = float(thr_eer)
    if args.use_mms_threshold and calib_out:
        thr_used = float(calib_out["calibrated_thr"])
    if args.threshold is not None:
        thr_used = float(args.threshold)

    preds = (test_scores >= thr_used).astype(np.int32)
    acc_thr = float((preds == test_labels).mean())

    print("\n[FINAL @ threshold]")
    print(f" thr={thr_used:.6f} | acc={acc_thr:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(test_labels, preds, thr_used, out_dir / "confusion_matrix.png")
    plot_roc_curve(test_labels, test_scores, thr_used, out_dir / "roc_curve.png")
    plot_expert_utilization_heatmap(gate_weights, test_labels, ptms, out_dir / "expert_utilization.png")
    plot_score_distribution(test_labels, test_scores, thr_used, out_dir / "score_distribution.png")

    plot_confusion_matrix(test_labels, preds, thr_used, out_dir / "confusion_matrix.pdf")
    plot_roc_curve(test_labels, test_scores, thr_used, out_dir / "roc_curve.pdf")
    plot_expert_utilization_heatmap(gate_weights, test_labels, ptms, out_dir / "expert_utilization.pdf")
    plot_score_distribution(test_labels, test_scores, thr_used, out_dir / "score_distribution.pdf")

    runs_dir = META_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": str(ckpt_path),
        "test_eer": float(test_eer),
        "thr_at_eer": float(thr_eer),
        "acc_at_0p5": float(acc05),
        "acc_at_thr_eer": float(acc_eer),
        "err_at_0p5": float(err05),
        "err_at_thr_eer": float(err_eerthr),
        "calibration": calib_out,
        "threshold_used": float(thr_used),
        "acc_at_threshold_used": float(acc_thr),
        "ptms": ptms,
        "ptm_dim": int(ptm_dim),
        "test_real_csv": str(args.test_real_csv),
        "test_fake_csv": str(args.test_fake_csv),
        "output_dir": str(out_dir),
    }

    out_json = runs_dir / f"eval_clean_test_{ckpt_path.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"\n[OK] Saved figures -> {out_dir}")
    print(f"[OK] Saved run manifest -> {out_json}")


if __name__ == "__main__":
    main()
