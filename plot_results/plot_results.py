# visualize_v5_results.py
# Standalone script to generate publication-quality figures from trained v5 model
# No retraining - just inference + visualization
# FIXED: Uses calibrated threshold from training output

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set publication-quality plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Color scheme (consistent throughout)
COLORS = {
    'real': '#2E86AB',      # Professional blue
    'fake': '#A23B72',      # Deep magenta
    'accent': '#F18F01',    # Orange accent
    'neutral': '#6C757D'    # Gray
}

ROOT = Path(r"G:\My Drive\hindi_dfake")
META = ROOT / "metadata"

# HARDCODED: Calibrated threshold from training output
CALIBRATED_THRESHOLD = 0.111145  # From training: "Acc@thrEER=0.9200"

# ==================== Model Architecture (copied from train_moe.v5.py) ====================
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
        return x * self.sig(w)

class ImprovedExpert(nn.Module):
    def __init__(self, in_dim=1536, bottleneck=768, drop=0.3, use_batchnorm=True, use_se=False):
        super().__init__()
        self.use_se = use_se
        if use_batchnorm:
            self.pre = nn.Sequential(
                nn.BatchNorm1d(in_dim), nn.Linear(in_dim, bottleneck),
                nn.GELU(), nn.Dropout(drop),
            )
        else:
            self.pre = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, bottleneck),
                nn.GELU(), nn.Dropout(drop),
            )
        if use_se:
            self.se = SE1D(bottleneck, reduction=16)
        self.mid = nn.Sequential(
            nn.Linear(bottleneck, bottleneck), nn.GELU(), nn.Dropout(drop),
        )
        self.head = nn.Linear(bottleneck, 2)

    def forward(self, x):
        h = self.pre(x)
        if self.use_se: h = self.se(h)
        h2 = self.mid(h)
        return self.head(h + h2)

class TinyGate(nn.Module):
    def __init__(self, in_dim_concat, hidden=64, drop=0.15, n_experts=2, simple=False):
        super().__init__()
        if simple:
            self.net = nn.Sequential(nn.Linear(in_dim_concat, n_experts))
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim_concat), nn.Linear(in_dim_concat, hidden),
                nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, n_experts)
            )
    def forward(self, x_concat):
        return torch.softmax(self.net(x_concat), dim=1)

class MoEModel(nn.Module):
    def __init__(self, ptms: List[str], in_dim_each=1536, expert_bottleneck=768, expert_drop=0.3,
                 gate_hidden=64, gate_drop=0.15, use_batchnorm=True, use_se=False, 
                 simple_gate=False, stochastic_depth=0.6, use_fusion=False, fusion_dropout=0.5):
        super().__init__()
        self.ptms = ptms
        self.stochastic_depth = stochastic_depth
        self.use_fusion = use_fusion
        self.experts = nn.ModuleDict({
            ptm: ImprovedExpert(in_dim_each, expert_bottleneck, expert_drop, use_batchnorm, use_se) 
            for ptm in ptms
        })
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
            gate_w = gate_w * mask / (gate_w.sum(dim=1, keepdim=True) + 1e-8)
        
        expert_logits = torch.stack([self.experts[p](x) for p, x in zip(ptm_order, xs)], dim=1)
        final_logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)
        
        if self.use_fusion:
            final_logits = self.fusion(final_logits)
        
        return final_logits, expert_logits, gate_w

# ==================== Data Loading ====================
def load_vec(path: str):
    try:
        if "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm" in path:
            path = path.replace(
                "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm",
                r"C:\Users\pc 1\hindi_df\ptm"
            )
        if not os.path.isfile(path): return None
        v = np.load(path, mmap_mode="r")
        if v.dtype != np.float32: v = v.astype(np.float32, copy=False)
        if v.ndim != 1 or v.shape[0] <= 0: return None
        return v
    except: return None

def to_int_labels(series: pd.Series) -> np.ndarray:
    s = series.copy()
    if s.dtype == object:
        low = s.astype(str).str.lower()
        mapped = np.where(low.isin(["fake", "1"]), 1,
                 np.where(low.isin(["real", "0"]), 0, np.nan))
        out = pd.Series(mapped)
    else:
        out = s
    return out.fillna(0).astype(np.int64).values

class PTMDataset(Dataset):
    def __init__(self, csv_path: str, ptm_list: List[str], ptm_columns: Dict[str, str]):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.ptms = ptm_list
        self.ptm_cols = ptm_columns
        self.labels = to_int_labels(self.df["label"])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs = {}
        for ptm in self.ptms:
            v = load_vec(row[self.ptm_cols[ptm]])
            if v is None: return None
            xs[ptm] = torch.from_numpy(v)
        return {"x": xs, "y": torch.tensor(int(self.labels[idx]), dtype=torch.long)}

def collate_fn(batch_list):
    batch_list = [b for b in batch_list if b is not None]
    if not batch_list: return {"x": {}, "y": torch.empty(0, dtype=torch.long)}
    ptm_names = list(batch_list[0]["x"].keys())
    xs = {ptm: torch.stack([b["x"][ptm] for b in batch_list]) for ptm in ptm_names}
    y = torch.stack([b["y"] for b in batch_list])
    return {"x": xs, "y": y}

# ==================== Inference ====================
@torch.no_grad()
def run_inference(model, loader, device, threshold):
    """Run inference and collect all outputs using calibrated threshold."""
    model.eval()
    all_logits, all_gate_weights, all_labels = [], [], []
    
    print(f"[Inference] Running model on test set (threshold={threshold:.6f})...")
    for batch in tqdm(loader, desc="Processing batches"):
        if batch is None or batch["y"].numel() == 0: continue
        xs = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y = batch["y"].to(device, non_blocking=True)
        
        logits, _, gate_w = model(xs)
        
        all_logits.append(logits.cpu().numpy())
        all_gate_weights.append(gate_w.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)      # (N, 2)
    gate_weights = np.concatenate(all_gate_weights, axis=0)  # (N, num_experts)
    labels = np.concatenate(all_labels, axis=0)      # (N,)
    
    # Convert logits to probabilities
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    scores = probs[:, 1]  # P(fake)
    predictions = (scores >= threshold).astype(int)  # USE CALIBRATED THRESHOLD
    
    print(f"[Inference] Collected {len(labels)} samples")
    return {
        'labels': labels,
        'predictions': predictions,
        'scores': scores,
        'gate_weights': gate_weights,
        'logits': logits,
        'threshold': threshold
    }

# ==================== Visualization Functions ====================

def plot_confusion_matrix(results, save_path):
    """Generate beautiful confusion matrix."""
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Normalize for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                square=True, cbar_kws={'label': 'Percentage'},
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                ax=ax, vmin=0, vmax=1)
    
    # Add counts as text
    for i in range(2):
        for j in range(2):
            text = ax.text(j+0.5, i+0.7, f'n={cm[i,j]}',
                          ha="center", va="center", color="gray", fontsize=8)
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix\n(v5 Model, thr={results["threshold"]:.6f})', 
                 fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_roc_curve(results, save_path):
    """Generate beautiful ROC curve."""
    fpr, tpr, thresholds = roc_curve(results['labels'], results['scores'])
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color=COLORS['accent'], lw=2.5, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
    
    # Mark EER point (where FPR ≈ FNR)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    ax.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10, 
            label=f'EER = {eer:.4f}', zorder=5)
    
    # Mark calibrated threshold point
    calib_idx = np.argmin(np.abs(thresholds - results['threshold']))
    ax.plot(fpr[calib_idx], tpr[calib_idx], 'go', markersize=10,
            label=f'Calibrated Thr = {results["threshold"]:.4f}', zorder=5)
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve\n(v5 Model on Test Set)', fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_expert_utilization_heatmap(results, ptm_names, save_path):
    """Generate beautiful expert utilization heatmap."""
    gate_weights = results['gate_weights']  # (N, num_experts)
    labels = results['labels']
    
    # Separate by class
    real_gates = gate_weights[labels == 0]
    fake_gates = gate_weights[labels == 1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Prepare data for heatmaps
    data_sets = [
        (real_gates, 'Real Samples', COLORS['real']),
        (fake_gates, 'Fake Samples', COLORS['fake']),
        (gate_weights, 'All Samples', COLORS['neutral'])
    ]
    
    for ax, (data, title, color) in zip(axes, data_sets):
        # Compute statistics
        mean_weights = data.mean(axis=0)
        std_weights = data.std(axis=0)
        
        # Create data matrix for heatmap (experts x metrics)
        heatmap_data = np.array([mean_weights, std_weights])
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(ptm_names)))
        ax.set_xticklabels(ptm_names, rotation=45, ha='right')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Mean', 'Std Dev'])
        
        # Add text annotations
        for i in range(2):
            for j in range(len(ptm_names)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10,
                              fontweight='bold')
        
        ax.set_title(f'{title}\n(n={len(data):,})', fontweight='bold', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Gate Weight', rotation=270, labelpad=15)
    
    plt.suptitle('Expert Utilization Analysis (Gate Weights)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_score_distribution(results, save_path):
    """Score distribution histogram with calibrated threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    real_scores = results['scores'][results['labels'] == 0]
    fake_scores = results['scores'][results['labels'] == 1]
    
    ax.hist(real_scores, bins=50, alpha=0.7, color=COLORS['real'], 
            label=f'Real (n={len(real_scores)})', edgecolor='black', linewidth=0.5)
    ax.hist(fake_scores, bins=50, alpha=0.7, color=COLORS['fake'], 
            label=f'Fake (n={len(fake_scores)})', edgecolor='black', linewidth=0.5)
    
    # Mark calibrated threshold
    ax.axvline(results['threshold'], color='green', linestyle='--', lw=2.5, 
               label=f'Calibrated Threshold ({results["threshold"]:.4f})')
    
    ax.set_xlabel('Model Score (Probability of Fake)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Score Distribution by True Label\n(v5 Model on Test Set)', 
                 fontweight='bold', pad=15)
    ax.legend(loc='upper center', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description='Generate publication figures from v5 model')
    parser.add_argument('--ckpt', type=str, default='checkpoints/moe_ptm2_v5_aggressive_best.pt',
                        help='Path to v5 best checkpoint')
    parser.add_argument('--test-real-csv', type=str, 
                        default=str(META / "fs_test_real.labeled.csv"))
    parser.add_argument('--test-fake-csv', type=str, 
                        default=str(META / "fs_test_fake_mms.labeled.csv"))
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Directory to save figures')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=CALIBRATED_THRESHOLD,
                        help=f'Classification threshold (default: {CALIBRATED_THRESHOLD:.6f} from training)')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Setup] Device: {device}")
    print(f"[Setup] Loading checkpoint: {args.ckpt}")
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    
    # Build model
    model = MoEModel(
        ptms=cfg['ptms'],
        in_dim_each=cfg['ptm_dim'],
        expert_bottleneck=cfg['expert_bottleneck'],
        expert_drop=cfg['expert_dropout'],
        gate_hidden=cfg['gate_hidden'],
        gate_drop=cfg['gate_dropout'],
        use_batchnorm=cfg['use_batchnorm'],
        use_se=cfg['use_se'],
        simple_gate=cfg['simple_gate'],
        stochastic_depth=cfg['stochastic_depth'],
        use_fusion=cfg['use_fusion'],
        fusion_dropout=cfg['fusion_dropout']
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"[Setup] Model loaded successfully (epoch {ckpt['epoch']})")
    print(f"[Setup] Using calibrated threshold: {args.threshold:.6f}")
    
    # Auto-detect PTM columns
    def resolve_ptm_columns(csv_path, ptm_list):
        df_head = pd.read_csv(csv_path, nrows=50)
        cols = list(df_head.columns)
        col_map = {}
        for ptm in ptm_list:
            target = ptm.lower().replace("-", "").replace("_", "")
            cands = [c for c in cols if target in c.lower().replace("-", "").replace("_", "")]
            if cands: col_map[ptm] = cands[0]
        return col_map
    
    PTM_COLUMNS = resolve_ptm_columns(args.test_real_csv, cfg['ptms'])
    print(f"[Setup] PTM columns: {PTM_COLUMNS}")
    
    # Load test data
    test_real_ds = PTMDataset(args.test_real_csv, cfg['ptms'], PTM_COLUMNS)
    test_fake_ds = PTMDataset(args.test_fake_csv, cfg['ptms'], PTM_COLUMNS)
    
    # Combine into single loader
    from torch.utils.data import ConcatDataset
    test_ds = ConcatDataset([test_real_ds, test_fake_ds])
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"[Setup] Test set: {len(test_real_ds)} real + {len(test_fake_ds)} fake = {len(test_ds)} total")
    
    # Run inference with calibrated threshold
    results = run_inference(model, test_loader, device, threshold=args.threshold)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate figures
    print("\n[Visualization] Generating figures...")
    
    plot_confusion_matrix(results, output_dir / "confusion_matrix.pdf")
    plot_roc_curve(results, output_dir / "roc_curve.pdf")
    plot_expert_utilization_heatmap(results, cfg['ptms'], output_dir / "expert_utilization.pdf")
    plot_score_distribution(results, output_dir / "score_distribution.pdf")
    
    # Also save as PNG for easy viewing
    plot_confusion_matrix(results, output_dir / "confusion_matrix.png")
    plot_roc_curve(results, output_dir / "roc_curve.png")
    plot_expert_utilization_heatmap(results, cfg['ptms'], output_dir / "expert_utilization.png")
    plot_score_distribution(results, output_dir / "score_distribution.png")
    
    # Print summary
    acc = (results['predictions'] == results['labels']).mean()
    print(f"\n[Results Summary]")
    print(f"  Threshold used: {results['threshold']:.6f}")
    print(f"  Total samples: {len(results['labels'])}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Real samples: {(results['labels']==0).sum()}")
    print(f"  Fake samples: {(results['labels']==1).sum()}")
    print(f"\n[Done] All figures saved to: {output_dir.absolute()}")
    print(f"  → Open the PDF files for highest quality!")

if __name__ == "__main__":
    main()