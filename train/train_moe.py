# train_moe.py — AGGRESSIVE regularization for 92k dataset + Mixup
import os, json, math, time, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from tqdm import tqdm
from contextlib import nullcontext

ROOT = Path(r"G:\My Drive\hindi_dfake")
META = ROOT / "metadata"

DEFAULTS = dict(
    seed=1337,
    train_csv=str(META / "split_train.labeled.csv"),
    val_csv=str(META / "split_val.labeled.csv"),
    test_real_csv=str(META / "fs_test_real.labeled.csv"),
    test_fake_csv=str(META / "fs_test_fake_mms.labeled.csv"),

    ptms=["wav2vec2-base", "hubert-base"],
    ptm_dim=1536,
    batch_size=512,
    balanced_sampler=True,
    num_workers=2,  # Windows-safe: 2 workers with proper spawning

    # AGGRESSIVE regularization for 92k dataset
    expert_bottleneck=768,
    expert_dropout=0.3,         # ↑ increased from 0.2

    gate_hidden=64,
    gate_dropout=0.15,          # ↑ increased from 0.1

    lr=8e-4,                    # ↓ slightly lower LR
    weight_decay=2e-2,          # ↑ stronger weight decay
    betas=(0.9, 0.999),
    warmup_ratio=0.05,
    cosine_final_lr=1e-5,       # ↓ lower final LR
    max_epochs=30,
    patience=7,                 # ↑ more patience for slower convergence
    label_smoothing=0.1,        # ↑ increased from 0.05
    aux_loss_lambda=0.15,       # ↓ reduced auxiliary loss weight
    gate_balance_lambda=0.15,   # ↑ stronger load balancing
    grad_clip=0.5,              # ↓ tighter gradient clipping
    amp=True,
    ckpt_dir="models",
    run_name="moe_ptm2_v5_aggressive",
    quiet=False,

    calibrate=True,
    calib_frac=0.2,
    
    # AGGRESSIVE architectural controls
    use_batchnorm=True,
    use_se=False,
    simple_gate=True,
    stochastic_depth=0.6,       # ↓ 40% expert dropout (was 0.8)
    use_fusion=True,
    fusion_dropout=0.5,         # ↑ heavy fusion dropout
    
    # NEW: Mixup augmentation
    mixup_alpha=0.3,            # Beta distribution parameter
    mixup_prob=0.5,             # 50% of batches use mixup
)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

MIN_NPY_BYTES = 1024

def load_vec(path: str) -> Optional[np.ndarray]:
    try:
        if "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm" in path:
            path = path.replace(
                "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm",
                r"C:\Users\pc 1\hindi_df\ptm"
            )
        if not os.path.isfile(path): 
            return None
        if os.path.getsize(path) < MIN_NPY_BYTES: return None
        v = np.load(path, mmap_mode="r")
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

# NEW: Mixup helper
def mixup_data(x_dict: Dict[str, torch.Tensor], y: torch.Tensor, alpha: float):
    """Apply mixup augmentation to feature dict and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = y.size(0)
    index = torch.randperm(batch_size).to(y.device)
    
    mixed_x = {}
    for k, v in x_dict.items():
        mixed_x[k] = lam * v + (1 - lam) * v[index, :]
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@torch.no_grad()
def compute_scores_and_labels(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        if batch is None or batch["y"].numel() == 0:
            continue
        xs = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y = batch["y"].to(device, non_blocking=True)
        with amp_ctx(device, enabled=True):
            logits, _, _ = model(xs)
            probs = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    if not all_scores:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(all_scores), np.concatenate(all_labels).astype(np.int32)

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item()

def eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if len(scores) == 0 or len(labels) == 0:
        return 0.5, 0.5
    order = np.argsort(-scores)
    scores = scores[order]; labels = labels[order]
    P = (labels == 1).sum(); N = (labels == 0).sum()
    if P == 0 or N == 0:
        return 0.5, 0.5
    tp = fp = 0; fn = P; tn = N
    best_diff = 1.0; eer = 1.0
    thr_at_eer = scores[0]
    prev_s = np.inf
    for i in range(len(scores)):
        s, y = scores[i], labels[i]
        if s != prev_s:
            fpr = fp / N; fnr = fn / P
            diff = abs(fpr - fnr)
            if diff < best_diff:
                best_diff = diff
                eer = (fpr + fnr) / 2.0
                thr_at_eer = prev_s
            prev_s = s
        if y == 1:
            tp += 1; fn -= 1
        else:
            fp += 1; tn -= 1
    fpr = fp / N; fnr = fn / P
    diff = abs(fpr - fnr)
    if diff < best_diff:
        eer = (fpr + fnr) / 2.0
        thr_at_eer = scores[-1]
    return float(eer), float(thr_at_eer)

def make_warmup_cosine(total_steps: int, warmup_ratio: float, base_lr: float, final_lr: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return final_lr / base_lr + (1 - final_lr / base_lr) * cosine
    return lr_lambda

def resolve_ptm_columns(csv_path: str, ptm_list: List[str]) -> Dict[str, str]:
    df_head = pd.read_csv(csv_path, nrows=200)
    cols = list(df_head.columns)
    def norm(s: str) -> str: return s.lower().replace("-", "").replace("_", "")
    npy_like = []
    for c in cols:
        if df_head[c].dtype == object:
            vals = df_head[c].dropna().astype(str)
            if not vals.empty and (vals.str.endswith(".npy").mean() > 0.7):
                npy_like.append(c)
    col_map = {}
    for ptm in ptm_list:
        target = norm(ptm)
        cands = [c for c in npy_like if target in norm(c)]
        if len(cands) == 1:
            col_map[ptm] = cands[0]
        elif len(cands) > 1:
            cands.sort(key=len); col_map[ptm] = cands[0]
        else:
            fallback = [c for c in npy_like if c not in col_map.values()]
            if not fallback:
                raise ValueError(f"Could not find .npy column for PTM '{ptm}' in {csv_path}.")
            col_map[ptm] = fallback[0]
    print("[ptm columns]", " | ".join([f"{k} -> {v}" for k, v in col_map.items()]))
    return col_map

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

def resolve_csv_with_fallback(p: str) -> str:
    path = Path(p)
    if path.exists():
        return str(path)
    tried = [str(path)]
    cand = []
    if path.suffix == ".csv":
        cand.append(path.with_suffix(".sizeok.csv"))
        cand.append(path.with_suffix(".clean.csv"))
    else:
        s = str(path)
        if s.endswith(".clean.csv"):
            cand.append(Path(s.replace(".clean.csv", ".sizeok.csv")))
            cand.append(Path(s.replace(".clean.csv", ".csv")))
        elif s.endswith(".sizeok.csv"):
            cand.append(Path(s.replace(".sizeok.csv", ".clean.csv")))
            cand.append(Path(s.replace(".sizeok.csv", ".csv")))
    if path.suffix != ".csv":
        base_csv = Path(str(path).replace(".clean.csv","").replace(".sizeok.csv",""))
        if not str(base_csv).endswith(".csv"):
            base_csv = base_csv.with_suffix(".csv")
        cand.append(base_csv)
    for c in cand:
        tried.append(str(c))
        if c.exists():
            print(f"[csv-resolve] '{p}' -> '{c}'")
            return str(c)
    raise FileNotFoundError(f"None of the candidate CSVs exist for '{p}'. Tried:\n  " + "\n  ".join(tried))

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

def make_loader(csv_path, ptms, ptm_cols, batch_size, shuffle, num_workers, balanced=False):
    ds = PTMDataset(csv_path, ptms, ptm_cols)
    if balanced:
        labels = to_int_labels(ds.df["label"])
        class_counts = np.bincount(labels, minlength=2).astype(np.float32)
        class_weights = class_counts.sum() / (2.0 * class_counts + 1e-9)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(
            ds, batch_size=batch_size, sampler=sampler, pin_memory=True,
            num_workers=num_workers, collate_fn=collate_fn,
            persistent_workers=(num_workers > 0), prefetch_factor=(2 if num_workers > 0 else None)
        )
    else:
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
            num_workers=num_workers, collate_fn=collate_fn,
            persistent_workers=(num_workers > 0), prefetch_factor=(2 if num_workers > 0 else None)
        )
    return ds, loader

# -------------------------
# Model (Same as v4)
# -------------------------
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
        out = self.head(h)
        return out

class TinyGate(nn.Module):
    def __init__(self, in_dim_concat, hidden=64, drop=0.15, n_experts=2, simple=False):
        super().__init__()
        if simple:
            self.net = nn.Sequential(
                nn.Linear(in_dim_concat, n_experts)
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim_concat),
                nn.Linear(in_dim_concat, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden, n_experts)
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
            self.fusion = nn.Sequential(
                nn.Linear(2, 2),
                nn.Dropout(fusion_dropout)  # HEAVY dropout
            )

    def forward(self, x_dict: Dict[str, torch.Tensor]):
        ptm_order = self.ptms
        xs = [x_dict[ptm] for ptm in ptm_order]
        x_concat = torch.cat(xs, dim=1)
        gate_w = self.gate(x_concat)
        
        if self.training and self.stochastic_depth < 1.0:
            keep_prob = torch.full((gate_w.shape[0], len(self.ptms)), self.stochastic_depth, 
                                   device=gate_w.device)
            mask = torch.bernoulli(keep_prob)
            gate_w = gate_w * mask
            gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-8)
        
        expert_logits = torch.stack([self.experts[p](x) for p, x in zip(ptm_order, xs)], dim=1)
        final_logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)
        
        if self.use_fusion:
            final_logits = self.fusion(final_logits)
        
        return final_logits, expert_logits, gate_w

def train_loop(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[env]", "device=", device, "| cuda_available=", torch.cuda.is_available(),
          "| gpu=", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), flush=True)

    set_seed(cfg["seed"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    cfg["train_csv"] = resolve_csv_with_fallback(cfg["train_csv"])
    cfg["val_csv"] = resolve_csv_with_fallback(cfg["val_csv"])
    cfg["test_real_csv"] = resolve_csv_with_fallback(cfg["test_real_csv"])
    cfg["test_fake_csv"] = resolve_csv_with_fallback(cfg["test_fake_csv"])

    PTM_COLUMNS = resolve_ptm_columns(cfg["train_csv"], cfg["ptms"])

    def label_counts(csvp):
        dfh = pd.read_csv(csvp, usecols=["label"])
        lab = to_int_labels(dfh["label"])
        return int((lab == 0).sum()), int((lab == 1).sum())
    tr_r, tr_f = label_counts(cfg["train_csv"])
    va_r, va_f = label_counts(cfg["val_csv"])
    te_r = len(pd.read_csv(cfg["test_real_csv"]))
    te_f = len(pd.read_csv(cfg["test_fake_csv"]))
    print("\n[split files in use]")
    print(f" train_csv     : {cfg['train_csv']}  | rows={tr_r+tr_f:,} (real={tr_r:,} fake={tr_f:,})")
    print(f" val_csv       : {cfg['val_csv']}  | rows={va_r+va_f:,} (real={va_r:,} fake={va_f:,})")
    print(f" test_real_csv : {cfg['test_real_csv']}  | rows={te_r:,}")
    print(f" test_fake_csv : {cfg['test_fake_csv']}  | rows={te_f:,}")
    print(f" combined test : {te_r+te_f:,} rows")

    train_ds, train_loader = make_loader(
        cfg["train_csv"], cfg["ptms"], PTM_COLUMNS, cfg["batch_size"],
        shuffle=not cfg["balanced_sampler"], num_workers=cfg["num_workers"],
        balanced=cfg["balanced_sampler"]
    )
    val_ds, val_loader = make_loader(
        cfg["val_csv"], cfg["ptms"], PTM_COLUMNS, cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], balanced=False
    )
    test_real_ds, test_real_loader = make_loader(
        cfg["test_real_csv"], cfg["ptms"], PTM_COLUMNS, cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], balanced=False
    )
    test_fake_ds, test_fake_loader = make_loader(
        cfg["test_fake_csv"], cfg["ptms"], PTM_COLUMNS, cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], balanced=False
    )

    def count_labels(ds):
        lv = to_int_labels(ds.df["label"])
        return int((lv==0).sum()), int((lv==1).sum())
    tr_r2, tr_f2 = count_labels(train_ds)
    va_r2, va_f2 = count_labels(val_ds)
    print(f"[data] train={len(train_ds)} (real={tr_r2}, fake={tr_f2}) | val={len(val_ds)} (real={va_r2}, fake={va_f2}) | test={len(test_real_ds)+len(test_fake_ds)}")
    
    print(f"\n[v5 config] Mixup={cfg['mixup_alpha']:.2f} (prob={cfg['mixup_prob']}) | StochDepth={cfg['stochastic_depth']} | FusionDrop={cfg['fusion_dropout']} | LabelSmooth={cfg['label_smoothing']}")

    model = MoEModel(
        ptms=cfg["ptms"],
        in_dim_each=cfg["ptm_dim"],
        expert_bottleneck=cfg["expert_bottleneck"],
        expert_drop=cfg["expert_dropout"],
        gate_hidden=cfg["gate_hidden"],
        gate_drop=cfg["gate_dropout"],
        use_batchnorm=cfg["use_batchnorm"],
        use_se=cfg["use_se"],
        simple_gate=cfg["simple_gate"],
        stochastic_depth=cfg["stochastic_depth"],
        use_fusion=cfg["use_fusion"],
        fusion_dropout=cfg["fusion_dropout"]
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=cfg["betas"])
    total_steps = cfg["max_epochs"] * max(1, math.ceil(len(train_ds)/cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=make_warmup_cosine(total_steps, cfg["warmup_ratio"], cfg["lr"], cfg["cosine_final_lr"]))
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = GradScaler(enabled=(device=="cuda") and cfg["amp"])

    best_val_eer = 1.0
    epochs_no_improve = 0
    best_ckpt = Path(cfg["ckpt_dir"]) / f"{cfg['run_name']}_best.pt"
    last_ckpt = Path(cfg["ckpt_dir"]) / f"{cfg['run_name']}_last.pt"

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        running_loss = 0.0; running_acc = 0.0; n_batches = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"epoch {epoch:02d}", leave=False)
        for batch in pbar:
            if batch is None or batch["y"].numel() == 0:
                continue
            xs = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
            y  = batch["y"].to(device, non_blocking=True)

            # NEW: Mixup augmentation
            use_mixup = (random.random() < cfg["mixup_prob"]) and (cfg["mixup_alpha"] > 0)
            if use_mixup:
                xs, y_a, y_b, lam = mixup_data(xs, y, cfg["mixup_alpha"])

            optim.zero_grad(set_to_none=True)
            with amp_ctx(device, enabled=cfg["amp"]):
                logits, expert_logits, gate_w = model(xs)
                
                if use_mixup:
                    main_loss = mixup_criterion(ce, logits, y_a, y_b, lam)
                else:
                    main_loss = ce(logits, y)

                aux_loss = 0.0
                if cfg["aux_loss_lambda"] > 0.0:
                    B, E, C = expert_logits.shape
                    aux = 0.0
                    for e in range(E):
                        if use_mixup:
                            aux += mixup_criterion(ce, expert_logits[:, e, :], y_a, y_b, lam)
                        else:
                            aux += ce(expert_logits[:, e, :], y)
                    aux_loss = cfg["aux_loss_lambda"] * (aux / E)

                balance_loss = 0.0
                if cfg["gate_balance_lambda"] > 0.0:
                    usage = gate_w.mean(dim=0)
                    balance_loss = usage.var() * cfg["gate_balance_lambda"]

                loss = main_loss + aux_loss + balance_loss

            scaler.scale(loss).backward()
            if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optim)
            scaler.update()
            sched.step()

            running_loss += loss.item()
            if not use_mixup:
                running_acc  += accuracy_from_logits(logits, y)
            else:
                # For mixup, use approximate accuracy with y_a
                running_acc  += accuracy_from_logits(logits, y_a)
            n_batches += 1
            if not cfg["quiet"]:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(running_acc/max(1,n_batches)):.4f}")

        val_scores, val_labels = compute_scores_and_labels(model, val_loader, device)
        val_eer, val_thr = eer_from_scores(val_scores, val_labels)

        model.eval()
        v_acc_num = 0; v_acc_den = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch["y"].numel()==0: continue
                xs = {k: v.to(device) for k, v in batch["x"].items()}
                y  = batch["y"].to(device)
                logits, _, _ = model(xs)
                v_acc_num += (torch.argmax(logits,1) == y).sum().item()
                v_acc_den += y.numel()
        val_acc = v_acc_num / max(1, v_acc_den)

        train_loss = running_loss / max(1, n_batches)
        train_acc  = running_acc  / max(1, n_batches)
        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_eer {val_eer:.4f} | val_acc {val_acc:.4f}")

        improved = val_eer < best_val_eer - 1e-5
        if improved:
            best_val_eer = val_eer
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "best_val_eer": best_val_eer}, best_ckpt)
        else:
            epochs_no_improve += 1

        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "best_val_eer": best_val_eer}, last_ckpt)
        if epochs_no_improve >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}. Best val EER={best_val_eer:.4f}")
            break

    state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    test_scores_real, test_labels_real = compute_scores_and_labels(model, test_real_loader, device)
    test_scores_fake, test_labels_fake = compute_scores_and_labels(model, test_fake_loader, device)
    test_scores = np.concatenate([test_scores_real, test_scores_fake])
    test_labels = np.concatenate([test_labels_real, test_labels_fake])

    test_eer, thr_eer = eer_from_scores(test_scores, test_labels)
    acc05 = float(((test_scores >= 0.5).astype(np.int32) == test_labels).mean())
    acc_eer = float(((test_scores >= thr_eer).astype(np.int32) == test_labels).mean())
    err05 = 1.0 - acc05
    err_eerthr = 1.0 - acc_eer

    print("\n[RESULTS]")
    print(f" Best Val EER : {best_val_eer:.4f}")
    print(f" Test EER     : {test_eer:.4f} (thr@eer={thr_eer:.6f})")
    print(f" Acc@0.5      : {acc05:.4f} | Error@0.5     : {err05:.4f}")
    print(f" Acc@thrEER   : {acc_eer:.4f} | Error@thrEER : {err_eerthr:.4f}")

    calib_out = {}
    if cfg["calibrate"] and len(test_scores) > 0:
        rng = np.random.default_rng(cfg["seed"])
        pos_idx = np.where(test_labels == 1)[0]
        neg_idx = np.where(test_labels == 0)[0]
        n_pos = max(1, int(len(pos_idx)*cfg["calib_frac"]))
        n_neg = max(1, int(len(neg_idx)*cfg["calib_frac"]))
        dev_idx = set(rng.choice(pos_idx, n_pos, replace=False).tolist() +
                      rng.choice(neg_idx, n_neg, replace=False).tolist())
        hold_idx = [i for i in range(len(test_labels)) if i not in dev_idx]

        dev_scores = test_scores[list(dev_idx)]; dev_labels = test_labels[list(dev_idx)]
        hold_scores = test_scores[hold_idx];     hold_labels = test_labels[hold_idx]

        uniq = np.unique(dev_scores)
        grid = np.linspace(0.0, 1.0, 401)
        thr_grid = np.unique(np.concatenate([uniq, grid]))
        best_thr = 0.5; best_acc = -1.0
        for t in thr_grid:
            a = float(((dev_scores >= t).astype(np.int32) == dev_labels).mean())
            if a > best_acc: best_acc, best_thr = a, t

        acc_hold = float(((hold_scores >= best_thr).astype(np.int32) == hold_labels).mean())
        err_hold = 1.0 - acc_hold
        print("\n[MMS calibration (on combined test)]")
        print(f"  dev_frac={cfg['calib_frac']:.2f} | dev_n={len(dev_scores)} | holdout_n={len(hold_scores)}")
        print(f"  Calibrated thr={best_thr:.6f} (dev best acc={best_acc:.4f})")
        print(f"  Holdout Acc   ={acc_hold:.4f} | Holdout Error ={err_hold:.4f}")

        calib_out = {
            "dev_frac": float(cfg["calib_frac"]),
            "dev_n": int(len(dev_scores)),
            "holdout_n": int(len(hold_scores)),
            "calibrated_thr": float(best_thr),
            "dev_best_acc": float(best_acc),
            "holdout_acc": float(acc_hold),
            "holdout_error": float(err_hold),
        }

    run_manifest = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_val_eer": float(best_val_eer),
        "test_eer": float(test_eer),
        "thr_at_eer": float(thr_eer),
        "acc_at_0p5": float(acc05),
        "err_at_0p5": float(err05),
        "acc_at_val_eer": float(acc_eer),
        "err_at_val_eer": float(err_eerthr),
        "calibration": calib_out,
        "cfg": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in cfg.items()},
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": int(len(test_real_ds) + len(test_fake_ds)),
    }
    runs_dir = META / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / f"{cfg['run_name']}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str, default=DEFAULTS["train_csv"])
    ap.add_argument("--val-csv",   type=str, default=DEFAULTS["val_csv"])
    ap.add_argument("--test-real-csv", type=str, default=DEFAULTS["test_real_csv"])
    ap.add_argument("--test-fake-csv", type=str, default=DEFAULTS["test_fake_csv"])
    ap.add_argument("--ptms", type=str, default=",".join(DEFAULTS["ptms"]))
    ap.add_argument("--ptm-dim", type=int, default=DEFAULTS["ptm_dim"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    ap.add_argument("--balanced-sampler", dest="balanced_sampler", action="store_true", default=DEFAULTS["balanced_sampler"])
    ap.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    ap.add_argument("--expert-bottleneck", type=int, default=DEFAULTS["expert_bottleneck"])
    ap.add_argument("--expert-dropout", type=float, default=DEFAULTS["expert_dropout"])
    ap.add_argument("--gate-hidden", type=int, default=DEFAULTS["gate_hidden"])
    ap.add_argument("--gate-dropout", type=float, default=DEFAULTS["gate_dropout"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--betas", type=str, default=f"{DEFAULTS['betas'][0]},{DEFAULTS['betas'][1]}")
    ap.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    ap.add_argument("--cosine-final-lr", type=float, default=DEFAULTS["cosine_final_lr"])
    ap.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--label-smoothing", type=float, default=DEFAULTS["label_smoothing"])
    ap.add_argument("--aux-loss-lambda", type=float, default=DEFAULTS["aux_loss_lambda"])
    ap.add_argument("--gate-balance-lambda", type=float, default=DEFAULTS["gate_balance_lambda"])
    ap.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    ap.add_argument("--amp", dest="amp", action="store_true", default=DEFAULTS["amp"])
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--ckpt-dir", type=str, default=DEFAULTS["ckpt_dir"])
    ap.add_argument("--run-name", type=str, default=DEFAULTS["run_name"])
    ap.add_argument("--quiet", dest="quiet", action="store_true", default=DEFAULTS["quiet"])
    ap.add_argument("--calibrate", dest="calibrate", action="store_true", default=DEFAULTS["calibrate"])
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--calib-frac", type=float, default=DEFAULTS["calib_frac"])
    ap.add_argument("--use-batchnorm", dest="use_batchnorm", action="store_true", default=DEFAULTS["use_batchnorm"])
    ap.add_argument("--no-batchnorm", dest="use_batchnorm", action="store_false")
    ap.add_argument("--use-se", dest="use_se", action="store_true", default=DEFAULTS["use_se"])
    ap.add_argument("--no-se", dest="use_se", action="store_false")
    ap.add_argument("--simple-gate", dest="simple_gate", action="store_true", default=DEFAULTS["simple_gate"])
    ap.add_argument("--complex-gate", dest="simple_gate", action="store_false")
    ap.add_argument("--stochastic-depth", type=float, default=DEFAULTS["stochastic_depth"])
    ap.add_argument("--use-fusion", dest="use_fusion", action="store_true", default=DEFAULTS["use_fusion"])
    ap.add_argument("--no-fusion", dest="use_fusion", action="store_false")
    ap.add_argument("--fusion-dropout", type=float, default=DEFAULTS["fusion_dropout"])
    ap.add_argument("--mixup-alpha", type=float, default=DEFAULTS["mixup_alpha"])
    ap.add_argument("--mixup-prob", type=float, default=DEFAULTS["mixup_prob"])
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = DEFAULTS.copy()
    cfg.update({
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_real_csv": args.test_real_csv,
        "test_fake_csv": args.test_fake_csv,
        "ptms": [s.strip() for s in args.ptms.split(",") if s.strip()],
        "ptm_dim": args.ptm_dim,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "balanced_sampler": args.balanced_sampler,
        "expert_bottleneck": args.expert_bottleneck,
        "expert_dropout": args.expert_dropout,
        "gate_hidden": args.gate_hidden,
        "gate_dropout": args.gate_dropout,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": tuple(map(float, args.betas.split(","))),
        "warmup_ratio": args.warmup_ratio,
        "cosine_final_lr": args.cosine_final_lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "label_smoothing": args.label_smoothing,
        "aux_loss_lambda": args.aux_loss_lambda,
        "gate_balance_lambda": args.gate_balance_lambda,
        "grad_clip": args.grad_clip,
        "amp": args.amp,
        "ckpt_dir": args.ckpt_dir,
        "run_name": args.run_name,
        "quiet": args.quiet,
        "calibrate": args.calibrate,
        "calib_frac": args.calib_frac,
        "use_batchnorm": args.use_batchnorm,
        "use_se": args.use_se,
        "simple_gate": args.simple_gate,
        "stochastic_depth": args.stochastic_depth,
        "use_fusion": args.use_fusion,
        "fusion_dropout": args.fusion_dropout,
        "mixup_alpha": args.mixup_alpha,
        "mixup_prob": args.mixup_prob,
    })
    set_seed(cfg["seed"])
    train_loop(cfg)

if __name__ == "__main__":
    main()