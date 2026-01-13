# app.py
# MUST BE FIRST - Setup environment variables
import env_setup  # Sets HF_HUB_DISABLE_SYMLINKS before any HF imports

import os, time, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import torch
import pandas as pd
from typing import Dict, Tuple

from audio import load_audio_mono_16k, SampleRateError
from ptm_feat import extract_both_with_debug, DEVICE
from moe_model import MoEModel
from lid_gate import load_lid_model, check_language
from xai_analysis import run_complete_xai_analysis
from xai_advanced import run_advanced_xai
from preprocess_strong import preprocess_strong_from_path
from export_xai_plots import generate_xai_plots

# =========================
# ENV / PATHS
# =========================
# IMPORTANT: Use your real project root & HF cache path (outside backend)
#   ROOT = G:\My Drive\hindi_dfake
#   HF cache = G:\My Drive\hindi_dfake\models\hf-cache
ROOT = Path(r"G:\My Drive\hindi_dfake")
META = ROOT / "metadata"

# Pick your checkpoint (stays local to backend/checkpoints)
CKPT_PATH = Path(os.environ.get("CKPT_PATH", "checkpoints/moe_ptm2_v5_aggressive_best.pt"))

# Test CSV used for ground-truth lookup (must be the SAME one you used to evaluate)
# You can override with TEST_CSV in .env
TEST_CSV = Path(os.environ.get(
    "TEST_CSV",
    str(META / "tests" / "test_mms.strong.ptm2.csv")
))

# Thresholds (env overrides allowed)
DEFAULT_THR      = 0.1059
MMS_FALLBACK_THR = 0.1059

# Optional calibration json (if present we use its calibrated thr)
CALIB_JSON = Path(os.environ.get(
    "CALIB_JSON",
    str(META / "runs" / "mms_calibration.json")
))

# Force MMS behavior (always use MMS thr) while testing
FORCE_MMS = os.environ.get("FORCE_MMS", "0") == "1"

# Build tag
APP_BUILD = os.environ.get("APP_BUILD", "v2-thr")

# =========================
# FastAPI
# =========================
app = FastAPI(title="Voice Deepfake Detector (MoE)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# =========================
# Load checkpoint & model
# =========================
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE if DEVICE=="cuda" else "cpu")
cfg = ckpt.get("cfg", {})
ptms = cfg.get("ptms", ["wav2vec2-base","hubert-base"])
in_dim = int(cfg.get("ptm_dim", 1536))
expert_bottleneck = int(cfg.get("expert_bottleneck", 768))
expert_drop   = float(cfg.get("expert_dropout", 0.3))
gate_hidden   = int(cfg.get("gate_hidden", 64))
gate_drop     = float(cfg.get("gate_dropout", 0.15))
use_batchnorm = bool(cfg.get("use_batchnorm", True))
use_se        = bool(cfg.get("use_se", False))
simple_gate   = bool(cfg.get("simple_gate", True))
stochastic_depth = float(cfg.get("stochastic_depth", 0.6))
use_fusion    = bool(cfg.get("use_fusion", True))
fusion_dropout = float(cfg.get("fusion_dropout", 0.5))
run_name      = cfg.get("run_name", "unknown")
best_val_eer  = float(ckpt.get("best_val_eer", -1))
start_epoch   = int(ckpt.get("epoch", -1))

model = MoEModel(ptms, in_dim, expert_bottleneck, expert_drop, gate_hidden, gate_drop,
                 use_batchnorm, use_se, simple_gate, stochastic_depth, use_fusion, fusion_dropout)
model.load_state_dict(ckpt["model"])
model.to(DEVICE).eval()

# Optional warmup (avoids first-request cold start)
with torch.inference_mode():
    xzero = {p: torch.zeros(1, in_dim, device=DEVICE, dtype=torch.float32) for p in ptms}
    model(xzero)

# =========================
# Load LID model (faster-whisper)
# =========================
# Use "base" for good accuracy (~70-150ms), runs on CUDA
LID_MODEL_SIZE = os.environ.get("LID_MODEL_SIZE", "base")
LID_DEVICE = DEVICE  # Use same device as MoE model
LID_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

load_lid_model(model_size=LID_MODEL_SIZE, device=LID_DEVICE, compute_type=LID_COMPUTE_TYPE)

# =========================
# Calibration (MMS)
# =========================
MMS_CAL_THR = None
MMS_CAL_OK = False
if CALIB_JSON.exists():
    try:
        j = json.loads(Path(CALIB_JSON).read_text(encoding="utf-8"))
        ckpt_in_json = str(j.get("ckpt", ""))
        if Path(ckpt_in_json).name == CKPT_PATH.name:
            MMS_CAL_THR = float(j.get("calibrated_thr", MMS_FALLBACK_THR))
            MMS_CAL_OK = True
        else:
            MMS_CAL_THR = MMS_FALLBACK_THR
    except Exception:
        MMS_CAL_THR = MMS_FALLBACK_THR
else:
    MMS_CAL_THR = MMS_FALLBACK_THR

def pick_threshold(filename_stem: str) -> Tuple[float, str]:
    """
    For MMS test files we want the calibrated (or fallback) MMS threshold.
    If FORCE_MMS=1, always use MMS thr for consistency during testing.
    """
    if FORCE_MMS:
        return MMS_CAL_THR, "mms_calibrated" if MMS_CAL_OK else "mms_fallback"
    # Heuristic: MMS files start with 'tts_mms_' in your test set.
    if filename_stem.startswith("tts_mms_"):
        return MMS_CAL_THR, "mms_calibrated" if MMS_CAL_OK else "mms_fallback"
    return DEFAULT_THR, "default"

# =========================
# Fuzzy uncertainty helper (inference only)
# =========================
def fuzzy_uncertainty(score: float, threshold: float, width_real: float = 0.02, width_fake: float = 0.0381) -> float:
    """
    Returns an uncertainty value in [0,1] based on distance from the decision threshold.
    Asymmetric band: narrower on real side, wider on fake side.
    """
    try:
        s = float(score)
        thr = float(threshold)
    except Exception:
        return 0.0
    if s < thr:
        d = thr - s
        u = max(0.0, 1.0 - d / float(width_real))
    else:
        d = s - thr
        u = max(0.0, 1.0 - d / float(width_fake))
    return float(min(u, 1.0))

# =========================
# Ground truth index (from TEST_CSV)
# =========================
_truth_index: Dict[str, int] = {}
_truth_mtime: float = 0.0

# XAI cache (stores features & wav for XAI endpoint)
_xai_cache: Dict[str, Dict] = {}

def _stem(path_like: str) -> str:
    return Path(str(path_like)).stem

def _load_truth_index(csv_path: Path) -> Dict[str, int]:
    """
    Build stem -> label map from the test CSV you used in evaluation.
    Works whether CSV has 'path_audio' OR only vec columns (.npy) with same stems.
    Accepts label in {0,1} or {real,fake}.
    """
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)

    # Get stems
    stems = set()
    if "path_audio" in df.columns:
        stems |= set(df["path_audio"].dropna().astype(str).map(_stem))

    # If no path_audio, infer stems from any .npy vec columns
    def is_npy_col(c):
        try:
            vals = df[c].dropna().astype(str)
            return len(vals) > 0 and (vals.str.endswith(".npy").mean() > 0.6)
        except Exception:
            return False

    npy_cols = [c for c in df.columns if is_npy_col(c)]
    for c in npy_cols:
        stems |= set(df[c].dropna().astype(str).map(_stem))

    # Resolve label column: numeric (0/1) or string (real/fake)
    label_col = None
    if "label" in df.columns:
        label_col = "label"
    else:
        # Try common variants
        for cand in ["y", "target", "is_fake"]:
            if cand in df.columns:
                label_col = cand
                break
    if label_col is None:
        # No label column -> can't build truth map, but keep stems set
        # (frontend will still show "—")
        return {}

    # Normalize to 0/1
    lab = df[label_col]
    if lab.dtype == object:
        lab_norm = lab.astype(str).str.strip().str.lower().map(
            {"fake": 1, "real": 0}
        )
    else:
        lab_norm = pd.to_numeric(lab, errors="coerce").fillna(-1).astype(int)

    # Prefer path_audio mapping; else map via first npy column we found
    key_series = None
    if "path_audio" in df.columns:
        key_series = df["path_audio"].astype(str)
    elif npy_cols:
        key_series = df[npy_cols[0]].astype(str)
    else:
        return {}

    idx = {}
    for k, v in zip(key_series, lab_norm):
        st = _stem(k)
        if v in (0, 1):
            idx[st] = int(v)
    return idx

def _ensure_truth_index():
    global _truth_index, _truth_mtime
    try:
        m = TEST_CSV.stat().st_mtime if TEST_CSV.exists() else 0.0
    except Exception:
        m = 0.0
    if (not _truth_index) or (m != _truth_mtime):
        _truth_index = _load_truth_index(TEST_CSV)
        _truth_mtime = m

def get_truth_for(stem: str) -> Tuple[int, str]:
    """
    Returns (label_int_or_-1, label_str_or_empty)
    """
    _ensure_truth_index()
    if not _truth_index:
        return -1, ""
    v = _truth_index.get(stem, -1)
    if v == 0: return 0, "REAL"
    if v == 1: return 1, "FAKE"
    return -1, ""

# =========================
# Helper for XAI feature extraction
# =========================
def _extract_features_for_segment(wav_segment: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract features from an audio segment for XAI analysis.
    
    NOTE: For temporal_heatmap, segments are already from preprocessed audio,
    so we don't re-apply RawBoost/EQ (that would be double-preprocessing).
    We just extract PTM features directly.
    """
    feats, _ = extract_both_with_debug(wav_segment)
    return feats

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    _ensure_truth_index()
    return {
        "status": "ok",
        "build": APP_BUILD,
        "device": DEVICE,
        "checkpoint": str(CKPT_PATH),
        "run_name": run_name,
        "best_val_eer": best_val_eer,
        "epoch": start_epoch,
        "ptms": ptms,
        "ptm_dim": in_dim,
        "hf_home": str(ROOT / "models" / "hf-cache"),
        "test_csv": str(TEST_CSV),
        "truth_index_size": len(_truth_index),
        "calibration": {
            "file": str(CALIB_JSON),
            "mms_cal_thr": MMS_CAL_THR,
            "mms_fallback_thr": MMS_FALLBACK_THR,
            "default_thr": DEFAULT_THR,
        },
        "force_mms": FORCE_MMS,
        "lid": {
            "model_size": LID_MODEL_SIZE,
            "device": LID_DEVICE,
            "compute_type": LID_COMPUTE_TYPE,
            "engine": "faster-whisper"
        }
    }

@app.post("/infer")
def infer(file: UploadFile = File(...)):
    raw = file.file.read()
    t0 = time.perf_counter()
    
    # =========================
    # STEP 1: Load CLEAN audio for LID (no preprocessing yet)
    # =========================
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    
    # Read clean audio (just resample to 16k, no augmentation)
    try:
        import subprocess
        tmp_clean = tmp_path.with_suffix(".clean.wav")
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(tmp_path), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(tmp_clean)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p.returncode != 0:
            raise RuntimeError(f"Audio load failed: {p.stderr.decode('utf-8', 'ignore')}")
        
        import soundfile as sf
        wav_clean, sr_clean = sf.read(str(tmp_clean), dtype="float32", always_2d=False)
        tmp_clean.unlink(missing_ok=True)
    except Exception as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except:
            pass
        return {"error": f"Audio load failed: {str(e)}", "hint": "Ensure audio is a valid format (WAV, MP3, etc.)"}
    
    # =========================
    # STEP 2: LANGUAGE GATE - Run on CLEAN audio (before preprocessing)
    # =========================
    t_lid_start = time.perf_counter()
    lid_result = check_language(wav_clean, sr_clean)
    lid_time_ms = int((time.perf_counter() - t_lid_start) * 1000)
    
    # If language gate rejects, return immediately (MoE skipped)
    if lid_result["gate"] == "rejected":
        fname = getattr(file, "filename", "") or ""
        print(
            f"[app] [infer] REJECTED by LID gate | file={fname} | "
            f"detected={lid_result['detected_lang']} | p_hi={lid_result['p_hi']:.4f} | "
            f"speech_frac={lid_result['speech_fraction']:.4f} | {lid_result['message']}",
            flush=True
        )
        return {
            "error_code": "not_hindi",
            "message": f"Audio is not confidently Hindi. {lid_result['message']}",
            "lid_debug": {
                "detected_lang": lid_result["detected_lang"],
                "p_hi": lid_result["p_hi"],
                "top3": lid_result["top3"],
                "speech_fraction": lid_result["speech_fraction"],
                "lid_engine": lid_result["lid_engine"],
                "t_lid_ms": lid_time_ms
            }
        }
    
    # =========================
    # STEP 3: Apply preprocessing for MoE (AFTER LID passes)
    # =========================
    # Language gate passed - now apply strong preprocessing for MoE
    try:
        wav, sr, dbg_pre = preprocess_strong_from_path(tmp_path)
    except Exception as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except:
            pass
        return {"error": f"Preprocessing failed: {str(e)}", "hint": "Audio preprocessing error."}
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except:
            pass
    
    # Extract PTM features from PREPROCESSED audio
    feats, dbg = extract_both_with_debug(wav)
    dbg.update(dbg_pre)  # Add preprocessing times
    dbg["t_lid_ms"] = lid_time_ms

    # Model forward
    xdict = {k: torch.from_numpy(v)[None, :].to(DEVICE) for k, v in feats.items()}
    with torch.inference_mode():
        t = time.perf_counter()
        logits, expert_logits, gates = model(xdict)
        dbg["t_moe_ms"] = int((time.perf_counter() - t) * 1000)
        probs = torch.softmax(logits, dim=1)
        p_fake = float(probs[0, 1].item())

    # File stem for truth/threshold lookup
    fname = getattr(file, "filename", "") or ""
    stem  = Path(fname).stem
    print(f"[app] [infer] file={fname} | p_fake={p_fake:.4f}", flush=True)

    thr, thr_src = pick_threshold(stem)
    label = int(p_fake >= thr)
    gate_w = {p: float(gates[0, i].item()) for i, p in enumerate(ptms)}

    # Fuzzy uncertainty (asymmetric band around threshold)
    u = fuzzy_uncertainty(p_fake, thr)
    if u >= 0.7:
        uncertainty_level = "high"
    elif u >= 0.3:
        uncertainty_level = "medium"
    else:
        uncertainty_level = "low"

    # Ground truth lookup
    truth_int, truth_str = get_truth_for(stem)
    correct = (truth_int in (0,1)) and (label == truth_int)

    dbg["t_overall_ms"] = int((time.perf_counter() - t0) * 1000)
    dbg["audio_sec"] = round(len(wav) / 16000, 3)

    # Cache features and wav for XAI endpoint (keyed by filename)
    _xai_cache[stem] = {
        'wav': wav,
        'feats': feats,
        'p_fake': p_fake,
        'timestamp': time.time()
    }

    # Log one tight line
    print(
        f"[app] [infer] file={fname} | stem={stem} | test_csv={TEST_CSV} | "
        f"index_keys={len(_truth_index)} | hit={truth_int in (0,1)} | truth={truth_str or '—'} | "
        f"thr={thr:.6f} ({thr_src}) | pred={'FAKE' if label==1 else 'REAL'} | p_fake={p_fake:.6f} | "
        f"LID: {lid_result['detected_lang']} (p_hi={lid_result['p_hi']:.4f})",
        flush=True
    )

    return {
        "prob_fake": p_fake,
        "label": label,  # 1=fake, 0=real
        "threshold_used": thr,
        "threshold_source": thr_src,
        "uncertainty": float(u),
        "uncertainty_level": uncertainty_level,
        "gate": gate_w,
        "language_check": {
            "lid_engine": lid_result["lid_engine"],
            "detected_lang": lid_result["detected_lang"],
            "p_hi": lid_result["p_hi"],
            "speech_fraction": lid_result["speech_fraction"],
            "gate": lid_result["gate"],
            "message": lid_result["message"],
            "t_lid_ms": lid_time_ms
        },
        "meta": {
            "device": DEVICE,
            "checkpoint": str(CKPT_PATH),
            "run_name": run_name,
            "best_val_eer": best_val_eer,
            "epoch": start_epoch,
        },
        "debug": dbg,
        "truth_label": truth_int if truth_int in (0,1) else None,
        "truth_label_str": truth_str or "—",
        "correct": correct if truth_int in (0,1) else None,
        "match_key": stem,
        "index_hit": (truth_int in (0,1)),
        "xai_cached": True
    }


@app.post("/xai")
def generate_xai(file: UploadFile = File(...)):
    """Generate XAI explanations reusing cached features."""
    fname = getattr(file, "filename", "") or ""
    stem = Path(fname).stem
    
    if stem not in _xai_cache:
        return {"error": "Audio not in cache. Run /infer first.", "stem": stem}
    
    cache_entry = _xai_cache[stem]
    if time.time() - cache_entry['timestamp'] > 300:
        del _xai_cache[stem]
        return {"error": "Cache expired. Run /infer again.", "stem": stem}
    
    wav = cache_entry['wav']
    feats = cache_entry['feats']
    
    print(f"[app] [XAI] Generating for {stem}...", flush=True)
    
    basic_xai = None
    try:
        basic_xai = run_complete_xai_analysis(
            model=model, wav=wav, sr=16000, feats=feats, device=DEVICE,
            feature_extractor_fn=_extract_features_for_segment, ptm_models=None
        )
    except Exception as e:
        basic_xai = {"error": str(e)}
    
    advanced_xai = None
    try:
        advanced_xai = run_advanced_xai(model=model, feats=feats, device=DEVICE)
    except Exception as e:
        advanced_xai = {"error": str(e)}
    
    total_time = (basic_xai.get('processing_time_ms', 0) + 
                  advanced_xai.get('processing_time_ms', 0))

    # Cache XAI results for export endpoint (does not change existing API fields)
    cache_entry["xai"] = {
        "basic_xai": basic_xai,
        "advanced_xai": advanced_xai,
    }
    
    print(f"[app] [XAI] Complete | {total_time}ms", flush=True)
    
    return {
        "stem": stem,
        "basic_xai": basic_xai,
        "advanced_xai": advanced_xai,
        "processing_time_ms": total_time
    }


@app.post("/xai/export")
def export_xai(file: UploadFile = File(...)):
    """Export XAI plots as a ZIP using existing cached XAI results.

    Requires that /infer and then /xai have already been called for this file.
    """
    fname = getattr(file, "filename", "") or ""
    stem = Path(fname).stem

    if stem not in _xai_cache:
        return {"error": "Audio not in cache. Run /infer first.", "stem": stem}

    cache_entry = _xai_cache[stem]
    # Reuse the same expiry policy as /xai
    if time.time() - cache_entry['timestamp'] > 300:
        del _xai_cache[stem]
        return {"error": "Cache expired. Run /infer again.", "stem": stem}

    xai_saved = cache_entry.get("xai")
    if not xai_saved:
        return {"error": "XAI not computed yet. Run /xai first.", "stem": stem}

    basic_xai = xai_saved.get("basic_xai") or {}
    advanced_xai = xai_saved.get("advanced_xai") or {}

    zip_path = generate_xai_plots(stem, basic_xai, advanced_xai)

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=zip_path.name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
