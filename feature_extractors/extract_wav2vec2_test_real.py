# ===== Wav2Vec2-Base — FULL-SCAN runner (FLEURS-only; resumable; strong-only; no CSV edits) =====
from pathlib import Path
import os, csv, time, gc, sys, signal
import numpy as np, pandas as pd
from tqdm import tqdm

# ---------- CONFIG ----------
ROOT        = Path(r"G:\My Drive\hindi_dfake")   # <— change to "/content/drive/MyDrive/hindi_dfake" on Colab
PTM_NAME    = "wav2vec2-base"
MODEL_ID    = "facebook/wav2vec2-base"
TARGET_SR   = 16000
SAVE_DTYPE  = "float16"
DRY_RUN     = False
PROFILE     = "strong"                           # extract from processed/wav/strong
LAST_K      = 1                                  # match your default pooling

# Optional cap per run (None = process all pending this session)
CAP = int(os.environ.get("PTM_CAP", "0")) or None

# ---------- Fixed paths ----------
PROC_DIR   = ROOT / "processed"
WAV_DIR    = PROC_DIR / "wav"
FEAT_DIR   = PROC_DIR / "features" / "ptm"
META_DIR   = ROOT / "metadata"
MANIFEST_IN  = META_DIR / "features_manifest.csv"               # read-only (skip logic)
MANIFEST_OUT = META_DIR / "features_manifest.fullscan.csv"      # write here (safe copy)
JOBS_DIR   = META_DIR / "ptm_jobs"
FLEURS_CSV = META_DIR / "thirdparty_real_test.fleurs.csv"       # scope to FLEURS only
TODO_CSV   = JOBS_DIR / f"{PTM_NAME}.FLEURS.{PROFILE}.todo.csv"
HF_CACHE   = ROOT / "models" / "hf-cache"

for d in (FEAT_DIR, META_DIR, JOBS_DIR, HF_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ---------- PATH SHIM ----------
_OLD_ROOTS = [
    "/content/drive/MyDrive/hindi_dfake",
    "C:/content/drive/MyDrive/hindi_dfake",
    "C:\\content\\drive\\MyDrive\\hindi_dfake",
    "G:/My Drive/hindi_dfake",
    "G:\\My Drive\\hindi_dfake",
]
def _norm(s: str) -> str:
    return str(s).replace("\\", "/")

def normalize_to_root(path_str: str) -> Path:
    s = _norm(path_str)
    if _norm(str(ROOT)) in s:
        return Path(s)
    for old in _OLD_ROOTS:
        o = _norm(old)
        if s.startswith(o):
            rel = s[len(o):].lstrip("/")
            return ROOT / Path(rel)
    return ROOT / Path(path_str).name

# processed STRONG path for any original WAV
def processed_strong_path_for(orig_path: str) -> Path:
    pa = normalize_to_root(orig_path)
    try:
        rel = pa.resolve().relative_to(ROOT.resolve())
    except Exception:
        rel = Path(pa.name)
    out = (WAV_DIR / PROFILE / rel).with_suffix(".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def vec_path_for(ptm_name: str, path_audio, profile: str=PROFILE) -> Path:
    pa = Path(path_audio)
    base = (WAV_DIR / profile).resolve()
    try:
        tail = pa.resolve().relative_to(base)
    except Exception:
        s = _norm(str(pa))
        needle = f"/wav/{profile}/"
        i = s.lower().find(needle)
        tail = Path(s[i + len(needle):]) if i != -1 else Path(pa.name)
    out = FEAT_DIR / ptm_name / tail.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

# ---------- Device ----------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(1)
else:
    torch.set_float32_matmul_precision("high")
print(f"[env] DEVICE={DEVICE}")

# ---------- Sanity checks ----------
def sanity():
    for p in (ROOT, PROC_DIR, WAV_DIR, META_DIR):
        if not p.exists(): raise FileNotFoundError(f"Missing folder: {p}")
    if not FLEURS_CSV.exists():
        raise FileNotFoundError(f"Missing FLEURS CSV: {FLEURS_CSV}")
    for pkg in ("torch","numpy","pandas","soundfile","transformers"):
        __import__(pkg)
sanity()

# ---------- IO helpers ----------
def load_pcm16_mono(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != TARGET_SR: raise RuntimeError(f"{path}: expected {TARGET_SR} Hz, got {sr}")
    if hasattr(x, "ndim") and x.ndim > 1: x = x.mean(axis=1)
    np.clip(x, -1.0, 1.0, out=x)
    return x

def _ensure_manifest_header(pth: Path):
    if not pth.exists() and not DRY_RUN:
        with open(pth, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["path_audio","profile","ptm","vec_path","dim","seconds"])

def append_manifest_rows(rows):
    if DRY_RUN or not rows: return
    _ensure_manifest_header(MANIFEST_OUT)
    with open(MANIFEST_OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

# ---------- Build TODO from FLEURS-only STRONG paths ----------
WAV_DIR_STRONG = (WAV_DIR / PROFILE)
FEAT_DIR_PTM   = (FEAT_DIR / PTM_NAME)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

fl = pd.read_csv(FLEURS_CSV)
fl_paths = fl["path"].astype(str).tolist()

all_wavs = []
for p in fl_paths:
    strong_p = processed_strong_path_for(p)
    if strong_p.exists():
        all_wavs.append(str(strong_p))
    # if not preprocessed yet, skip — runner is resume-safe

print(f"[scan:FLEURS] profile={PROFILE} | STRONG files found: {len(all_wavs):,}")

if TODO_CSV.exists():
    todo = pd.read_csv(TODO_CSV)
else:
    todo = pd.DataFrame({"path_audio": all_wavs})
    todo.drop_duplicates(subset=["path_audio"], inplace=True)
    todo["profile"]  = PROFILE
    todo["vec_path"] = todo["path_audio"].map(lambda s: str(vec_path_for(PTM_NAME, Path(s))))
    todo["status"]   = "PENDING"
    todo.to_csv(TODO_CSV, index=False)

# Re-scan: mark DONE if npy exists or manifest already has it (either manifest)
done_manifest = set()
for mani in [MANIFEST_IN, MANIFEST_OUT]:
    if mani.exists():
        try:
            mf = pd.read_csv(mani, usecols=["ptm","vec_path","profile"])
            mf = mf[(mf["ptm"]==PTM_NAME) & (mf["profile"]==PROFILE)]
            done_manifest |= set(mf["vec_path"].astype(str))
        except Exception:
            pass

exists = todo["vec_path"].astype(str).map(lambda s: Path(s).exists())
inmani = todo["vec_path"].astype(str).map(lambda s: s in done_manifest)
todo.loc[exists | inmani, "status"] = "DONE"
todo.to_csv(TODO_CSV, index=False)

pending = todo[todo["status"]=="PENDING"].copy()
if CAP is not None:
    pending = pending.head(CAP)
print(f"[resume] Pending now: {len(pending):,} (of {len(todo):,}) | DRY_RUN={DRY_RUN} | CAP={CAP} | LAST_K={LAST_K}")

# ---------- Load HF model (auto-download once to cache) ----------
os.environ["HF_HOME"] = str(HF_CACHE)
from transformers import Wav2Vec2FeatureExtractor, AutoModel
fe  = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
m   = AutoModel.from_pretrained(
        MODEL_ID,
        cache_dir=str(HF_CACHE),
        use_safetensors=False
    ).to(DEVICE).eval()

@torch.inference_mode()
def pool_lastk(hidden_states, last_k=1):
    hs = hidden_states[-last_k:] if last_k>0 else [hidden_states[-1]]
    x  = torch.stack(hs, dim=0).mean(dim=0)  # (B,T,H)
    x  = x[0]                                # (T,H)
    mu = x.mean(0)
    sd = x.std(0, unbiased=False)
    return torch.cat([mu, sd], 0).cpu().numpy().astype("float32")

# ---------- Graceful exit handler (flush before exit) ----------
rows_buf = []
def _flush_and_exit(code=0):
    if not DRY_RUN and rows_buf:
        append_manifest_rows(rows_buf)
        todo.to_csv(TODO_CSV, index=False)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[exit] Progress saved. Bye.")
    sys.exit(code)

def _sigint_handler(signum, frame):
    print("\n[ctrl-c] Caught interrupt, saving progress …")
    _flush_and_exit(0)

signal.signal(signal.SIGINT, _sigint_handler)

# ---------- Main loop ----------
created = 0
with tqdm(total=len(pending), desc=f"{PTM_NAME}:FLEURS:{PROFILE}", unit="file") as pbar:
    for idx, row in pending.iterrows():
        pa, profile, vp = row["path_audio"], row["profile"], row["vec_path"]
        try:
            wav = load_pcm16_mono(pa)
            batch = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt")
            out = m(batch.input_values.to(DEVICE),
                    output_hidden_states=True,
                    return_dict=True)
            hs  = out.hidden_states if getattr(out, "hidden_states", None) is not None else [out.last_hidden_state]
            vec = pool_lastk(hs, last_k=LAST_K)

            if not DRY_RUN:
                np.save(vp, vec.astype(SAVE_DTYPE))
                dur = len(wav)/TARGET_SR
                rows_buf.append([pa, profile, PTM_NAME, vp, int(vec.shape[0]), round(dur,3)])
                todo.loc[row.name, "status"] = "DONE"
            created += 1

            if created % 200 == 0 and not DRY_RUN:
                append_manifest_rows(rows_buf); rows_buf.clear()
                todo.to_csv(TODO_CSV, index=False)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\n   [warn] {pa}: {e}")

        finally:
            pbar.update(1)

# final flush
_flush_and_exit(0)
