# runner_hubert_test.py  —  TEST feature extractor (MMS / EDGE / ALL)
# Mirrors your train runner: hardcoded ROOT, LAST_K via env, resume-safe TODO ledger.

from pathlib import Path
import os, csv, gc
import numpy as np, pandas as pd
from tqdm import tqdm

# ======= EDIT THESE =======
ROOT     = Path(r"G:\My Drive\hindi_dfake")  # <-- your real root (matches your check_env)
TEST_SET = "mms"                              # "mms" | "edge" | "all"
PROFILE  = "strong"                           # "strong" | "base"
# ==========================

PTM_NAME   = "hubert-base"
MODEL_ID   = "facebook/hubert-base-ls960"
TARGET_SR  = 16000
SAVE_DTYPE = "float16"
LAST_K     = int(os.environ.get("PTM_LASTK", "1"))
CAP        = int(os.environ.get("PTM_CAP", "0")) or None
DRY_RUN    = (os.environ.get("PTM_DRYRUN", "0") == "1")

# Fixed paths (same layout as train)
PROC_DIR   = ROOT / "processed"
WAV_DIR    = PROC_DIR / "wav" / PROFILE
FEAT_DIR   = PROC_DIR / "features" / "ptm" / PTM_NAME
META_DIR   = ROOT / "metadata"
MANIFEST   = META_DIR / "features_manifest.csv"
JOBS_DIR   = META_DIR / "ptm_jobs"
HF_CACHE   = ROOT / "models" / "hf-cache"
for d in (FEAT_DIR, META_DIR, JOBS_DIR, HF_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# Identify test sets from preserved tail
needle_mms  = "/raw/fake_tts_mms/"
needle_edge = "/raw/fake_tts/"

# ---- device & HF ----
import torch
from transformers import AutoFeatureExtractor, AutoModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(1)
else:
    torch.set_float32_matmul_precision("high")
os.environ["HF_HOME"] = str(HF_CACHE)
print(f"[env] DEVICE={DEVICE} | ROOT={ROOT}")
print(f"[cfg] PTM={PTM_NAME} | SET={TEST_SET} | PROFILE={PROFILE} | LAST_K={LAST_K} | CAP={CAP} | DRY_RUN={DRY_RUN}")

# ---- sanity ----
if not WAV_DIR.exists():
    raise FileNotFoundError(f"[check] Missing preprocessed folder: {WAV_DIR}")

def _ensure_manifest_header():
    if not MANIFEST.exists() and not DRY_RUN:
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["path_audio","profile","ptm","vec_path","dim","seconds"])

def append_manifest_rows(rows):
    if DRY_RUN or not rows: return
    _ensure_manifest_header()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

def vec_path_for(p_audio: Path) -> Path:
    # mirror the processed/wav/<profile>/ tail
    try:
        tail = p_audio.resolve().relative_to(WAV_DIR.resolve())
    except Exception:
        tail = Path(p_audio.name)
    out = FEAT_DIR / tail.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def want(p: Path) -> bool:
    s = str(p).replace("\\", "/")
    if TEST_SET == "mms":  return needle_mms in s
    if TEST_SET == "edge": return needle_edge in s
    return (needle_mms in s) or (needle_edge in s)

def load_pcm16_mono(path: Path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != TARGET_SR: raise RuntimeError(f"{path}: expected {TARGET_SR} Hz, got {sr}")
    if x.ndim > 1: x = x.mean(axis=1)
    np.clip(x, -1.0, 1.0, out=x)
    return x

@torch.inference_mode()
def pool_lastk(hidden_states, last_k=1):
    hs = hidden_states[-last_k:] if last_k>0 else [hidden_states[-1]]
    x  = torch.stack(hs, dim=0).mean(dim=0)[0]  # (T,H)
    mu = x.mean(0); sd = x.std(0, unbiased=False)
    return torch.cat([mu, sd], 0).cpu().numpy().astype("float32")

# ---- scan ----
all_wavs = sorted(WAV_DIR.rglob("*.wav"))
wavs = [p for p in all_wavs if want(p)]
if CAP: wavs = wavs[:CAP]
print(f"[scan] preprocessed files: {len(all_wavs):,} | selected (set={TEST_SET}) = {len(wavs):,}")

# ---- TODO ledger ----
todo_csv = JOBS_DIR / f"{PTM_NAME}.{TEST_SET}.{PROFILE}.todo.csv"
if todo_csv.exists():
    todo = pd.read_csv(todo_csv)
else:
    todo = pd.DataFrame({"path_audio": [str(p) for p in wavs]})
    todo.drop_duplicates(subset=["path_audio"], inplace=True)
    todo["vec_path"] = todo["path_audio"].map(lambda s: str(vec_path_for(Path(s))))
    todo["status"]   = "PENDING"
    todo.to_csv(todo_csv, index=False)

# mark DONE by existing files or manifest
done_manifest = set()
if MANIFEST.exists():
    try:
        mf = pd.read_csv(MANIFEST, usecols=["ptm","vec_path"])
        done_manifest = set(mf.loc[mf["ptm"]==PTM_NAME, "vec_path"].astype(str))
    except Exception:
        pass

exists = todo["vec_path"].astype(str).map(lambda s: Path(s).exists())
inmani = todo["vec_path"].astype(str).map(lambda s: s in done_manifest)
todo.loc[exists | inmani, "status"] = "DONE"; todo.to_csv(todo_csv, index=False)
pending = todo[todo["status"]=="PENDING"].copy()
print(f"[resume] pending={len(pending)} (of {len(todo)})")

# ---- load model ----
fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
m  = AutoModel.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE), use_safetensors=True).to(DEVICE).eval()

# ---- main ----
rows_buf, created = [], 0
FLUSH_EVERY = 200
with tqdm(total=len(pending), desc=f"{PTM_NAME}:{TEST_SET}:{PROFILE}", unit="file") as pbar:
    for _, r in pending.iterrows():
        pa, vp = Path(r["path_audio"]), r["vec_path"]
        try:
            wav = load_pcm16_mono(pa)
            batch = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt")
            out = m(batch.input_values.to(DEVICE), output_hidden_states=True, return_dict=True)
            hs  = out.hidden_states if getattr(out,"hidden_states",None) is not None else [out.last_hidden_state]
            vec = pool_lastk(hs, last_k=LAST_K)
            if not DRY_RUN:
                np.save(vp, vec.astype(SAVE_DTYPE))
                dur = len(wav)/TARGET_SR
                rows_buf.append([str(pa), PROFILE, PTM_NAME, vp, int(vec.shape[0]), round(dur,3)])
                todo.loc[r.name, "status"] = "DONE"
            created += 1
            if created % FLUSH_EVERY == 0 and not DRY_RUN:
                append_manifest_rows(rows_buf); rows_buf.clear()
                todo.to_csv(todo_csv, index=False)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"\n[warn] {pa}: {e}")
        pbar.update(1)

if not DRY_RUN and rows_buf:
    append_manifest_rows(rows_buf); rows_buf.clear()
    todo.to_csv(todo_csv, index=False)
print("[ok] HuBERT test features done.")
