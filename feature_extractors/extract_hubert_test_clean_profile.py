# runner_hubert_test_clean.py — TEST feature extractor (HuBERT) for CLEAN preprocessed test set, resume-safe
# Uses fs_test_real.labeled.csv + fs_test_fake_mms.labeled.csv to build the exact test list.

from __future__ import annotations

from pathlib import Path
import os, csv, gc, sys, signal

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent

# ======= EDIT THESE =======
ROOT = Path(r"G:\My Drive\hindi_dfake")  # change to "/content/drive/MyDrive/hindi_dfake" on Colab
PROFILE = "clean"  # uses processed/wav/clean
# ==========================

PTM_NAME = "hubert-base"
MODEL_ID = "facebook/hubert-base-ls960"
TARGET_SR = 16000
SAVE_DTYPE = "float16"
LAST_K = int(os.environ.get("PTM_LASTK", "1"))
CAP = int(os.environ.get("PTM_CAP", "0")) or None
DRY_RUN = (os.environ.get("PTM_DRYRUN", "0") == "1")

PROC_DIR = ROOT / "processed"
WAV_DIR = PROC_DIR / "wav" / PROFILE
FEAT_DIR = REPO_ROOT / "ptm" / PTM_NAME / PROFILE
META_DIR = REPO_ROOT / "metadata"
JOBS_DIR = META_DIR / "ptm_jobs"
MANIFEST_OUT = META_DIR / "features_manifest.clean_test.csv"
HF_CACHE = REPO_ROOT / "models" / "hf-cache"

TEST_REAL_CSV = META_DIR / "fs_test_real.labeled.csv"
TEST_FAKE_CSV = META_DIR / "fs_test_fake_mms.labeled.csv"

for d in (FEAT_DIR, META_DIR, JOBS_DIR, HF_CACHE):
    d.mkdir(parents=True, exist_ok=True)

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


def clean_audio_path_from_test_csv_path(path_audio_strong: str) -> Path:
    """Map test CSV 'path_audio' (strong) to the corresponding clean wav path.

    The test CSVs typically contain something like:
      <root>/processed/wav/strong/raw/<suffix>
    We map to:
      <root>/processed/wav/clean/<suffix>
    """
    p = normalize_to_root(path_audio_strong)
    s = _norm(str(p))
    marker = "hindi_dfake/processed/wav/strong/raw/"
    if marker not in s:
        # Fall back: assume it's already a clean path or at least within WAV_DIR
        return Path(s)
    suffix = s.split(marker, 1)[1]
    out = (ROOT / "processed" / "wav" / PROFILE / Path(suffix)).with_suffix(".wav")
    return out


def _ensure_manifest_header() -> None:
    if not MANIFEST_OUT.exists() and not DRY_RUN:
        with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["path_audio", "profile", "ptm", "vec_path", "dim", "seconds"])


def append_manifest_rows(rows) -> None:
    if DRY_RUN or not rows:
        return
    _ensure_manifest_header()
    with open(MANIFEST_OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def vec_path_for(p_audio: Path) -> Path:
    try:
        tail = p_audio.resolve().relative_to(WAV_DIR.resolve())
    except Exception:
        tail = Path(p_audio.name)
    out = FEAT_DIR / tail.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def load_pcm16_mono(path: Path):
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != TARGET_SR:
        raise RuntimeError(f"{path}: expected {TARGET_SR} Hz, got {sr}")
    if hasattr(x, "ndim") and x.ndim > 1:
        x = x.mean(axis=1)
    np.clip(x, -1.0, 1.0, out=x)
    return x


def _audit_and_build_list() -> list[Path]:
    if not TEST_REAL_CSV.exists():
        raise FileNotFoundError(f"Missing: {TEST_REAL_CSV}")
    if not TEST_FAKE_CSV.exists():
        raise FileNotFoundError(f"Missing: {TEST_FAKE_CSV}")
    if not WAV_DIR.exists():
        raise FileNotFoundError(f"Missing clean wav folder: {WAV_DIR}")

    df_r = pd.read_csv(TEST_REAL_CSV)
    df_f = pd.read_csv(TEST_FAKE_CSV)

    if "path_audio" not in df_r.columns or "path_audio" not in df_f.columns:
        raise RuntimeError("Expected 'path_audio' in both test CSVs")

    clean_paths = []
    for s in df_r["path_audio"].astype(str).tolist():
        clean_paths.append(clean_audio_path_from_test_csv_path(s))
    for s in df_f["path_audio"].astype(str).tolist():
        clean_paths.append(clean_audio_path_from_test_csv_path(s))

    # Dedup while preserving order
    seen = set()
    uniq = []
    for p in clean_paths:
        ps = str(p)
        if ps not in seen:
            seen.add(ps)
            uniq.append(p)

    missing = sum(1 for p in uniq if not p.exists())
    print("[audit] Test CSV rows:")
    print(f"  real={len(df_r):,} | fake={len(df_f):,} | total={len(df_r)+len(df_f):,}")
    print("[audit] Clean wav targets (unique):")
    print(f"  total_unique={len(uniq):,} | missing_on_disk={missing:,}")
    if uniq:
        print("[audit] sample mapped clean wav:")
        for p in uniq[:3]:
            print(f"  {p} | exists={p.exists()}")

    if missing > 0:
        raise RuntimeError(
            "Some clean wav files are missing. "
            "If you just generated them, confirm your Drive path/root and that preprocessing completed."
        )

    if CAP is not None:
        uniq = uniq[:CAP]

    return uniq


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
print(f"[cfg] PTM={PTM_NAME} | PROFILE={PROFILE} | LAST_K={LAST_K} | CAP={CAP} | DRY_RUN={DRY_RUN}")


@torch.inference_mode()
def pool_lastk(hidden_states, last_k=1):
    hs = hidden_states[-last_k:] if last_k > 0 else [hidden_states[-1]]
    x = torch.stack(hs, dim=0).mean(dim=0)[0]  # (T,H)
    mu = x.mean(0)
    sd = x.std(0, unbiased=False)
    return torch.cat([mu, sd], 0).cpu().numpy().astype("float32")


# ---- Build list + TODO ledger ----
wavs = _audit_and_build_list()

todo_csv = JOBS_DIR / f"{PTM_NAME}.test.{PROFILE}.todo.csv"
if todo_csv.exists():
    todo = pd.read_csv(todo_csv)
else:
    todo = pd.DataFrame({"path_audio": [str(p) for p in wavs]})
    todo.drop_duplicates(subset=["path_audio"], inplace=True)
    todo["vec_path"] = todo["path_audio"].map(lambda s: str(vec_path_for(Path(s))))
    todo["status"] = "PENDING"
    todo.to_csv(todo_csv, index=False)

# mark DONE by existing npy or manifest
_done_manifest = set()
if MANIFEST_OUT.exists():
    try:
        mf = pd.read_csv(MANIFEST_OUT, usecols=["ptm", "vec_path", "profile"])
        mf = mf[(mf["ptm"] == PTM_NAME) & (mf["profile"] == PROFILE)]
        _done_manifest = set(mf["vec_path"].astype(str))
    except Exception:
        pass

exists = todo["vec_path"].astype(str).map(lambda s: Path(s).exists())
inmani = todo["vec_path"].astype(str).map(lambda s: s in _done_manifest)

todo.loc[exists | inmani, "status"] = "DONE"
todo.to_csv(todo_csv, index=False)

pending = todo[todo["status"] == "PENDING"].copy()
print(f"[resume] pending={len(pending):,} (of {len(todo):,})")

# ---- Load model ----
fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
m = AutoModel.from_pretrained(
    MODEL_ID,
    cache_dir=str(HF_CACHE),
    use_safetensors=True,
).to(DEVICE).eval()

# ---- Graceful exit (flush before exit) ----
rows_buf = []

def _flush_and_exit(code=0):
    if not DRY_RUN and rows_buf:
        append_manifest_rows(rows_buf)
        todo.to_csv(todo_csv, index=False)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[exit] Progress saved.")
    sys.exit(code)


def _sigint_handler(signum, frame):
    print("\n[ctrl-c] Caught interrupt, saving progress …")
    _flush_and_exit(0)


signal.signal(signal.SIGINT, _sigint_handler)

# ---- Main ----
FLUSH_EVERY = 200
created = 0
with tqdm(total=len(pending), desc=f"{PTM_NAME}:test:{PROFILE}", unit="file") as pbar:
    for _, r in pending.iterrows():
        pa, vp = Path(r["path_audio"]), r["vec_path"]
        try:
            wav = load_pcm16_mono(pa)
            batch = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt")
            out = m(batch.input_values.to(DEVICE), output_hidden_states=True, return_dict=True)
            hs = out.hidden_states if getattr(out, "hidden_states", None) is not None else [out.last_hidden_state]
            vec = pool_lastk(hs, last_k=LAST_K)

            if not DRY_RUN:
                np.save(vp, vec.astype(SAVE_DTYPE))
                dur = len(wav) / TARGET_SR
                rows_buf.append([str(pa), PROFILE, PTM_NAME, vp, int(vec.shape[0]), round(dur, 3)])
                todo.loc[r.name, "status"] = "DONE"

            created += 1
            if created % FLUSH_EVERY == 0 and not DRY_RUN:
                append_manifest_rows(rows_buf)
                rows_buf.clear()
                todo.to_csv(todo_csv, index=False)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\n[warn] {pa}: {e}")

        finally:
            pbar.update(1)

_flush_and_exit(0)
