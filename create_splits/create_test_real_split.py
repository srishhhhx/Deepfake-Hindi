# fs_split_mms_plus_realtest.py
# Filesystem-first splitter:
#   TEST  = FLEURS real + extra real (moved) + MMS-TTS fake
#   TRAIN = everything else (+ attacks.labeled.csv)
# Universe = processed/wav/<profile> AND (optionally) intersection of PTM feature npys
# Leaves labels as-is. No reliance on old split CSVs.

import argparse, os, sys, csv, hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# ---------- utils ----------
def _norm(s: str) -> str:
    return str(s).replace("\\", "/")

def tail_after_processed(path_str: str, root: Path, profile: str) -> str:
    """Return tail under processed/wav/<profile>/ if present; else filename."""
    s = _norm(path_str)
    needle = f"/processed/wav/{profile}/"
    i = s.lower().find(needle)
    if i != -1:
        return s[i + len(needle):]
    # might be already under current root's processed
    base = _norm(str((root / "processed" / "wav" / profile)))
    if s.startswith(base):
        return s[len(base):].lstrip("/")
    return Path(s).name

def tail_after_raw(path_str: str, profile: str):
    """Root-agnostic tail after /raw/; if not raw, try processed tail, else filename."""
    s = _norm(path_str)
    if "/raw/" in s:
        return s.split("/raw/", 1)[1]
    # accept processed tail as identical structure
    # /processed/wav/<profile>/raw/...
    needle = f"/processed/wav/{profile}/"
    i = s.lower().find(needle)
    if i != -1:
        return s[i + len(needle):]
    return Path(s).name

def processed_path_for_raw_tail(root: Path, profile: str, tail_raw: str) -> Path:
    # processed/wav/<profile>/raw/<tail_raw>
    return (root / "processed" / "wav" / profile / "raw" / Path(tail_raw)).with_suffix(".wav")

def processed_path_for_any(root: Path, profile: str, any_path: str) -> Path:
    """
    Given any original path (raw or already-processed), produce processed path under root.
    """
    s = _norm(any_path)
    if "/raw/" in s:
        tail = s.split("/raw/", 1)[1]
        return processed_path_for_raw_tail(root, profile, tail)
    # if already under processed, ensure suffix .wav and return
    needle = f"/processed/wav/{profile}/"
    i = s.lower().find(needle)
    if i != -1:
        tail = s[i + len(needle):]
        return (root / "processed" / "wav" / profile / tail).with_suffix(".wav")
    # fallback: keep filename
    return (root / "processed" / "wav" / profile / Path(s).name).with_suffix(".wav")

def vec_path_for_tail(root: Path, ptm: str, profile: str, tail_proc: str) -> Path:
    # features/ptm/<ptm>/<tail_proc with .npy>
    return (root / "processed" / "features" / "ptm" / ptm / Path(tail_proc)).with_suffix(".npy")

def exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def read_csv_safe(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # DO NOT fix labels; just ensure columns exist for later joins if present.
    for c in ["utt_id","path","speaker_id","label","fake_type","source","tts_model","voice","duration"]:
        if c not in df.columns:
            df[c] = ""  # duration will be string if absent, OK for this splitter
    return df

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Filesystem-driven split: TEST (FLEURS+extra real + MMS-TTS fake) + TRAIN(all else + attacks)")
    ap.add_argument("--root", required=True, help="Root folder (e.g. G:/My Drive/hindi_dfake)")
    ap.add_argument("--profile", default="strong", help="Preprocessed profile, default strong")
    ap.add_argument("--ptms", nargs="+", default=[], help="PTM names to require (intersection). Leave empty to skip feature checks.")
    ap.add_argument("--mode", choices=["intersection","audio"], default="intersection",
                    help="'intersection' = require features for ALL PTMs; 'audio' = only preprocessed wav presence")
    ap.add_argument("--fleurs-csv", required=True, help="metadata/thirdparty_real_test.fleurs.csv")
    ap.add_argument("--extra-test-real", required=False, default=None, help="metadata/test_real.csv (moved-from-train)")
    ap.add_argument("--write", action="store_true", help="Write CSV outputs under metadata/")
    args = ap.parse_args()

    ROOT = Path(args.root)
    PROFILE = args.profile
    PTMS = list(args.ptms)
    MODE = args.mode

    META = ROOT / "metadata"
    PROC_WAV = ROOT / "processed" / "wav" / PROFILE
    FEAT_DIR = ROOT / "processed" / "features" / "ptm"
    MASTER_REAL = META / "master_real.csv"
    MASTER_FAKE = META / "master_fake.csv"
    ATTACKS = META / "attacks.labeled.csv"
    FLEURS = (ROOT / args.fleurs_csv) if not str(args.fleurs_csv).startswith(str(ROOT)) else Path(args.fleurs_csv)
    EXTRA = None
    if args.extra_test_real:
        EXTRA = (ROOT / args.extra_test_real) if not str(args.extra_test_real).startswith(str(ROOT)) else Path(args.extra_test_real)

    # --- sanity ---
    for d in [PROC_WAV, META]:
        if not d.exists():
            print(f"[err] Missing folder: {d}")
            sys.exit(2)
    if not MASTER_REAL.exists() or not MASTER_FAKE.exists():
        print("[err] Missing master_real.csv or master_fake.csv in metadata/")
        sys.exit(2)
    if not FLEURS.exists():
        print(f"[err] Missing FLEURS CSV: {FLEURS}")
        sys.exit(2)
    if EXTRA and not EXTRA.exists():
        print(f"[warn] extra-test-real CSV not found: {EXTRA} (ignored)")
        EXTRA = None
    for p in PTMS:
        if MODE == "intersection":
            if not (FEAT_DIR / p).exists():
                print(f"[err] Missing PTM feature folder: {FEAT_DIR/p}")
                sys.exit(2)

    # ---- build universes ----
    print(f"[scan] audio @ {PROC_WAV}")
    all_wavs = sorted([p for p in PROC_WAV.rglob("*.wav")])
    print(f"[scan] audio files: {len(all_wavs):,}")

    # map: tail_proc -> processed path
    tails = {}
    for p in all_wavs:
        tail = tail_after_processed(str(p), ROOT, PROFILE)
        tails[tail] = p

    # PTM intersections
    ptm_sets = []
    if MODE == "intersection" and PTMS:
        for ptm in PTMS:
            feat_root = FEAT_DIR / ptm
            have = []
            for tail in tails.keys():
                npy = vec_path_for_tail(ROOT, ptm, PROFILE, tail)
                if exists(npy):
                    have.append(tail)
            s = set(have)
            print(f"[scan] features for {ptm}: {len(s):,} npy (as wav-like tails)")
            ptm_sets.append(s)
        inter = set(tails.keys())
        for s in ptm_sets:
            inter &= s
        universe_tails = inter
        print(f"[scan] features intersection across {PTMS}: {len(universe_tails):,}")
        audio_only = set(tails.keys()) - universe_tails
        print(f"[coverage] audio with ALL PTMs present: {len(universe_tails):,}/{len(tails):,}")
        print(f"[coverage] audio missing feature (ALL PTMs): {len(audio_only):,}")
    else:
        universe_tails = set(tails.keys())

    # convenience: convert tail->vec paths
    def vecs_for_tail(tail: str) -> dict:
        d = {}
        for ptm in PTMS:
            vp = vec_path_for_tail(ROOT, ptm, PROFILE, tail)
            d[f"vec_{ptm}"] = str(vp)
        return d

    # ---- load masters & attacks (for metadata join and selecting MMS) ----
    real = read_csv_safe(MASTER_REAL)
    fake = read_csv_safe(MASTER_FAKE)
    atk  = read_csv_safe(ATTACKS) if ATTACKS.exists() else pd.DataFrame()

    for df in (real, fake, atk):
        if len(df) == 0: continue
        df["path"] = df["path"].astype(str).map(_norm)
        df["tail_raw"] = df["path"].map(lambda s: tail_after_raw(s, PROFILE))
        # processed counterpart
        df["path_audio"] = df["tail_raw"].map(lambda t: str(processed_path_for_raw_tail(ROOT, PROFILE, t)))

    # ---- TEST REAL: FLEURS + EXTRA ----
    fleurs = pd.read_csv(FLEURS)
    if "path" not in fleurs.columns:
        print("[err] FLEURS CSV must have 'path' column")
        sys.exit(2)
    fleurs["path"] = fleurs["path"].astype(str).map(_norm)
    fleurs["tail_raw"] = fleurs["path"].map(lambda s: tail_after_raw(s, PROFILE))
    fleurs["path_audio"] = fleurs["tail_raw"].map(lambda t: str(processed_path_for_raw_tail(ROOT, PROFILE, t)))
    # restrict to universe
    fleurs["tail_proc"] = fleurs["path_audio"].map(lambda s: tail_after_processed(s, ROOT, PROFILE))
    fleurs = fleurs[fleurs["tail_proc"].isin(universe_tails)].copy()

    extra = pd.DataFrame()
    if EXTRA:
        extra = pd.read_csv(EXTRA)
        if "path" not in extra.columns:
            print(f"[warn] {EXTRA} missing 'path' column; ignoring file")
            extra = pd.DataFrame()
        else:
            extra["path"] = extra["path"].astype(str).map(_norm)
            extra["tail_raw"] = extra["path"].map(lambda s: tail_after_raw(s, PROFILE))
            extra["path_audio"] = extra["tail_raw"].map(lambda t: str(processed_path_for_raw_tail(ROOT, PROFILE, t)))
            extra["tail_proc"] = extra["path_audio"].map(lambda s: tail_after_processed(s, ROOT, PROFILE))
            extra = extra[extra["tail_proc"].isin(universe_tails)].copy()

    test_real = pd.concat([fleurs[["path_audio","tail_proc"]], extra[["path_audio","tail_proc"]]], ignore_index=True)
    test_real.drop_duplicates(subset=["tail_proc"], inplace=True)

    # enrich test_real with master metadata (if available)
    real_idx = real.drop_duplicates("tail_raw").set_index("tail_raw") if len(real) else None
    def enrich_real_row(row):
        tail_proc = row["tail_proc"]
        # tail_raw equals tail_proc for our processed mirror layout under processed/wav/<profile>/raw/...
        tail_raw = tail_proc
        md = {}
        if real_idx is not None and tail_raw in real_idx.index:
            r = real_idx.loc[tail_raw]
            for c in ["utt_id","speaker_id","label","source","duration"]:
                md[c] = r.get(c, "")
        return md
    meta_cols = defaultdict(list)
    for _, rr in test_real.iterrows():
        md = enrich_real_row(rr)
        for k in ["utt_id","speaker_id","label","source","duration"]:
            meta_cols[k].append(md.get(k, ""))
    for k, v in meta_cols.items():
        test_real[k] = v

    # ---- TEST FAKE (MMS-TTS) ----
    def is_mms_row(row):
        # prefer stable fake_type
        ft = str(row.get("fake_type","")).lower()
        if ft == "tts_mms": return True
        # fallback by raw tail/ path heuristic
        s = str(row.get("path",""))
        return "/fake_tts_mms/" in _norm(s)

    if len(fake):
        fake["is_mms"] = fake.apply(is_mms_row, axis=1)
        mms = fake[fake["is_mms"]].copy()
        if len(mms) == 0:
            print("[warn] No fake rows flagged as MMS (fake_type==tts_mms); falling back to path heuristic only.")
            fake["is_mms"] = fake["path"].map(lambda s: "/fake_tts_mms/" in _norm(s))
            mms = fake[fake["is_mms"]].copy()
        mms["tail_proc"] = mms["path_audio"].map(lambda s: tail_after_processed(s, ROOT, PROFILE))
        mms = mms[mms["tail_proc"].isin(universe_tails)]
        test_fake = mms[["path_audio","tail_proc","utt_id","speaker_id","label","fake_type","source","duration"]].drop_duplicates("tail_proc").copy()
    else:
        test_fake = pd.DataFrame(columns=["path_audio","tail_proc","utt_id","speaker_id","label","fake_type","source","duration"])

    # ---- TRAIN+VAL = Universe - TEST + attacks ----
    test_tails = set(test_real["tail_proc"].tolist()) | set(test_fake["tail_proc"].tolist())
    rest_tails = [t for t in universe_tails if t not in test_tails]

    trainval_rows = []
    for t in rest_tails:
        pa = tails[t]
        row = {"path_audio": str(pa), "tail_proc": t}
        # try enrich from masters (either side)
        md = {}
        tr = real[real["tail_raw"] == t]
        if len(tr):
            r = tr.iloc[0]
            for c in ["utt_id","speaker_id","label","source","duration","fake_type","tts_model","voice"]:
                md[c] = r.get(c, "")
        else:
            tf = fake[fake["tail_raw"] == t]
            if len(tf):
                r = tf.iloc[0]
                for c in ["utt_id","speaker_id","label","source","duration","fake_type","tts_model","voice"]:
                    md[c] = r.get(c, "")
        row.update(md)
        trainval_rows.append(row)

    trainval = pd.DataFrame(trainval_rows)

    # Add attacks (all of them) to TRAIN pool
    if len(atk):
        atk = atk.copy()
        atk["tail_proc"] = atk["path_audio"].map(lambda s: tail_after_processed(s, ROOT, PROFILE))
        atk = atk[atk["tail_proc"].isin(universe_tails)]
        atk_rows = atk[["path_audio","tail_proc","utt_id","speaker_id","label","source","duration","fake_type","tts_model","voice"]].copy()
        trainval = pd.concat([trainval, atk_rows], ignore_index=True)
        trainval.drop_duplicates(subset=["tail_proc","path_audio"], inplace=True)

    # ---- attach vec columns if intersection mode ----
    if MODE == "intersection" and PTMS:
        for c in [f"vec_{p}" for p in PTMS]:
            test_real[c] = test_real["tail_proc"].map(lambda t: str(vec_path_for_tail(ROOT, c.replace("vec_",""), PROFILE, t)))
            test_fake[c] = test_fake["tail_proc"].map(lambda t: str(vec_path_for_tail(ROOT, c.replace("vec_",""), PROFILE, t)))
            trainval[c]  = trainval["tail_proc"].map(lambda t: str(vec_path_for_tail(ROOT, c.replace("vec_",""), PROFILE, t)))

    # ---- minimal column order ----
    def order_cols(df: pd.DataFrame) -> pd.DataFrame:
        base = ["path_audio","utt_id","speaker_id","label","fake_type","source","duration"]
        vecs = [c for c in df.columns if c.startswith("vec_")]
        extras = [c for c in df.columns if c not in base and c not in vecs and c != "tail_proc"]
        cols = [c for c in base if c in df.columns] + vecs + extras
        return df.reindex(columns=cols)

    test_real_o = order_cols(test_real.copy())
    test_fake_o = order_cols(test_fake.copy())
    trainval_o  = order_cols(trainval.copy())

    # ---- audits ----
    def spk_count(df):
        return df["speaker_id"].astype(str).nunique() if "speaker_id" in df.columns else 0

    print("\n========== SUMMARY (filesystem-driven) ==========")
    print(f"Universe mode : {MODE} | PTMs: {PTMS if PTMS else '(none)'}")
    print(f"TEST real     : {len(test_real_o):,}  (FLEURS + extra-real)")
    print(f"TEST fake MMS : {len(test_fake_o):,}")
    print(f"TRAIN+VAL     : {len(trainval_o):,}  (incl. attacks if present)")
    print(f"Speakers TEST real={spk_count(test_real_o)}, TEST fake={spk_count(test_fake_o)}, TRAIN+VAL={spk_count(trainval_o)}")

    # Disjointness check real speakers (TEST real vs TRAIN+VAL)
    spk_test_real = set(test_real_o["speaker_id"].astype(str)) if "speaker_id" in test_real_o.columns else set()
    spk_trainval  = set(trainval_o["speaker_id"].astype(str)) if "speaker_id" in trainval_o.columns else set()
    overlap = spk_test_real & spk_trainval
    print(f"Speaker-disjointness (TEST real vs TRAIN+VAL): {'OK' if len(overlap)==0 else 'OVERLAP='+str(len(overlap))}")

    # Coverage: all vec paths exist (if intersection)
    if MODE == "intersection" and PTMS:
        def all_vec_exist(df):
            ok = True
            for p in PTMS:
                c = f"vec_{p}"
                if c in df.columns and len(df):
                    ok = ok and bool((df[c].map(lambda s: Path(s).exists())).all())
            return ok
        print(f"Vec presence: TEST real={all_vec_exist(test_real_o)} | TEST fake={all_vec_exist(test_fake_o)} | TRAIN+VAL={all_vec_exist(trainval_o)}")

    # ---- write ----
    if args.write:
        out_real = META / "fs_test_real.ptm.csv"
        out_fake = META / "fs_test_fake_mms.ptm.csv"
        out_rest = META / "fs_trainval_rest.ptm.csv"
        test_real_o.to_csv(out_real, index=False)
        test_fake_o.to_csv(out_fake, index=False)
        trainval_o.to_csv(out_rest, index=False)
        print("\n[done] wrote:")
        print(f"  {out_real}")
        print(f"  {out_fake}")
        print(f"  {out_rest}")
    else:
        print("\n[dry-run] Use --write to save CSVs.")

if __name__ == "__main__":
    main()
