# preprocess_strong.py
import subprocess, uuid, time, hashlib, os
from pathlib import Path
import numpy as np
import soundfile as sf

TARGET_SR   = 16000
TRIM_THR_DB = -45
TRIM_DUR_S  = 0.20
TARGET_DB   = -26.0  # per-file RMS target

def _rms_dbfs_arr(x: np.ndarray) -> float:
    if x.ndim > 1: x = x.mean(axis=1)
    if len(x) == 0: return -120.0
    
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 1e-9: return -120.0
    return 20.0*np.log10(min(max(rms, 1e-9), 1.0))
def _ffmpeg(*args):
    return subprocess.run(
        ["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

def _build_filter_chain_strong(gain_db: float) -> str:
    return ",".join([
        "highpass=f=20",
        "equalizer=f=3000:t=q:w=1.0:g=2.5",
        "equalizer=f=4800:t=q:w=0.9:g=2.0",
        "treble=g=1.0:f=6000:t=h:w=0.7",
        f"volume={gain_db}dB",
        f"silenceremove=start_periods=1:start_duration={TRIM_DUR_S}:start_threshold={TRIM_THR_DB}dB:stop_periods=1:stop_duration={TRIM_DUR_S}:stop_threshold={TRIM_THR_DB}dB"
    ])

def preprocess_strong_from_path(path_in: Path):
    """Return (wav_float32_mono_16k, sr, debug_times) using EXACT training chain."""
    t0 = time.perf_counter()
    dbg = {}

    # Read & (if needed) resample to 16k PCM16 via ffmpeg to match training
    t = time.perf_counter()
    tmp_res = Path(path_in).with_suffix(f".res16k.{uuid.uuid4().hex}.wav")
    p = _ffmpeg("-i", str(path_in), "-ac","1","-ar", str(TARGET_SR), "-sample_fmt","s16", str(tmp_res))
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {p.stderr.decode('utf-8','ignore')}")
    x, sr = sf.read(str(tmp_res), dtype="float32", always_2d=False)
    dbg["t_read_resample_ms"] = int((time.perf_counter() - t) * 1000)

    # Write RB temp, then EQ+gain+trim with ffmpeg
    tmp_rb  = Path(path_in).with_suffix(f".rb.{uuid.uuid4().hex}.wav")
    tmp_out = Path(path_in).with_suffix(f".tmpout.{uuid.uuid4().hex}.wav")
    sf.write(str(tmp_rb), x, TARGET_SR, subtype="PCM_16")

    gain_db = float(np.clip(TARGET_DB - _rms_dbfs_arr(x), -20.0, 20.0))
    filt = _build_filter_chain_strong(gain_db)

    t = time.perf_counter()
    p = _ffmpeg("-i", str(tmp_rb), "-ac","1","-ar", str(TARGET_SR), "-af", filt, "-sample_fmt","s16", str(tmp_out))
    dbg["t_eq_gain_trim_ms"] = int((time.perf_counter() - t) * 1000)
    try: tmp_rb.unlink(missing_ok=True)
    except: pass
    if p.returncode != 0:
        try: tmp_out.unlink(missing_ok=True)
        except: pass
        raise RuntimeError(f"ffmpeg filter failed: {p.stderr.decode('utf-8','ignore')}")

    z, sr2 = sf.read(str(tmp_out), dtype="float32", always_2d=False)
    try: tmp_out.unlink(missing_ok=True)
    except: pass

    dbg["t_pre_ms"] = int((time.perf_counter() - t0) * 1000)
    return z.astype(np.float32, copy=False), sr2, dbg
