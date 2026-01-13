# audio.py
# Kept for completeness (not used in the new pipeline directly).
from pathlib import Path
import numpy as np
import soundfile as sf

TARGET_SR = 16000

class SampleRateError(RuntimeError):
    pass

def load_audio_mono_16k(path_or_bytes):
    # If you need the strict loader elsewhere; backend now preprocesses regardless of input SR.
    if isinstance(path_or_bytes, (str, Path)):
        x, sr = sf.read(str(path_or_bytes), dtype="float32", always_2d=False)
    else:
        import io
        buf = io.BytesIO(path_or_bytes)
        x, sr = sf.read(buf, dtype="float32", always_2d=False)

    if x.ndim > 1:
        x = x.mean(axis=1)

    if sr != TARGET_SR:
        raise SampleRateError(f"Expected {TARGET_SR} Hz, got {sr} Hz")

    np.clip(x, -1.0, 1.0, out=x)
    return x, TARGET_SR
