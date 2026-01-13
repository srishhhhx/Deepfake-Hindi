# ptm_feat.py
import os, time
os.environ.setdefault("HF_HOME", "../feature_extraction_models/hf-cache")

import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor

def ensure_device():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAST_K = int(os.environ.get("PTM_LASTK", "1"))
_HF = os.environ.get("HF_HOME", "../feature_extraction_models/hf-cache")

_HUBERT_FE = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960", cache_dir=_HF)
_HUBERT    = AutoModel.from_pretrained("facebook/hubert-base-ls960", cache_dir=_HF, use_safetensors=True).to(DEVICE).eval()

_W2V_FE = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", cache_dir=_HF)
_W2V    = AutoModel.from_pretrained("facebook/wav2vec2-base", cache_dir=_HF, use_safetensors=True).to(DEVICE).eval()

@torch.inference_mode()
def _pool_lastk(hidden_states, last_k=1):
    hs = hidden_states[-last_k:] if last_k>0 else [hidden_states[-1]]
    x  = torch.stack(hs, dim=0).mean(dim=0)  # (B,T,H)
    x  = x[0]                                # (T,H)
    mu = x.mean(0)
    sd = x.std(0, unbiased=False)
    return torch.cat([mu, sd], 0).cpu().numpy().astype("float32")

@torch.inference_mode()
def extract_both_with_debug(wav_16k: np.ndarray):
    dbg = {}
    t0 = time.perf_counter()

    # HuBERT
    t = time.perf_counter()
    b1 = _HUBERT_FE(wav_16k, sampling_rate=16000, return_tensors="pt")
    o1 = _HUBERT(b1.input_values.to(DEVICE), output_hidden_states=True, return_dict=True)
    hs1 = o1.hidden_states if getattr(o1, "hidden_states", None) is not None else [o1.last_hidden_state]
    hubert_vec = _pool_lastk(hs1, last_k=int(os.environ.get("PTM_LASTK", "1")))
    dbg["t_hubert_ms"] = int((time.perf_counter() - t) * 1000)

    # W2V2
    t = time.perf_counter()
    b2 = _W2V_FE(wav_16k, sampling_rate=16000, return_tensors="pt")
    o2 = _W2V(b2.input_values.to(DEVICE), output_hidden_states=True, return_dict=True)
    hs2 = o2.hidden_states if getattr(o2, "hidden_states", None) is not None else [o2.last_hidden_state]
    w2v_vec = _pool_lastk(hs2, last_k=int(os.environ.get("PTM_LASTK", "1")))
    dbg["t_w2v_ms"] = int((time.perf_counter() - t) * 1000)

    # sanity
    for name, v in [("hubert-base", hubert_vec), ("wav2vec2-base", w2v_vec)]:
        dbg[f"{name}_shape"] = list(v.shape)
        dbg[f"{name}_dtype"] = str(v.dtype)
        dbg[f"{name}_min"]   = float(np.min(v))
        dbg[f"{name}_max"]   = float(np.max(v))
        dbg[f"{name}_mean"]  = float(np.mean(v))
        dbg[f"{name}_std"]   = float(np.std(v))
        if v.shape != (1536,):
            dbg[f"{name}_warn"] = f"Unexpected dim {v.shape}, expected (1536,)"

    dbg["device"] = DEVICE
    dbg["t_total_ms"] = int((time.perf_counter() - t0) * 1000)

    return {
        "hubert-base": hubert_vec,
        "wav2vec2-base": w2v_vec
    }, dbg
