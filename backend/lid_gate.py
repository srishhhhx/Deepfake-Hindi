# lid_gate.py
"""
Language identification gate using faster-whisper.
Checks if audio is confidently Hindi before running the MoE model.
"""
import env_setup  # MUST BE FIRST - Sets HF env vars

import os
import numpy as np
from faster_whisper import WhisperModel
from typing import Dict, Tuple

# Global model instance (loaded at startup)
_whisper_model = None


def load_lid_model(model_size: str = "base", device: str = "cuda", compute_type: str = "float16"):
    """
    Load faster-whisper model at startup.
    
    Args:
        model_size: "base" (recommended) or "small" for better accuracy
        device: "cuda" or "cpu"
        compute_type: "float16" (GPU) or "int8" (CPU)
    """
    global _whisper_model
    if _whisper_model is None:
        print(f"[LID] Loading faster-whisper model: {model_size} on {device}...", flush=True)
        
        # Windows workaround: Download to local directory without symlinks
        import huggingface_hub
        from pathlib import Path
        
        # Create local models directory
        local_model_dir = Path(__file__).parent / "models" / f"faster-whisper-{model_size}"
        local_model_dir.parent.mkdir(exist_ok=True)
        
        try:
            # Download directly to local directory (no symlinks)
            print(f"[LID] Downloading model to local directory: {local_model_dir}", flush=True)
            model_path = huggingface_hub.snapshot_download(
                f"Systran/faster-whisper-{model_size}",
                cache_dir=None,
                local_dir=str(local_model_dir),
                local_dir_use_symlinks=False  # Critical for Windows
            )
            print(f"[LID] Model downloaded successfully to: {model_path}", flush=True)
        except Exception as e:
            print(f"[LID] Error downloading model: {e}", flush=True)
            print(f"[LID] Falling back to default download method...", flush=True)
            model_path = model_size
        
        _whisper_model = WhisperModel(
            str(model_path) if isinstance(model_path, Path) else model_path,
            device=device,
            compute_type=compute_type
        )
        # Warmup with dummy audio
        dummy = np.zeros(16000, dtype=np.float32)
        _, _ = _whisper_model.transcribe(dummy, language="hi")
        print(f"[LID] Model loaded and warmed up.", flush=True)
    return _whisper_model


def compute_speech_fraction(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Simple energy-based speech detection.
    Returns fraction of audio that has speech energy.
    """
    # Frame-based energy
    frame_len = int(0.025 * sr)  # 25ms frames
    hop = int(0.010 * sr)        # 10ms hop
    
    energy = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i+frame_len]
        energy.append(np.sqrt(np.mean(frame**2)))
    
    if len(energy) == 0:
        return 0.0
    
    energy = np.array(energy)
    threshold = np.percentile(energy, 40)  # Adaptive threshold
    speech_frames = np.sum(energy > threshold)
    return speech_frames / len(energy)


def check_language(audio: np.ndarray, sr: int = 16000) -> Dict:
    """
    Run language detection on the audio.
    
    Returns:
        dict with:
            - detected_lang: ISO 639-1 code (e.g., "hi")
            - p_hi: probability for Hindi
            - top3: list of (lang, prob) tuples for top 3 languages
            - speech_fraction: estimated speech content
            - gate: "passed" or "rejected"
            - message: explanation if rejected
    """
    model = _whisper_model
    if model is None:
        raise RuntimeError("LID model not loaded. Call load_lid_model() first.")
    
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Run language detection
    segments, info = model.transcribe(
        audio,
        task="transcribe",
        language=None  # Let it detect
    )
    
    # Get language probabilities
    detected_lang = info.language
    lang_probs = info.language_probability
    
    # Get Hindi probability
    p_hi = lang_probs if detected_lang == "hi" else 0.0
    
    # If whisper doesn't provide full prob map, use the detected language prob
    # In faster-whisper, info.language_probability is just the probability of detected language
    # We need to check if Hindi is the detected language
    if detected_lang == "hi":
        p_hi = lang_probs
    else:
        # If not Hindi, assume Hindi probability is low
        p_hi = 0.0
    
    # Compute speech fraction
    speech_frac = compute_speech_fraction(audio, sr)
    
    # Build top3 list (faster-whisper only gives detected language)
    # We'll work with what we have
    top3 = [(detected_lang, lang_probs)]
    if detected_lang != "hi":
        top3.append(("hi", p_hi))
    
    # Decision policy
    gate_status = "rejected"
    message = ""
    
    # Check speech content first
    if speech_frac < 0.15:
        gate_status = "rejected"
        message = f"Insufficient speech content (speech_fraction={speech_frac:.2f})"
    
    # Check Hindi probability
    elif p_hi >= 0.70:
        gate_status = "passed"
        message = "Clear Hindi detected"
    
    elif 0.50 <= p_hi < 0.70 and (detected_lang == "hi" or p_hi > 0.40):
        gate_status = "passed"
        message = "Hindi detected (code-switch rule)"
    
    elif detected_lang == "hi" and p_hi >= 0.50:
        gate_status = "passed"
        message = "Hindi is primary language"
    
    else:
        if detected_lang != "hi" and lang_probs >= 0.85:
            gate_status = "rejected"
            message = f"Audio is confidently {detected_lang} (p={lang_probs:.2f}), not Hindi (p_hi={p_hi:.2f})"
        else:
            gate_status = "rejected"
            message = f"Hindi probability too low (p_hi={p_hi:.2f})"
    
    return {
        "detected_lang": detected_lang,
        "p_hi": round(float(p_hi), 4),
        "top3": [(lang, round(float(prob), 4)) for lang, prob in top3[:3]],
        "speech_fraction": round(float(speech_frac), 4),
        "gate": gate_status,
        "message": message,
        "lid_engine": "faster-whisper-base"
    }
