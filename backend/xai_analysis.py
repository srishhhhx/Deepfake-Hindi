# xai_analysis.py
"""
Explainable AI Analysis Module for Deepfake Detection
Handles all visualization computations for frontend rendering.

CRITICAL ARCHITECTURE NOTE:
===========================
Your MoE model uses POOLED FEATURES (global statistics), not frame-level sequences.
This fundamentally limits what temporal information we can extract.

METHODS BREAKDOWN:
==================

✅ TRUE TEMPORAL INFORMATION:
1. compute_temporal_heatmap() - Re-extracts features for each audio segment
   - EXPENSIVE: ~40 PTM forward passes for 10s audio
   - Only method that shows model predictions over time
   
2. compute_frequency_contribution() - Mel spectrogram analysis
   - MODEL-INDEPENDENT: Pure acoustic analysis
   - Shows frequency patterns over time
   
3. detect_breathing_patterns() - Silence detection
   - MODEL-INDEPENDENT: Pure acoustic analysis
   - Shows pause patterns over time

⚠️ FEATURE-LEVEL IMPORTANCE (NOT TEMPORAL):
4. compute_expert_agreement() - Expert voting patterns
   - Uses final logits only
   - No temporal dimension
   
5. compute_attention_rollout() - Feature gradient importance
   - MISLEADINGLY NAMED: Shows which of 1536 feature dimensions were important
   - NOT which time segments (features are already pooled)
   - Should be called "feature_gradient_importance"

For advanced methods (Integrated Gradients, SHAP, Gradient×Input), see xai_advanced.py.
All advanced methods show FEATURE DIMENSION importance, not temporal patterns.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import librosa
from pathlib import Path
import time


# =========================
# TIER 1: TEMPORAL CONSISTENCY
# =========================
def compute_temporal_heatmap(
    model, 
    wav: np.ndarray, 
    sr: int,
    feature_extractor_fn,
    device: str,
    window_sec: float = 0.5,
    hop_sec: float = 0.25
) -> Dict:
    """
    Slide a window over audio and compute fakeness score for each segment.
    Returns temporal heatmap showing consistency/variation in scores.
    
    ⚠️ COMPUTATIONAL COST WARNING:
    This is the ONLY method that provides TRUE TEMPORAL INFORMATION, but it's expensive.
    For 10-second audio with 0.25s hop, this runs ~40 PTM forward passes!
    
    Why it's necessary:
    - Your MoE model uses POOLED features (global statistics)
    - To get temporal patterns, we must re-extract features for each segment
    - Each segment gets its own pooled features → model prediction → track score over time
    
    Alternative (not implemented): Perturbation on already-extracted features would be faster
    but would lose true temporal information since features are already pooled.
    
    Returns:
        {
            'timestamps': List[float],  # Center time of each window
            'scores': List[float],      # Fakeness score per window (TRUE TEMPORAL DATA)
            'mean_score': float,
            'std_score': float,
            'consistency_index': float  # Lower = more consistent (suspicious)
        }
    """
    model.eval()
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    timestamps = []
    scores = []
    
    # OPTIMIZATION: Process segments in batches to reduce overhead
    segments = []
    segment_times = []
    
    for start in range(0, len(wav) - window_samples + 1, hop_samples):
        end = start + window_samples
        segment = wav[start:end]
        
        # Pad if needed
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)), mode='constant')
        
        segments.append(segment)
        segment_times.append((start + end) / 2 / sr)
    
    # Extract features and compute scores for all segments
    for segment, timestamp in zip(segments, segment_times):
        try:
            feats = feature_extractor_fn(segment)
            xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
            
            with torch.inference_mode():
                logits, _, _ = model(xdict)
                probs = F.softmax(logits, dim=1)  # Use F.softmax for consistency
                score = float(probs[0, 1].item())
            
            timestamps.append(timestamp)
            scores.append(score)
        except Exception:
            continue
    
    if len(scores) == 0:
        return {
            'timestamps': [],
            'scores': [],
            'mean_score': 0.0,
            'std_score': 0.0,
            'consistency_index': 0.0
        }
    
    scores_arr = np.array(scores)
    mean_score = float(np.mean(scores_arr))
    std_score = float(np.std(scores_arr))
    
    # Consistency index: coefficient of variation (lower = more robotic)
    consistency_index = std_score / (mean_score + 1e-8)
    
    return {
        'timestamps': timestamps,
        'scores': scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'consistency_index': consistency_index
    }


# =========================
# TIER 1: FREQUENCY BAND CONTRIBUTION
# =========================
def compute_frequency_contribution(
    wav: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> Dict:
    """
    Analyze frequency distribution and highlight suspicious bands.
    TTS often fails at high frequencies or has unnatural formants.
    
    🚨 MODEL-INDEPENDENT METHOD:
    This is a pure ACOUSTIC ANALYSIS - it does NOT use your MoE model at all.
    It analyzes the raw audio signal using signal processing (mel spectrogram).
    This is useful for detecting TTS artifacts but is separate from model explanations.
    
    FIXED: Downsamples mel_spectrogram to prevent frontend crashes.
    FIXED: Improved suspicious band detection with energy threshold.
    
    Returns:
        {
            'mel_spectrogram': List[List[float]],  # Downsampled (freq, time)
            'freq_bins': List[float],              # Hz values
            'time_bins': List[float],              # Seconds (TRUE TEMPORAL DATA)
            'suspicious_bands': List[Dict],         # Bands with anomalies
            'high_freq_energy': float,             # Energy above 8kHz
            'formant_consistency': float           # Formant variation metric
        }
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # CRITICAL FIX: Downsample to max 100 time bins to prevent frontend crash
    max_time_bins = 100
    if mel_spec_db.shape[1] > max_time_bins:
        # Downsample time axis
        downsample_factor = mel_spec_db.shape[1] // max_time_bins
        mel_spec_db_downsampled = mel_spec_db[:, ::downsample_factor][:, :max_time_bins]
    else:
        mel_spec_db_downsampled = mel_spec_db
    
    # Frequency bins in Hz
    freq_bins = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr/2)
    time_bins = librosa.frames_to_time(
        np.arange(mel_spec.shape[1]), sr=sr, hop_length=hop_length
    )
    
    # Downsample time_bins to match
    if len(time_bins) > max_time_bins:
        downsample_factor = len(time_bins) // max_time_bins
        time_bins = time_bins[::downsample_factor][:max_time_bins]
    
    nyquist_hz = float(sr) / 2.0
    cutoff_hz = 8000.0
    cutoff_eff_hz = cutoff_hz if nyquist_hz > cutoff_hz else (0.75 * nyquist_hz)
    high_freq_idx = np.where(freq_bins >= cutoff_eff_hz)[0]
    if len(high_freq_idx) > 0:
        high_freq_energy = float(np.mean(mel_spec_db[high_freq_idx, :]))
    else:
        high_freq_energy = float(np.min(mel_spec_db)) if mel_spec_db.size > 0 else -80.0
    
    # FIXED: Detect suspicious bands with improved logic
    suspicious_bands = []
    for freq_idx in range(n_mels):
        freq_energy = mel_spec[freq_idx, :]
        std_energy = np.std(freq_energy)
        mean_energy = np.mean(freq_energy)
        
        # FIXED: Check both low variation AND sufficient energy (avoid false positives on silence)
        coefficient_of_variation = std_energy / (mean_energy + 1e-8)
        if mean_energy > 1e-3 and coefficient_of_variation < 0.2:  # Added energy threshold
            suspicious_bands.append({
                'freq_hz': float(freq_bins[freq_idx]),
                'freq_range': f"{freq_bins[freq_idx-1] if freq_idx > 0 else 0:.0f}-{freq_bins[freq_idx]:.0f} Hz",
                'uniformity': float(coefficient_of_variation)
            })
    
    # Formant consistency (simplified)
    formant_consistency = float(np.mean([np.std(mel_spec[i, :]) for i in range(20, min(60, n_mels))]))
    
    return {
        'mel_spectrogram': mel_spec_db_downsampled.tolist(),  # FIXED: Downsampled
        'freq_bins': freq_bins.tolist(),
        'time_bins': time_bins.tolist(),
        'suspicious_bands': suspicious_bands[:5],  # Top 5
        'high_freq_energy': high_freq_energy,
        'formant_consistency': formant_consistency
    }


# =========================
# TIER 1: EXPERT AGREEMENT
# =========================
def compute_expert_agreement(
    model,
    feats: Dict[str, np.ndarray],
    device: str
) -> Dict:
    """
    Compute individual expert predictions and agreement metrics.
    
    Returns:
        {
            'experts': {
                'hubert-base': {'prob_fake': float, 'prediction': str},
                'wav2vec2-base': {'prob_fake': float, 'prediction': str}
            },
            'agreement_score': float,  # 0-1, higher = more agreement
            'interpretation': str      # Human-readable explanation
        }
    """
    model.eval()
    xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
    
    with torch.inference_mode():
        logits, expert_logits, gate_weights = model(xdict)
    
    # Extract individual expert predictions
    experts = {}
    expert_names = model.ptms
    
    for idx, expert_name in enumerate(expert_names):
        expert_probs = torch.softmax(expert_logits[0, idx, :], dim=0)
        prob_fake = float(expert_probs[1].item())
        
        experts[expert_name] = {
            'prob_fake': prob_fake,
            'prediction': 'FAKE' if prob_fake >= 0.5 else 'REAL',
            'gate_weight': float(gate_weights[0, idx].item())
        }
    
    # FIXED: Compute agreement score with confidence check
    probs = [experts[e]['prob_fake'] for e in expert_names]
    
    # Check if both experts are confident (not around 0.5)
    confidences = [abs(p - 0.5) for p in probs]
    avg_confidence = np.mean(confidences)
    
    # Agreement score considers both similarity AND confidence
    prob_std = np.std(probs)
    if avg_confidence < 0.1:  # Both uncertain
        agreement_score = 0.5  # Neutral agreement
    else:
        agreement_score = max(0.0, 1.0 - prob_std)  # Higher std = lower agreement
    
    # Generate interpretation
    all_fake = all(experts[e]['prediction'] == 'FAKE' for e in expert_names)
    all_real = all(experts[e]['prediction'] == 'REAL' for e in expert_names)
    
    if all_fake and avg_confidence > 0.2:
        interpretation = "Both experts confidently detected anomalies"
    elif all_real and avg_confidence > 0.2:
        interpretation = "Both experts confidently found the audio natural"
    elif avg_confidence < 0.1:
        interpretation = "Both experts are uncertain - borderline case"
    else:
        # Mixed signals
        fake_expert = [e for e in expert_names if experts[e]['prediction'] == 'FAKE'][0]
        interpretation = f"{fake_expert} detected anomalies, but other expert(s) disagree"
    
    return {
        'experts': experts,
        'agreement_score': float(agreement_score),
        'interpretation': interpretation
    }


# =========================
# TIER 3: FEATURE-LEVEL GRADIENT IMPORTANCE
# =========================
def compute_attention_rollout(
    model,
    feats: Dict[str, np.ndarray],
    device: str,
    ptm_models: Dict  # HuBERT and Wav2Vec2 models from ptm_feat (unused - kept for API compatibility)
) -> Dict:
    """
    RENAMED: This is NOT true attention rollout - it's feature-level gradient importance.
    
    CRITICAL LIMITATION: Your MoE model uses POOLED features (global statistics), not sequences.
    This means we CANNOT extract temporal attention patterns. True attention rollout would require:
    1. Access to PTM transformer attention weights BEFORE pooling
    2. Modifying feature extraction to preserve attention maps
    
    What this actually shows: Which of the 1536 feature dimensions (768×2 PTMs) had high gradients.
    This is feature-level saliency, NOT temporal attention.
    
    For TRUE temporal analysis, use compute_temporal_heatmap() which re-processes audio segments.
    
    Returns:
        {
            'hubert_feature_importance': List[float],    # Feature dimension importance (NOT time)
            'wav2vec2_feature_importance': List[float],  # Feature dimension importance (NOT time)
            'combined_feature_importance': List[float],  # Weighted combination
            'note': str  # Warning about pooled features
        }
    """
    model.eval()
    xdict_grad = {k: torch.from_numpy(v)[None, :].to(device).requires_grad_(True) 
                  for k, v in feats.items()}
    
    # Forward pass
    logits, expert_logits, gate_weights = model(xdict_grad)
    
    # Use softmax probability for proper gradient normalization
    probs = F.softmax(logits, dim=1)
    fake_score = probs[0, 1]
    fake_score.backward()
    
    # Extract FEATURE-LEVEL importance (NOT temporal)
    feature_importance_maps = {}
    for ptm_name, feat_tensor in xdict_grad.items():
        if feat_tensor.grad is not None:
            grad_magnitude = torch.abs(feat_tensor.grad[0]).cpu().numpy()
            # Normalize to [0, 1]
            grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)
            
            # Downsample to 50 points for visualization
            if len(grad_magnitude) > 50:
                indices = np.linspace(0, len(grad_magnitude)-1, 50, dtype=int)
                grad_magnitude = grad_magnitude[indices]
            
            feature_importance_maps[ptm_name] = grad_magnitude.tolist()
        else:
            feature_importance_maps[ptm_name] = [0.0] * 50
    
    # Combined importance (weighted by gate weights)
    combined = np.zeros(len(feature_importance_maps[model.ptms[0]]))
    for idx, ptm_name in enumerate(model.ptms):
        weight = float(gate_weights[0, idx].item())
        combined += weight * np.array(feature_importance_maps[ptm_name])
    
    return {
        'hubert_feature_importance': feature_importance_maps.get('hubert-base', []),
        'wav2vec2_feature_importance': feature_importance_maps.get('wav2vec2-base', []),
        'combined_feature_importance': combined.tolist(),
        'note': 'CRITICAL: Model uses POOLED features. This shows which of the 1536 feature dimensions were important, NOT which time segments. For true temporal analysis, use temporal_heatmap instead.',
        'method': 'feature_gradient_importance'
    }


# =========================
# TIER 4: BREATHING PATTERNS
# =========================
def detect_breathing_patterns(
    wav: np.ndarray,
    sr: int,
    top_db: int = 40,
    min_silence_len: float = 0.3  # FIXED: Increased from 0.1 to 0.3 (human breaths are 0.2-0.5s)
) -> Dict:
    """
    Detect pauses/breathing patterns in speech.
    Real speech: Irregular pauses (0.2-0.8s, varies by phrase)
    TTS: Mechanically regular or absent pauses
    
    🚨 MODEL-INDEPENDENT METHOD:
    This is a pure ACOUSTIC ANALYSIS - it does NOT use your MoE model at all.
    It analyzes silence patterns in the raw audio using signal processing.
    This is useful for detecting TTS artifacts but is separate from model explanations.
    
    Returns:
        {
            'pauses': List[Dict],  # [{start: float, end: float, duration: float}] (TRUE TEMPORAL DATA)
            'pause_intervals': List[float],  # Time between pauses
            'regularity_score': float,  # Higher = more regular (suspicious)
            'mean_pause_duration': float,
            'std_pause_duration': float,
            'interpretation': str
        }
    """
    # Detect non-silent intervals
    intervals = librosa.effects.split(
        wav, top_db=top_db, frame_length=2048, hop_length=512
    )
    
    # Convert to pauses (gaps between speech)
    pauses = []
    for i in range(len(intervals) - 1):
        pause_start = intervals[i][1] / sr
        pause_end = intervals[i + 1][0] / sr
        pause_duration = pause_end - pause_start
        
        if pause_duration >= min_silence_len:
            pauses.append({
                'start': float(pause_start),
                'end': float(pause_end),
                'duration': float(pause_duration)
            })
    
    if len(pauses) == 0:
        return {
            'pauses': [],
            'pause_intervals': [],
            'regularity_score': 1.0,  # No pauses = very suspicious
            'mean_pause_duration': 0.0,
            'std_pause_duration': 0.0,
            'interpretation': "No breathing pauses detected - highly suspicious for TTS"
        }
    
    # Compute pause statistics
    durations = [p['duration'] for p in pauses]
    mean_duration = float(np.mean(durations))
    std_duration = float(np.std(durations))
    
    # Compute intervals between pauses
    pause_intervals = []
    for i in range(len(pauses) - 1):
        interval = pauses[i + 1]['start'] - pauses[i]['end']
        pause_intervals.append(float(interval))
    
    # FIXED: Regularity score with proper clamping to avoid division issues
    # Low CV = regular = suspicious
    if len(pause_intervals) > 1:
        interval_std = np.std(pause_intervals)
        interval_mean = np.mean(pause_intervals)
        if interval_mean > 0.01:  # Avoid division by near-zero
            coefficient_of_variation = interval_std / interval_mean
            regularity_score = max(0.0, min(1.0, 1.0 - coefficient_of_variation))
        else:
            regularity_score = 0.5
    else:
        regularity_score = 0.5
    
    # Interpretation
    if regularity_score > 0.7:
        interpretation = "Breathing patterns are suspiciously regular - typical of TTS"
    elif regularity_score > 0.4:
        interpretation = "Breathing patterns show moderate regularity"
    else:
        interpretation = "Breathing patterns are natural and varied"
    
    return {
        'pauses': pauses,
        'pause_intervals': pause_intervals,
        'regularity_score': float(regularity_score),
        'mean_pause_duration': mean_duration,
        'std_pause_duration': std_duration,
        'interpretation': interpretation
    }


# =========================
# MASTER ANALYSIS FUNCTION
# =========================
def run_complete_xai_analysis(
    model,
    wav: np.ndarray,
    sr: int,
    feats: Dict[str, np.ndarray],
    device: str,
    feature_extractor_fn,
    ptm_models: Optional[Dict] = None
) -> Dict:
    """
    Run all XAI analyses and return comprehensive results.
    
    IMPORTANT: Only 3 methods provide TRUE TEMPORAL INFORMATION:
    - temporal_heatmap: Model predictions over time (expensive - re-extracts features)
    - frequency_contribution: Mel spectrogram over time (model-independent)
    - breathing_patterns: Pause patterns over time (model-independent)
    
    Other methods show FEATURE-LEVEL importance (which of 1536 dimensions were important),
    NOT temporal patterns, because the model uses pooled features.
    
    Args:
        model: MoEModel instance
        wav: Audio waveform (mono, 16kHz)
        sr: Sample rate (16000)
        feats: Pre-extracted PTM features (ALREADY POOLED - no temporal info)
        device: 'cuda' or 'cpu'
        feature_extractor_fn: Function to extract features from audio segment
        ptm_models: Optional dict of PTM models (unused - kept for API compatibility)
    
    Returns:
        {
            'temporal_heatmap': {...},           # ✅ TRUE TEMPORAL (expensive)
            'frequency_contribution': {...},     # ✅ TRUE TEMPORAL (model-independent)
            'expert_agreement': {...},           # ⚠️ Feature-level only
            'attention_rollout': {...},          # ⚠️ Feature-level only (misleading name)
            'breathing_patterns': {...},         # ✅ TRUE TEMPORAL (model-independent)
            'processing_time_ms': float
        }
    """
    start_time = time.perf_counter()
    
    results = {}
    
    try:
        # Tier 1: Temporal Consistency
        results['temporal_heatmap'] = compute_temporal_heatmap(
            model, wav, sr, feature_extractor_fn, device
        )
    except Exception as e:
        results['temporal_heatmap'] = {'error': str(e)}
    
    try:
        # Tier 1: Frequency Contribution
        results['frequency_contribution'] = compute_frequency_contribution(
            wav, sr
        )
    except Exception as e:
        results['frequency_contribution'] = {'error': str(e)}
    
    try:
        # Tier 1: Expert Agreement
        results['expert_agreement'] = compute_expert_agreement(
            model, feats, device
        )
    except Exception as e:
        results['expert_agreement'] = {'error': str(e)}
    
    try:
        # Tier 3: Attention Rollout (GradCAM-style)
        results['attention_rollout'] = compute_attention_rollout(
            model, feats, device, ptm_models or {}
        )
    except Exception as e:
        results['attention_rollout'] = {'error': str(e)}

    results['feature_gradient_importance'] = results.get('attention_rollout', {})
    
    try:
        # Tier 4: Breathing Patterns
        results['breathing_patterns'] = detect_breathing_patterns(
            wav, sr
        )
    except Exception as e:
        results['breathing_patterns'] = {'error': str(e)}
    
    results['processing_time_ms'] = int((time.perf_counter() - start_time) * 1000)
    
    return results
