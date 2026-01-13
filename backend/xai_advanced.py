# xai_advanced.py
"""
Advanced XAI Methods: Integrated Gradients, SHAP-like, Gradient×Input

🚨 CRITICAL ARCHITECTURE LIMITATION:
====================================
Your MoE model uses POOLED features (global statistics), not frame-level sequences.

What this means:
  ✅ We CAN show which of the 1536 FEATURE DIMENSIONS (768×2 PTMs) were important
  ❌ We CANNOT show which TIME SEGMENTS were important
  
All methods in this file return FEATURE DIMENSION IMPORTANCE, not temporal patterns.
The output is smoothed/downsampled to 50 points for visualization, but this is
cosmetic - it does NOT represent true temporal information.

WHAT EACH METHOD ACTUALLY SHOWS:
=================================
1. compute_integrated_gradients() 
   → Which feature dimensions contributed to FAKE prediction
   → Uses path integral from baseline to input
   
2. compute_shap_approximation()
   → Feature dimension importance using Shapley values
   → Shows expert-level contributions
   
3. compute_gradient_x_input()
   → Feature dimension saliency using gradient × input
   → NOT true Layer-wise Relevance Propagation (LRP)

For TRUE temporal analysis, use xai_analysis.py methods:
- temporal_heatmap: Re-extracts features for each segment (expensive but accurate)
- frequency_contribution: Mel spectrogram over time (model-independent)
- breathing_patterns: Pause patterns over time (model-independent)
"""
import numpy as np
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time


def compute_integrated_gradients(
    model,
    feats: Dict[str, np.ndarray],
    device: str,
    steps: int = 50  # Restored to 50 for accuracy
) -> Dict:
    """
    Integrated Gradients: Shows which features contributed to FAKE prediction.
    
    FIXED: Gradient accumulation bug - now properly clones gradients.
    FIXED: Handles pooled features (no temporal dimension).
    
    Method:
    1. Create baseline (zero vector)
    2. Interpolate from baseline to actual input
    3. Compute gradients at each step
    4. Integrate (sum) gradients
    
    Returns:
        {
            'feature_attribution': List[float],  # Feature dimension importance (NOT temporal)
            'attribution_per_expert': Dict[str, List[float]],  # Per expert feature attribution
            'method': 'integrated_gradients',
            'note': 'Shows which FEATURE DIMENSIONS were important, NOT time segments'
        }
    """
    model.eval()
    
    # Convert to tensors
    xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
    
    # Get baseline (zeros)
    baseline = {k: torch.zeros_like(v) for k, v in xdict.items()}
    
    # Storage for gradients
    integrated_grads = {k: torch.zeros_like(v) for k, v in xdict.items()}
    
    # CRITICAL FIX: Interpolate and accumulate gradients properly
    for step in range(steps):
        alpha = (step + 1) / steps
        
        # Interpolated input
        interpolated = {
            k: baseline[k] + alpha * (xdict[k] - baseline[k])
            for k in xdict.keys()
        }
        
        # Require gradients
        for v in interpolated.values():
            v.requires_grad_(True)
        
        # Forward pass
        logits, _, _ = model(interpolated)
        fake_score = logits[0, 1]
        
        # Backward
        model.zero_grad()  # CRITICAL FIX: Zero gradients before backward
        fake_score.backward()
        
        # CRITICAL FIX: Clone gradients to prevent accumulation bug
        for k in xdict.keys():
            if interpolated[k].grad is not None:
                integrated_grads[k] += interpolated[k].grad.clone().detach()
    
    # Average and multiply by (input - baseline)
    for k in xdict.keys():
        integrated_grads[k] = (integrated_grads[k] / steps) * (xdict[k] - baseline[k])
    
    # REALITY CHECK: Features are POOLED - no true temporal information exists
    # We return feature-level attribution, NOT temporal attribution
    # The model sees global statistics, not time-specific patterns
    
    def _bin_mean(arr: np.ndarray, n_bins: int = 50) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size == 0:
            return np.array([], dtype=float)
        if arr.size <= n_bins:
            return arr
        chunks = np.array_split(arr, n_bins)
        return np.asarray([float(np.mean(c)) if c.size > 0 else 0.0 for c in chunks], dtype=float)

    feature_attribution = {}
    for ptm_name, grad_tensor in integrated_grads.items():
        grad_np = grad_tensor[0].cpu().numpy()
        
        # Get absolute attribution per feature dimension
        if len(grad_np.shape) == 2:  # [seq_len, feature_dim] - if somehow not pooled
            feat_attr = np.abs(grad_np).mean(axis=0)  # Average over time
        elif len(grad_np.shape) == 1:  # [feature_dim] - EXPECTED for pooled features
            feat_attr = np.abs(grad_np)
        else:
            feat_attr = np.abs(grad_np.flatten())
        
        feature_attribution[ptm_name] = feat_attr.astype("float32", copy=False)
    
    ptm_names = list(feature_attribution.keys())
    per_expert_viz_raw: Dict[str, np.ndarray] = {}
    per_expert_viz: Dict[str, List[float]] = {}
    for ptm_name in ptm_names:
        arr = feature_attribution.get(ptm_name)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            per_expert_viz_raw[ptm_name] = np.array([], dtype=float)
            per_expert_viz[ptm_name] = []
            continue
        binned = _bin_mean(arr, n_bins=50)
        per_expert_viz_raw[ptm_name] = binned

    global_mx = 0.0
    for v in per_expert_viz_raw.values():
        if isinstance(v, np.ndarray) and v.size > 0:
            global_mx = max(global_mx, float(np.max(v)))

    if per_expert_viz_raw:
        lengths = [v.size for v in per_expert_viz_raw.values() if isinstance(v, np.ndarray) and v.size > 0]
        if lengths:
            L = int(min(lengths))
            stack = []
            for v in per_expert_viz_raw.values():
                if isinstance(v, np.ndarray) and v.size >= L:
                    stack.append(v[:L])
            if stack:
                combined_raw = np.mean(np.vstack(stack), axis=0)
                if combined_raw.size > 0:
                    global_mx = max(global_mx, float(np.max(combined_raw)))
                combined_viz = combined_raw
            else:
                combined_viz = np.array([])
        else:
            combined_viz = np.array([])
    else:
        combined_viz = np.array([])

    if global_mx > 0:
        if isinstance(combined_viz, np.ndarray) and combined_viz.size > 0:
            combined_viz = combined_viz / global_mx
        for ptm_name, v in per_expert_viz_raw.items():
            if isinstance(v, np.ndarray) and v.size > 0:
                per_expert_viz[ptm_name] = (v / global_mx).tolist()
            else:
                per_expert_viz[ptm_name] = []
    else:
        if isinstance(combined_viz, np.ndarray) and combined_viz.size > 0:
            combined_viz = combined_viz
        for ptm_name, v in per_expert_viz_raw.items():
            per_expert_viz[ptm_name] = v.tolist() if isinstance(v, np.ndarray) else []
    
    return {
        'feature_attribution': combined_viz.tolist(),  # Feature dimension importance (smoothed for viz)
        'attribution_per_expert': per_expert_viz,
        'note': 'IMPORTANT: Model uses POOLED features. This shows which feature dimensions were important, NOT which time segments. Each PTM vector is 1536 dims (768 mean + 768 std), and the plot is downsampled/binned for visualization.',
        'method': 'integrated_gradients'
    }


def compute_shap_approximation(
    model,
    feats: Dict[str, np.ndarray],
    device: str,
    n_samples: int = 100
) -> Dict:
    """
    SHAP-like approximation: Shows feature importance per expert.
    
    FIXED: Proper Shapley value formula with coalition weighting.
    FIXED: Handles pooled features correctly.
    
    Returns:
        {
            'feature_importance': List[float],  # Feature dimension importance (NOT temporal)
            'expert_contributions': Dict[str, float],  # Overall contribution per expert
            'method': 'shap_approximation',
            'note': 'Shows which FEATURE DIMENSIONS were important, NOT time segments'
        }
    """
    model.eval()
    
    # Get baseline prediction (zero input)
    baseline = {k: torch.zeros_like(torch.from_numpy(v)[None, :]).to(device) 
                for k, v in feats.items()}
    with torch.inference_mode():
        baseline_logits, _, _ = model(baseline)
        baseline_pred = F.softmax(baseline_logits, dim=1)[0, 1].item()
    
    # Get actual prediction
    xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
    with torch.inference_mode():
        actual_logits, expert_logits, gate_weights = model(xdict)
        actual_pred = F.softmax(actual_logits, dim=1)[0, 1].item()
    
    # More accurate expert Shapley values:
    # - For small n_players, compute exact Shapley values by enumerating all coalitions.
    # - For larger n_players, fall back to permutation sampling (unbiased estimator).
    ptm_names = list(feats.keys())
    n_players = len(ptm_names)
    contributions = {ptm: 0.0 for ptm in ptm_names}

    def _pred_for_mask(mask_int: int, cache: Dict[int, float]) -> float:
        if mask_int in cache:
            return cache[mask_int]
        masked_input = {}
        for j, ptm in enumerate(ptm_names):
            if (mask_int >> j) & 1:
                masked_input[ptm] = xdict[ptm]
            else:
                masked_input[ptm] = baseline[ptm]
        with torch.inference_mode():
            masked_logits, _, _ = model(masked_input)
            masked_pred = float(F.softmax(masked_logits, dim=1)[0, 1].item())
        cache[mask_int] = masked_pred
        return masked_pred

    if n_players <= 10:
        cache: Dict[int, float] = {}
        n_fact = math.factorial(n_players)
        for i, ptm in enumerate(ptm_names):
            phi = 0.0
            for mask_int in range(0, 1 << n_players):
                if (mask_int >> i) & 1:
                    continue
                s = int(bin(mask_int).count("1"))
                weight = (math.factorial(s) * math.factorial(n_players - s - 1)) / float(n_fact)
                f_s = _pred_for_mask(mask_int, cache)
                f_s_i = _pred_for_mask(mask_int | (1 << i), cache)
                phi += weight * (f_s_i - f_s)
            contributions[ptm] = float(phi)
    else:
        # Permutation sampling estimator.
        cache: Dict[int, float] = {}
        n_perm = int(max(1, n_samples))
        for _ in range(n_perm):
            perm = np.random.permutation(n_players)
            mask_int = 0
            prev = _pred_for_mask(mask_int, cache)
            for i in perm:
                mask_int |= (1 << int(i))
                cur = _pred_for_mask(mask_int, cache)
                contributions[ptm_names[int(i)]] += float(cur - prev)
                prev = cur
        for ptm in ptm_names:
            contributions[ptm] /= float(n_perm)

    # Keep coalition predictions for transparency/debugging (small for 2 experts).
    coalition_preds = {}
    try:
        for mask_int, pred in cache.items():
            coalition = [ptm for j, ptm in enumerate(ptm_names) if ((mask_int >> j) & 1)]
            coalition_key = "+".join(coalition) if coalition else "BASELINE"
            coalition_preds[coalition_key] = float(pred)
    except Exception:
        coalition_preds = {}

    # IMPORTANT:
    # We intentionally do NOT compute per-feature (1536-dim) SHAP here.
    # True SHAP over 1536 dims requires a background distribution and many evaluations.
    feature_importance_full = np.array([])
    top_features = []
    combined_viz = np.array([])
    
    return {
        'feature_importance': combined_viz.tolist(),  # Feature dimension importance (smoothed for viz)
        'feature_importance_full': feature_importance_full.tolist(),
        'top_features': top_features,
        'expert_contributions': contributions,
        'coalition_preds': coalition_preds,
        'baseline_pred': float(baseline_pred),
        'actual_pred': float(actual_pred),
        'note': 'expert_contributions are true Shapley/SHAP values over experts (PTM inputs). Per-feature SHAP over 1536 embedding dims is intentionally omitted here to avoid misleading results without a proper background distribution.',
        'method': 'expert_shapley'
    }


def compute_gradient_x_input(
    model,
    feats: Dict[str, np.ndarray],
    device: str
) -> Dict:
    """
    Gradient × Input: Shows input contribution via gradient-based saliency.
    
    RENAMED: This is NOT true Layer-wise Relevance Propagation (LRP).
    True LRP requires layer-by-layer relevance propagation.
    This is simply gradient × input (also called input × gradient saliency).
    
    FIXED: Handles pooled features correctly.
    
    Returns:
        {
            'feature_relevance': List[float],  # Feature dimension relevance (NOT temporal)
            'relevance_per_expert': Dict[str, List[float]],
            'method': 'gradient_x_input',
            'note': 'Shows which FEATURE DIMENSIONS were important, NOT time segments'
        }
    """
    model.eval()
    
    # Forward pass with gradient
    xdict = {k: torch.from_numpy(v)[None, :].to(device).requires_grad_(True) 
             for k, v in feats.items()}
    
    logits, expert_logits, gate_weights = model(xdict)
    fake_score = F.softmax(logits, dim=1)[0, 1]  # Use softmax
    
    # Backward pass
    model.zero_grad()
    fake_score.backward()
    
    # REALITY CHECK: Extract feature-level relevance (NOT temporal)
    relevance_per_expert = {}
    for ptm_name, feat_tensor in xdict.items():
        if feat_tensor.grad is not None:
            # Gradient × Input
            relevance = (feat_tensor.grad * feat_tensor)[0].cpu().detach().numpy()
            
            # Handle pooled features
            if len(relevance.shape) == 2:  # [seq_len, feat_dim] - unlikely
                feature_rel = np.abs(relevance).mean(axis=0)
            elif len(relevance.shape) == 1:  # [feat_dim] - EXPECTED
                feature_rel = np.abs(relevance)
            else:
                feature_rel = np.abs(relevance.flatten())
            
            # Normalize
            if feature_rel.max() > 0:
                feature_rel = feature_rel / feature_rel.max()
            
            # Downsample to 50 points for visualization
            if len(feature_rel) > 50:
                indices = np.linspace(0, len(feature_rel)-1, 50, dtype=int)
                feature_rel = feature_rel[indices]
            
            relevance_per_expert[ptm_name] = feature_rel.tolist()
    
    # Combine relevance from all experts
    if relevance_per_expert:
        max_len = max(len(relevance_per_expert[p]) for p in relevance_per_expert)
        combined_relevance = np.zeros(max_len)
        for ptm_name, rel in relevance_per_expert.items():
            if len(rel) < max_len:
                rel = rel + [0.0] * (max_len - len(rel))
            combined_relevance += np.array(rel[:max_len])
        combined_relevance /= len(relevance_per_expert)
    else:
        combined_relevance = np.array([])
    
    return {
        'feature_relevance': combined_relevance.tolist(),  # Feature dimension relevance (smoothed for viz)
        'relevance_per_expert': relevance_per_expert,
        'note': 'IMPORTANT: Model uses POOLED features. This shows which of the 1536 feature dimensions were important, NOT which time segments.',
        'method': 'gradient_x_input'
    }


def run_advanced_xai(
    model,
    feats: Dict[str, np.ndarray],
    device: str
) -> Dict:
    """
    Run all advanced XAI methods.
    
    IMPORTANT: These methods reuse already-extracted POOLED features.
    They show FEATURE DIMENSION IMPORTANCE (which of 1536 dimensions were important),
    NOT temporal patterns (which time segments were important).
    
    Fast (no re-extraction) but limited to feature-level explanations.
    For true temporal analysis, use xai_analysis.compute_temporal_heatmap().
    
    Returns:
        {
            'integrated_gradients': {
                'feature_attribution': List[float],  # Feature dimension importance
                'note': str  # Warning about pooled features
            },
            'shap_approximation': {
                'feature_importance': List[float],  # Feature dimension importance
                'expert_contributions': Dict[str, float],
                'note': str
            },
            'lrp': {  # Actually gradient×input, not true LRP
                'feature_relevance': List[float],  # Feature dimension relevance
                'note': str
            },
            'processing_time_ms': int
        }
    """
    start_time = time.perf_counter()
    
    results = {}
    
    try:
        results['integrated_gradients'] = compute_integrated_gradients(
            model, feats, device, steps=50  # Restored to 50 for accuracy
        )
    except Exception as e:
        # FIXED: Better error messages
        error_msg = f"Integrated Gradients failed: {str(e)}"
        if "shape" in str(e).lower():
            error_msg += " (Likely due to feature shape mismatch)"
        results['integrated_gradients'] = {'error': error_msg, 'error_type': 'computation_error'}
    
    try:
        results['shap_approximation'] = compute_shap_approximation(
            model, feats, device, n_samples=100  # Increased for better approximation
        )
    except Exception as e:
        error_msg = f"SHAP approximation failed: {str(e)}"
        results['shap_approximation'] = {'error': error_msg, 'error_type': 'computation_error'}
    
    try:
        results['lrp'] = compute_gradient_x_input(model, feats, device)  # RENAMED
    except Exception as e:
        error_msg = f"Gradient×Input failed: {str(e)}"
        results['lrp'] = {'error': error_msg, 'error_type': 'computation_error'}

    results['gradient_x_input'] = results.get('lrp', {})
    
    results['processing_time_ms'] = int((time.perf_counter() - start_time) * 1000)
    
    return results
