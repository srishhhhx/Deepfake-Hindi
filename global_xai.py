#!/usr/bin/env python3
"""
Global XAI Analysis 

"""

import os, sys, argparse, json, warnings, importlib.util
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor, AutoModel
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# Try to import epitran for phoneme analysis
try:
    from epitran import Epitran
    EPITRAN_AVAILABLE = True
    epi = None  # Will be initialized later to avoid encoding issues
except ImportError:
    EPITRAN_AVAILABLE = False
    epi = None
    print("Warning: epitran not installed. Phoneme analysis will be skipped.")
    print("Install with: pip install epitran")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
WAV2VEC2_ID = "facebook/wav2vec2-base"
HUBERT_ID = "facebook/hubert-base-ls960"

# Import modules
sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("train_moe", Path(__file__).parent / "train_moe.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
MoEModel = train_module.MoEModel

spec_xai = importlib.util.spec_from_file_location("xai_analysis", Path(__file__).parent / "backend" / "xai_analysis.py")
xai_module = importlib.util.module_from_spec(spec_xai)
spec_xai.loader.exec_module(xai_module)
compute_temporal_heatmap = xai_module.compute_temporal_heatmap
compute_frequency_contribution = xai_module.compute_frequency_contribution

spec_xai_adv = importlib.util.spec_from_file_location("xai_advanced", Path(__file__).parent / "backend" / "xai_advanced.py")
xai_adv_module = importlib.util.module_from_spec(spec_xai_adv)
spec_xai_adv.loader.exec_module(xai_adv_module)
compute_integrated_gradients = xai_adv_module.compute_integrated_gradients
compute_shap_approximation = xai_adv_module.compute_shap_approximation

def load_model_checkpoint(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", ckpt.get("config", {}))  # FIXED: v5 uses "cfg", v4 uses "config"
    ptms = cfg.get("ptms", ["wav2vec2-base", "hubert-base"])
    
    model = MoEModel(
        ptms=ptms, in_dim_each=1536,
        expert_bottleneck=cfg.get("expert_bottleneck", 768),
        expert_drop=cfg.get("expert_dropout", 0.3),
        gate_hidden=cfg.get("gate_hidden", 64),
        gate_drop=cfg.get("gate_dropout", 0.15),
        use_batchnorm=cfg.get("use_batchnorm", True),
        use_se=cfg.get("use_se", False),
        simple_gate=cfg.get("simple_gate", True),  # FIXED: v5 uses True
        stochastic_depth=cfg.get("stochastic_depth", 0.6),
        use_fusion=cfg.get("use_fusion", True),  # FIXED: v5 uses True
        fusion_dropout=cfg.get("fusion_dropout", 0.5)
    ).to(device)
    
    # FIXED: v5 uses "model" key, not "model_state_dict"
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg, ptms


class PTMFeatureExtractor:
    def __init__(self, device: str):
        self.device = device
        self.wav2vec2_fe = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ID)
        self.wav2vec2_model = AutoModel.from_pretrained(WAV2VEC2_ID, use_safetensors=True).to(device).eval()
        self.hubert_fe = AutoFeatureExtractor.from_pretrained(HUBERT_ID)
        self.hubert_model = AutoModel.from_pretrained(HUBERT_ID, use_safetensors=True).to(device).eval()
    
    @torch.inference_mode()
    def extract_wav2vec2_frames(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        batch = self.wav2vec2_fe(audio, sampling_rate=sr, return_tensors="pt")
        out = self.wav2vec2_model(batch.input_values.to(self.device), output_hidden_states=True, return_dict=True)
        hs = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
        return hs[-1][0].cpu().numpy().astype(np.float32)
    
    @torch.inference_mode()
    def extract_hubert_frames(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        batch = self.hubert_fe(audio, sampling_rate=sr, return_tensors="pt")
        out = self.hubert_model(batch.input_values.to(self.device), output_hidden_states=True, return_dict=True)
        hs = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
        return hs[-1][0].cpu().numpy().astype(np.float32)
    
    @staticmethod
    def pool_to_1536(frames: np.ndarray) -> np.ndarray:
        mu = frames.mean(axis=0)
        sd = frames.std(axis=0)
        return np.concatenate([mu, sd], axis=0).astype(np.float32)


def count_syllables_hindi(word: str) -> int:
    hindi_vowels = set('अआइईउऊएऐओऔaeiouAEIOU')
    return max(1, sum(1 for c in word if c in hindi_vowels))


def extract_phonemes(word: str) -> List[str]:
    """Extract phonemes from Hindi word using epitran"""
    global epi
    if not EPITRAN_AVAILABLE or epi is None:
        return []
    try:
        # Transliterate to IPA phonemes
        ipa = epi.transliterate(word)
        # Split into individual phonemes (simple split by character for now)
        phonemes = list(ipa)
        return [p for p in phonemes if p.strip()]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv-real", default="metadata/fs_test_real.labeled.csv")  # FIXED: Use same as train_moe.py
    parser.add_argument("--test-csv-fake", default="metadata/fs_test_fake_mms.labeled.csv")  # FIXED: Use same as train_moe.py
    parser.add_argument("--master-real", default="metadata/master_real.csv")
    parser.add_argument("--master-real-fleurs", default="metadata/thirdparty_real_test.fleurs.csv")
    parser.add_argument("--master-real-train", default="metadata/test_real.from_train.roundrobin_least_damage.csv")
    parser.add_argument("--master-fake", default="metadata/master_fake.csv")
    parser.add_argument("--output-dir", default="global_xai_results_seed1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--shap-n-samples", type=int, default=200)
    parser.add_argument("--ptm-max-seconds", type=float, default=12.0)
    args = parser.parse_args()
    
    # Disable TensorFlow to avoid import issues
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['USE_TF'] = '0'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}\nGlobal XAI Analysis - CORRECTED\n{'='*70}")
    print(f"Device: {DEVICE}")
    
    # Initialize epitran if available
    global epi
    if EPITRAN_AVAILABLE:
        print("Note: Epitran has encoding issues on Windows. Phoneme analysis will be skipped.")
        epi = None
        # Uncomment below to try epitran (may fail on Windows):
        # try:
        #     print("Initializing epitran for phoneme analysis...")
        #     import pandas as pd
        #     original_read_csv = pd.read_csv
        #     def read_csv_utf8(*args, **kwargs):
        #         if 'encoding' not in kwargs:
        #             kwargs['encoding'] = 'utf-8'
        #         return original_read_csv(*args, **kwargs)
        #     pd.read_csv = read_csv_utf8
        #     epi = Epitran('hin-Deva')
        #     print("  Epitran initialized successfully")
        # except Exception as e:
        #     print(f"  Warning: Could not initialize epitran: {e}")
        #     epi = None
    
    # Load model
    print("\n[1/5] Loading model...")
    model, cfg, ptms = load_model_checkpoint(args.checkpoint, DEVICE)
    print(f"  Model loaded: {ptms}")
    
    # Load PTM extractors
    print("\n[2/5] Loading PTM feature extractors...")
    ptm_extractor = PTMFeatureExtractor(device=DEVICE)
    
    # Feature extractor function for compute_temporal_heatmap
    def feature_extractor_fn(audio_segment: np.ndarray) -> Dict[str, np.ndarray]:
        w2v_frames = ptm_extractor.extract_wav2vec2_frames(audio_segment, TARGET_SR)
        hub_frames = ptm_extractor.extract_hubert_frames(audio_segment, TARGET_SR)
        return {
            'wav2vec2-base': PTMFeatureExtractor.pool_to_1536(w2v_frames),
            'hubert-base': PTMFeatureExtractor.pool_to_1536(hub_frames)
        }
    
    # Load test data
    print("\n[3/5] Loading test data...")
    test_real = pd.read_csv(args.test_csv_real, dtype={"utt_id": str})
    test_fake = pd.read_csv(args.test_csv_fake, dtype={"utt_id": str})
    if args.max_samples:
        n_real = int(min(args.max_samples, len(test_real)))
        n_fake = int(min(args.max_samples, len(test_fake)))
        if args.random_sample:
            test_real = test_real.sample(n=n_real, random_state=args.sample_seed).reset_index(drop=True)
            test_fake = test_fake.sample(n=n_fake, random_state=args.sample_seed).reset_index(drop=True)
        else:
            test_real = test_real.head(n_real)
            test_fake = test_fake.head(n_fake)
    print(f"  Real: {len(test_real)}, Fake: {len(test_fake)}")
    
    # Load transcripts
    master_real = pd.read_csv(args.master_real, dtype={"utt_id": str})
    master_real_fleurs = None
    master_real_train = None
    if args.master_real_fleurs and Path(args.master_real_fleurs).exists():
        master_real_fleurs = pd.read_csv(args.master_real_fleurs, dtype={"utt_id": str})
    if args.master_real_train and Path(args.master_real_train).exists():
        master_real_train = pd.read_csv(args.master_real_train, dtype={"utt_id": str})
    master_fake = pd.read_csv(args.master_fake, dtype={"utt_id": str})

    def _norm_utt_id(x) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        s = str(x)
        if s.lower() == "nan":
            return ""
        return s.strip().lower()

    transcript_lookup = {}

    def _add_transcripts_df(df: pd.DataFrame):
        if df is None:
            return
        for _, row in df.iterrows():
            text = row.get("text") or row.get("transcript")
            if pd.notna(text):
                uid_norm = _norm_utt_id(row.get("utt_id"))
                if uid_norm:
                    transcript_lookup[uid_norm] = str(text)

    _add_transcripts_df(master_real_fleurs)
    _add_transcripts_df(master_real_train)
    _add_transcripts_df(master_real)
    _add_transcripts_df(master_fake)
    print(f"  Transcripts loaded: {len(transcript_lookup)} entries")
    
    # Debug: Check for transcript match with test data
    test_all = pd.concat([test_real, test_fake])
    def _row_transcript_key(r: pd.Series) -> str:
        uid = r.get("utt_id", "")
        uid = "" if pd.isna(uid) else str(uid).strip()
        if uid:
            return _norm_utt_id(uid)
        p = r.get("path_audio") or r.get("path")
        p = "" if pd.isna(p) else str(p).strip()
        if p:
            return _norm_utt_id(Path(p).stem)
        return ""

    real_matches = 0
    for _, r in test_real.iterrows():
        k = _row_transcript_key(r)
        if k and k in transcript_lookup:
            real_matches += 1

    fake_matches = 0
    for _, r in test_fake.iterrows():
        k = _row_transcript_key(r)
        if k and k in transcript_lookup:
            fake_matches += 1
    matched_transcripts = int(real_matches + fake_matches)
    print("  Transcript matches with test data:")
    print(f"    Real: {int(real_matches)}/{len(test_real)}")
    print(f"    Fake: {int(fake_matches)}/{len(test_fake)}")
    print(f"    Total: {matched_transcripts}/{len(test_all)}")
    if matched_transcripts == 0:
        print("  WARNING: No transcript matches found! Cross-reference linguistic analysis will be limited.")
    if int(real_matches) == 0:
        print("  WARNING: No transcripts matched for REAL samples!")
        print("           Linguistic analysis will likely cover only FAKE samples.")
    
    # Initialize results
    temporal_patterns = {
        "real": {"avg_scores_by_position": None, "std_scores_by_position": None, "hotspot_regions": [], "n_samples": 0},
        "fake": {"avg_scores_by_position": None, "std_scores_by_position": None, "hotspot_regions": [], "n_samples": 0},
        "comparison": {}
    }
    frequency_analysis = {"suspicious_bands": [], "comparison_heatmap": {}, "band_statistics": {"real": {}, "fake": {}}}
    linguistic_patterns = {"high_risk_words": [], "syllable_complexity": {"real": {}, "fake": {}}}
    expert_shap_analysis = {
        "real": {"baseline_pred": [], "actual_pred": [], "expert_contributions": defaultdict(list), "n_samples": 0},
        "fake": {"baseline_pred": [], "actual_pred": [], "expert_contributions": defaultdict(list), "n_samples": 0},
        "comparison": {}
    }
    integrated_gradients_analysis = {
        "real": {"combined": [], "per_expert": defaultdict(list), "second_half_to_first_half_ratio": [], "n_samples": 0},
        "fake": {"combined": [], "per_expert": defaultdict(list), "second_half_to_first_half_ratio": [], "n_samples": 0},
        "comparison": {}
    }
    cross_referenced_insights = []

    processing_stats = {
        "real": {
            "seen": 0,
            "missing_path": 0,
            "missing_file": 0,
            "bad_sr": 0,
            "temporal_ok": 0,
            "freq_ok": 0,
            "shap_ok": 0,
            "ig_ok": 0,
            "errors": defaultdict(int),
            "first_error": None
        },
        "fake": {
            "seen": 0,
            "missing_path": 0,
            "missing_file": 0,
            "bad_sr": 0,
            "temporal_ok": 0,
            "freq_ok": 0,
            "shap_ok": 0,
            "ig_ok": 0,
            "errors": defaultdict(int),
            "first_error": None
        }
    }
    
    # Process samples
    print("\n[4/5] Processing samples...")
    word_predictions = []
    all_sample_data = []  # Store per-sample data for cross-referencing
    phoneme_predictions = []  # Store (phoneme, score, label) for phoneme analysis

    samples_with_transcripts_real = 0
    samples_with_transcripts_fake = 0

    def _word_scores_from_temporal(words: List[str], temporal_bins: np.ndarray) -> List[float]:
        if temporal_bins is None:
            return []
        ts = np.asarray(temporal_bins, dtype=float).ravel()
        if ts.size == 0 or not words:
            return []

        syllable_counts = [int(count_syllables_hindi(w)) for w in words]
        total_syllables = int(sum(syllable_counts))

        # Fallback: uniform distribution if syllables are not available
        if total_syllables <= 0:
            n_words = len(words)
            scores = []
            for i in range(n_words):
                lo = int(np.floor(i * ts.size / n_words))
                hi = int(np.floor((i + 1) * ts.size / n_words))
                hi = max(hi, lo + 1)
                hi = min(hi, ts.size)
                scores.append(float(np.mean(ts[lo:hi])) if hi > lo else float(np.mean(ts)))
            return scores

        scores = []
        cumulative_syllables = 0
        for syll_count in syllable_counts:
            start_frac = cumulative_syllables / total_syllables
            end_frac = (cumulative_syllables + syll_count) / total_syllables
            lo = int(start_frac * ts.size)
            hi = int(end_frac * ts.size)
            hi = max(hi, lo + 1)
            hi = min(hi, ts.size)
            scores.append(float(np.mean(ts[lo:hi])) if hi > lo else float(np.mean(ts)))
            cumulative_syllables += syll_count

        return scores

    for label, test_df in [("real", test_real), ("fake", test_fake)]:
        print(f"\n  Processing {label}... ({len(test_df)} samples)")
        temporal_scores_list = []
        freq_results = []
        all_syllables = []
        mel_spectrograms = []  # For comparison heatmap
        samples_processed = 0
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {label}"):
            try:
                utt_id = row.get("utt_id", "")  # FIXED: Define utt_id at the top
                utt_id = "" if pd.isna(utt_id) else str(utt_id).strip()
                audio_path = row.get("path_audio") or row.get("path")

                processing_stats[label]["seen"] += 1
                
                # FIXED: Resolve path for real samples (convert /content/drive/MyDrive to G:\My Drive)
                if audio_path and "/content/drive/MyDrive/" in audio_path:
                    audio_path = audio_path.replace("/content/drive/MyDrive/", "G:\\My Drive\\")
                    audio_path = audio_path.replace("/", "\\")

                if (not utt_id) and audio_path:
                    utt_id = Path(audio_path).stem
                
                if not audio_path or not Path(audio_path).exists():
                    if not audio_path:
                        processing_stats[label]["missing_path"] += 1
                    else:
                        processing_stats[label]["missing_file"] += 1
                    continue
                
                audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != TARGET_SR:
                    processing_stats[label]["bad_sr"] += 1
                    continue

                np.clip(audio, -1.0, 1.0, out=audio)
                
                # Temporal analysis using compute_temporal_heatmap
                temporal_result = compute_temporal_heatmap(
                    model, audio, sr, feature_extractor_fn, DEVICE,
                    window_sec=0.5, hop_sec=0.25
                )
                
                if temporal_result['scores']:
                    scores = np.array(temporal_result['scores'])
                    binned = np.interp(
                        np.linspace(0, len(scores)-1, args.n_bins),
                        np.arange(len(scores)),
                        scores
                    )
                    temporal_scores_list.append(binned)
                    overall_score = temporal_result['mean_score']
                else:
                    overall_score = 0.0
                    binned = None

                processing_stats[label]["temporal_ok"] += 1
                
                # Frequency analysis
                freq_result = compute_frequency_contribution(audio, sr)
                freq_results.append(freq_result)

                processing_stats[label]["freq_ok"] += 1
                
                # Store mel spectrogram for comparison heatmap
                if 'mel_spectrogram' in freq_result:
                    mel_spectrograms.append(freq_result['mel_spectrogram'])
                
                # Store per-sample data for cross-referencing
                transcript_key = _norm_utt_id(utt_id)
                if (not transcript_key) and audio_path:
                    transcript_key = _norm_utt_id(Path(audio_path).stem)
                transcript_text = transcript_lookup.get(transcript_key, '')

                if transcript_text:
                    if label == "real":
                        samples_with_transcripts_real += 1
                    else:
                        samples_with_transcripts_fake += 1
                
                sample_data = {
                    'utt_id': utt_id,
                    'label': label,
                    'temporal_scores': binned if temporal_result['scores'] else None,
                    'temporal_timestamps': temporal_result.get('timestamps', []),
                    'freq_result': freq_result,
                    'transcript': transcript_text,
                    'audio_duration': len(audio) / sr,
                    'overall_score': overall_score
                }
                all_sample_data.append(sample_data)

                max_sec = float(args.ptm_max_seconds) if args.ptm_max_seconds is not None else 0.0
                audio_for_ptm = audio
                if max_sec > 0:
                    max_len = int(max_sec * sr)
                    if audio_for_ptm.size > max_len:
                        audio_for_ptm = audio_for_ptm[:max_len]

                full_feats = None
                try:
                    full_feats = feature_extractor_fn(audio_for_ptm)
                except Exception as e:
                    processing_stats[label]["errors"]["feature_extract_full"] += 1
                    if processing_stats[label]["first_error"] is None:
                        processing_stats[label]["first_error"] = f"feature_extract_full: {type(e).__name__}: {str(e)[:200]}"

                if full_feats is not None:
                    try:
                        shap_result = compute_shap_approximation(model, full_feats, DEVICE, n_samples=args.shap_n_samples)
                        expert_shap_analysis[label]["baseline_pred"].append(float(shap_result.get("baseline_pred", 0.0)))
                        expert_shap_analysis[label]["actual_pred"].append(float(shap_result.get("actual_pred", 0.0)))
                        for k, v in (shap_result.get("expert_contributions") or {}).items():
                            expert_shap_analysis[label]["expert_contributions"][k].append(float(v))
                        expert_shap_analysis[label]["n_samples"] += 1
                        processing_stats[label]["shap_ok"] += 1
                    except Exception as e:
                        processing_stats[label]["errors"]["shap"] += 1
                        if processing_stats[label]["first_error"] is None:
                            processing_stats[label]["first_error"] = f"shap: {type(e).__name__}: {str(e)[:200]}"

                    try:
                        ig_result = compute_integrated_gradients(model, full_feats, DEVICE, steps=int(args.ig_steps))
                        combined_ig = np.asarray(ig_result.get("feature_attribution", []), dtype=float).ravel()
                        if combined_ig.size > 0:
                            integrated_gradients_analysis[label]["combined"].append(combined_ig.tolist())
                            half = int(combined_ig.size // 2)
                            first = float(np.sum(np.abs(combined_ig[:half])))
                            second = float(np.sum(np.abs(combined_ig[half:])))
                            integrated_gradients_analysis[label]["second_half_to_first_half_ratio"].append(float(second / (first + 1e-9)))
                        for k, v in (ig_result.get("attribution_per_expert") or {}).items():
                            arr = np.asarray(v, dtype=float).ravel()
                            if arr.size > 0:
                                integrated_gradients_analysis[label]["per_expert"][k].append(arr.tolist())
                        integrated_gradients_analysis[label]["n_samples"] += 1
                        processing_stats[label]["ig_ok"] += 1
                    except Exception as e:
                        processing_stats[label]["errors"]["ig"] += 1
                        if processing_stats[label]["first_error"] is None:
                            processing_stats[label]["first_error"] = f"ig: {type(e).__name__}: {str(e)[:200]}"

                if transcript_text:
                    words = transcript_text.split()
                    per_word_scores = _word_scores_from_temporal(words, binned)
                    for wi, word in enumerate(words):
                        syllables = count_syllables_hindi(word)
                        all_syllables.append(syllables)
                        if per_word_scores:
                            word_score = per_word_scores[min(wi, len(per_word_scores) - 1)]
                        else:
                            word_score = overall_score
                        word_predictions.append((word, word_score, label))

                        if EPITRAN_AVAILABLE:
                            phonemes = extract_phonemes(word)
                            for phoneme in phonemes:
                                phoneme_predictions.append((phoneme, word_score, label))

                samples_processed += 1

                if samples_processed == 10:
                    if processing_stats[label]["shap_ok"] == 0:
                        print(f"  WARNING: SHAP failed on first 10 {label} samples")
                        if processing_stats[label]["first_error"]:
                            print(f"    First error: {processing_stats[label]['first_error']}")
                    if processing_stats[label]["ig_ok"] == 0:
                        print(f"  WARNING: Integrated Gradients failed on first 10 {label} samples")
                        if processing_stats[label]["first_error"]:
                            print(f"    First error: {processing_stats[label]['first_error']}")

            except Exception as e:
                processing_stats[label]["errors"]["sample"] += 1
                if processing_stats[label]["first_error"] is None:
                    processing_stats[label]["first_error"] = f"sample: {type(e).__name__}: {str(e)[:200]}"
                continue

        total_errors = int(sum(processing_stats[label]["errors"].values()))
        print(f"\n  Error Summary for {label}:")
        if total_errors > 0:
            for error_type, count in processing_stats[label]["errors"].items():
                if int(count) > 0:
                    print(f"    {error_type}: {int(count)}")
            if processing_stats[label]["first_error"]:
                print(f"    First error: {processing_stats[label]['first_error']}")
        else:
            print("    No errors encountered")

        print(f"  {label}: Successfully processed {samples_processed}/{len(test_df)} samples")
        
        # Aggregate temporal
        if temporal_scores_list:
            arr = np.array(temporal_scores_list)
            avg_scores = arr.mean(axis=0)
            temporal_patterns[label]["avg_scores_by_position"] = avg_scores.tolist()
            temporal_patterns[label]["std_scores_by_position"] = arr.std(axis=0).tolist()
            temporal_patterns[label]["n_samples"] = len(temporal_scores_list)
            
            # HOTSPOT DETECTION
            threshold = 0.6
            hotspots = []
            in_hotspot = False
            hotspot_start = None
            
            for i, score in enumerate(avg_scores):
                if score > threshold and not in_hotspot:
                    hotspot_start = i
                    in_hotspot = True
                elif score <= threshold and in_hotspot:
                    position_pct = f"{int(hotspot_start * 100 / len(avg_scores))}-{int(i * 100 / len(avg_scores))}%"
                    hotspots.append({
                        "position_pct": position_pct,
                        "avg_score": float(np.mean(avg_scores[hotspot_start:i])),
                        "interpretation": "Consistently high detection scores in this region"
                    })
                    in_hotspot = False
            
            # Handle case where hotspot extends to end
            if in_hotspot:
                position_pct = f"{int(hotspot_start * 100 / len(avg_scores))}-100%"
                hotspots.append({
                    "position_pct": position_pct,
                    "avg_score": float(np.mean(avg_scores[hotspot_start:])),
                    "interpretation": "Consistently high detection scores in this region"
                })
            
            temporal_patterns[label]["hotspot_regions"] = hotspots
        
        # Aggregate frequency - BAND-BY-BAND BREAKDOWN
        if freq_results:
            # Overall high-freq stats
            high_freq = [f.get('high_freq_energy', 0.0) for f in freq_results]
            
            # Band-by-band analysis (16 bands from 0-8kHz)
            band_stats = {}
            for band_idx in range(16):
                freq_low = band_idx * 500
                freq_high = (band_idx + 1) * 500
                band_name = f"{freq_low}-{freq_high}Hz"
                
                # Extract energy and variance for this band from all samples
                band_energies = []
                band_variances = []
                
                for freq_res in freq_results:
                    mel_spec = freq_res.get('mel_spectrogram', [])
                    freq_bins = np.asarray(freq_res.get('freq_bins', []), dtype=float).ravel()
                    mel_arr = np.asarray(mel_spec, dtype=float)
                    if mel_arr.ndim == 2 and mel_arr.size > 0 and freq_bins.size == mel_arr.shape[0]:
                        idxs = np.where((freq_bins >= freq_low) & (freq_bins < freq_high))[0]
                        if idxs.size > 0:
                            band_data = mel_arr[idxs, :]
                            if band_data.size > 0:
                                band_data_linear = np.power(10.0, band_data / 10.0)
                                band_energies.append(float(np.mean(band_data_linear)))
                                band_variances.append(float(np.var(band_data_linear)))
                
                if band_energies:
                    band_stats[band_name] = {
                        "mean_energy": float(np.mean(band_energies)),
                        "mean_variance": float(np.mean(band_variances)),
                        "samples": len(band_energies)
                    }
            
            frequency_analysis["band_statistics"][label] = band_stats
            
            # Flag suspicious bands (low variance = unnaturally uniform)
            if label == "fake":
                for band_name, stats in band_stats.items():
                    if stats["mean_variance"] < 0.01:
                        frequency_analysis["suspicious_bands"].append({
                            "freq_range": band_name,
                            "mean_variance": stats["mean_variance"],
                            "interpretation": f"Unnaturally uniform energy in {band_name} (typical of TTS)"
                        })
        
        # Generate comparison heatmap
        if mel_spectrograms:
            # FIXED: Normalize all spectrograms to same time dimension before averaging
            n_freq_bins = 20
            n_time_bins = 50
            
            # Downsample each spectrogram to fixed size first
            normalized_specs = []
            for mel_spec in mel_spectrograms:
                if len(mel_spec) == 0 or len(mel_spec[0]) == 0:
                    continue
                
                # Downsample this spectrogram to 20x50
                downsampled = np.zeros((n_freq_bins, n_time_bins))
                n_mels = len(mel_spec)
                n_time = len(mel_spec[0])
                
                for i in range(n_freq_bins):
                    for j in range(n_time_bins):
                        freq_start = i * n_mels // n_freq_bins
                        freq_end = (i + 1) * n_mels // n_freq_bins
                        time_start = j * n_time // n_time_bins
                        time_end = (j + 1) * n_time // n_time_bins
                        
                        # Extract region and compute mean
                        region = [mel_spec[f][time_start:time_end] for f in range(freq_start, freq_end)]
                        if region and len(region[0]) > 0:
                            downsampled[i, j] = np.mean([np.mean(row) for row in region if len(row) > 0])
                
                normalized_specs.append(downsampled)
            
            # Now average all normalized spectrograms
            if normalized_specs:
                avg_spec = np.mean(normalized_specs, axis=0)
                frequency_analysis["comparison_heatmap"][f"{label}_avg_spectrogram"] = avg_spec.tolist()
        
        # Linguistic stats
        if all_syllables:
            linguistic_patterns["syllable_complexity"][label] = {
                "avg_syllables_per_word": float(np.mean(all_syllables)),
                "words_4plus_syllables": int(sum(1 for s in all_syllables if s >= 4))
            }
    
    # Word-score correlations
    print("\n  Computing linguistic correlations...")
    word_scores = defaultdict(list)
    word_scores_by_label = defaultdict(lambda: {"real": [], "fake": []})
    for word, score, label in word_predictions:
        word_scores[word].append(score)
        if label in {"real", "fake"}:
            word_scores_by_label[word][label].append(score)
    
    word_avg = [(w, np.mean(s), len(s)) for w, s in word_scores.items() if len(s) >= 3]
    word_avg.sort(key=lambda x: x[1], reverse=True)
    
    linguistic_patterns["high_risk_words"] = [
        {"word": w, "avg_score_when_present": float(s), "occurrences": int(c)}
        for w, s, c in word_avg[:20]
    ]

    try:
        rows = []
        for w, scores_all in word_scores.items():
            s_all = np.asarray(scores_all, dtype=float)
            s_real = np.asarray(word_scores_by_label[w]["real"], dtype=float)
            s_fake = np.asarray(word_scores_by_label[w]["fake"], dtype=float)
            row = {
                "word": w,
                "count_total": int(s_all.size),
                "mean_total": float(np.mean(s_all)) if s_all.size else None,
                "std_total": float(np.std(s_all)) if s_all.size else None,
                "count_real": int(s_real.size),
                "mean_real": float(np.mean(s_real)) if s_real.size else None,
                "std_real": float(np.std(s_real)) if s_real.size else None,
                "count_fake": int(s_fake.size),
                "mean_fake": float(np.mean(s_fake)) if s_fake.size else None,
                "std_fake": float(np.std(s_fake)) if s_fake.size else None,
            }
            if row["mean_fake"] is not None and row["mean_real"] is not None:
                row["mean_fake_minus_real"] = float(row["mean_fake"] - row["mean_real"])
            else:
                row["mean_fake_minus_real"] = None
            rows.append(row)

        df_words = pd.DataFrame(rows)
        if not df_words.empty:
            df_words = df_words.sort_values(["count_total", "mean_total"], ascending=[False, False])
            df_words.to_csv(output_dir / "word_stats.csv", index=False, encoding="utf-8")
            with open(output_dir / "word_stats.json", "w", encoding="utf-8") as f:
                json.dump(df_words.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    
    # Phoneme analysis (if epitran available)
    if EPITRAN_AVAILABLE and phoneme_predictions:
        print("  Computing phoneme analysis...")
        phoneme_counts = defaultdict(lambda: {"real": 0, "fake": 0, "real_high_score": 0, "fake_high_score": 0})
        
        for phoneme, score, label in phoneme_predictions:
            phoneme_counts[phoneme][label] += 1
            if score > 0.7:  # High score threshold
                phoneme_counts[phoneme][f"{label}_high_score"] += 1
        
        # Compute mispronunciation rate
        phoneme_analysis = []
        for phoneme, counts in phoneme_counts.items():
            if counts["fake"] >= 5:  # Minimum occurrences
                fake_high_pct = (counts["fake_high_score"] / counts["fake"]) * 100 if counts["fake"] > 0 else 0
                real_high_pct = (counts["real_high_score"] / counts["real"]) * 100 if counts["real"] > 0 else 0
                
                # Only include if significantly different
                if abs(fake_high_pct - real_high_pct) > 10:
                    phoneme_analysis.append({
                        "phoneme": phoneme,
                        "mispronounced_pct_fake": float(fake_high_pct),
                        "mispronounced_pct_real": float(real_high_pct),
                        "occurrences_fake": int(counts["fake"]),
                        "occurrences_real": int(counts["real"]),
                        "interpretation": f"TTS struggles with this phoneme" if fake_high_pct > real_high_pct else "Real audio struggles more"
                    })
        
        # Sort by difference in mispronunciation rate
        phoneme_analysis.sort(key=lambda x: abs(x["mispronounced_pct_fake"] - x["mispronounced_pct_real"]), reverse=True)
        linguistic_patterns["phoneme_analysis"] = phoneme_analysis[:20]  # Top 20
    else:
        linguistic_patterns["phoneme_analysis"] = []
    
    # Temporal comparison
    print("\n[5/5] Computing comparison metrics...")
    if temporal_patterns["real"]["avg_scores_by_position"] and temporal_patterns["fake"]["avg_scores_by_position"]:
        real_dist = np.abs(np.array(temporal_patterns["real"]["avg_scores_by_position"]))
        fake_dist = np.abs(np.array(temporal_patterns["fake"]["avg_scores_by_position"]))
        real_dist /= (real_dist.sum() + 1e-9)
        fake_dist /= (fake_dist.sum() + 1e-9)
        
        kl_div = float(scipy_stats.entropy(fake_dist + 1e-9, real_dist + 1e-9))
        if np.isinf(kl_div) or np.isnan(kl_div):
            kl_div = None
        
        temporal_patterns["comparison"] = {
            "divergence_metric": kl_div,
            "interpretation": "Different temporal patterns" if kl_div and kl_div > 0.3 else "Similar patterns"
        }
    
    # Compute difference heatmap
    if "real_avg_spectrogram" in frequency_analysis["comparison_heatmap"] and "fake_avg_spectrogram" in frequency_analysis["comparison_heatmap"]:
        real_spec = np.array(frequency_analysis["comparison_heatmap"]["real_avg_spectrogram"])
        fake_spec = np.array(frequency_analysis["comparison_heatmap"]["fake_avg_spectrogram"])
        difference = fake_spec - real_spec
        frequency_analysis["comparison_heatmap"]["difference_map"] = difference.tolist()
    
    # CROSS-REFERENCED INSIGHTS - PER-SAMPLE ALIGNMENT
    print("\n  Computing cross-referenced insights...")

    linguistic_patterns["transcript_coverage"] = {
        "real_samples_with_transcripts": int(samples_with_transcripts_real),
        "fake_samples_with_transcripts": int(samples_with_transcripts_fake),
        "warning": ("Real samples have missing transcripts" if int(samples_with_transcripts_real) < 10 else None)
    }
    
    temporal_freq_alignments = 0
    temporal_linguistic_alignments = 0
    all_three_alignments = 0
    total_peaks = 0
    samples_with_transcripts = 0
    samples_with_freq_data = 0
    
    for sample_data in all_sample_data[:100]:  # Limit to first 100 for performance
        # FIXED: Check if temporal_scores is None or empty (not numpy array boolean)
        temporal_scores = sample_data['temporal_scores']
        if temporal_scores is None or len(temporal_scores) == 0:
            continue
        
        audio_duration = sample_data['audio_duration']
        transcript = sample_data['transcript']
        words = transcript.split() if transcript else []
        
        # Track data availability
        if words:
            samples_with_transcripts += 1
        
        # Find temporal peaks using dynamic threshold (top 3 above 85th percentile)
        ts = np.array(temporal_scores, dtype=float)
        if ts.size == 0:
            continue
        perc85 = float(np.percentile(ts, 85))
        top_idx = np.argsort(ts)[-3:][::-1]
        peak_indices = [int(i) for i in top_idx if ts[int(i)] >= perc85]
        if not peak_indices:
            # Fallback: take top 3 anyway
            peak_indices = [int(i) for i in top_idx]
        total_peaks += len(peak_indices)
        
        for peak_idx in peak_indices[:3]:  # Top 3 peaks per sample
            # Convert bin index to time
            time_sec = (peak_idx / len(temporal_scores)) * audio_duration
            peak_score = temporal_scores[peak_idx]
            
            # Check frequency anomaly at that time
            freq_result = sample_data['freq_result']
            mel_spec = freq_result.get('mel_spectrogram', [])
            has_freq_anomaly = False
            freq_bands = []
            
            if len(mel_spec) > 0 and len(mel_spec[0]) > 0:
                samples_with_freq_data += 1
                time_idx = int((time_sec / audio_duration) * len(mel_spec[0]))
                time_idx = min(time_idx, len(mel_spec[0]) - 1)
                
                # Extract frequency slice at this time
                time_slice = np.array([row[time_idx] if time_idx < len(row) else 0 for row in mel_spec])
                
                if len(time_slice) > 0:
                    slice_mean = float(np.mean(time_slice))
                    slice_std = float(np.std(time_slice))
                    is_very_quiet = slice_mean < -60.0
                    is_unnaturally_uniform = slice_std < 3.0
                    is_erratic = slice_std > 15.0

                    if is_very_quiet or is_unnaturally_uniform or is_erratic:
                        has_freq_anomaly = True
                        
                        # Find which bands are anomalous
                        for band_idx in range(min(16, len(time_slice) // 8)):
                            start = band_idx * len(time_slice) // 16
                            end = (band_idx + 1) * len(time_slice) // 16
                            band_mean = np.mean(time_slice[start:end])
                            band_var = np.var(time_slice[start:end])
                            
                            # Flag band if low energy OR low variance
                            if band_mean < -45 or band_var < 2.0:
                                freq_bands.append(f"{band_idx*500}-{(band_idx+1)*500}Hz")
            
            # Estimate word at this time (uniform distribution)
            word = None
            if words:
                word_idx = int((time_sec / audio_duration) * len(words))
                word_idx = min(word_idx, len(words) - 1)
                word = words[word_idx]
            
            # Count alignments
            if has_freq_anomaly:
                temporal_freq_alignments += 1
            if word:
                temporal_linguistic_alignments += 1
            if has_freq_anomaly and word:
                all_three_alignments += 1

            # Add to insights (limit to 10), even if only some modalities present
            if len(cross_referenced_insights) < 10:
                insight_obj = {
                    "sample_id": sample_data.get('utt_id', ''),
                    "temporal_peak": {
                        "time_range": f"{time_sec:.1f}s",
                        "score": float(peak_score)
                    }
                }
                if has_freq_anomaly:
                    insight_obj["frequency_anomaly"] = {
                        "freq_range": ", ".join(freq_bands[:2]) if freq_bands else "",
                        "bands": freq_bands,
                        "interpretation": "Low variance/energy at this time"
                    }
                if word:
                    insight_obj["linguistic_content"] = {
                        "word": word,
                        "estimated_time": f"~{time_sec:.1f}s",
                        "syllables": count_syllables_hindi(word)
                    }
                # Combined insight summary
                parts = [f"Model flagged {time_sec:.1f}s (score: {peak_score:.2f})"]
                if word:
                    parts.append(f"word '{word}'")
                if freq_bands:
                    parts.append(f"freq bands {', '.join(freq_bands[:2])}")
                insight_obj["combined_insight"] = "; ".join(parts)
                
                cross_referenced_insights.append(insight_obj)
    
    # Add alignment statistics with debug info
    print(f"\n  Cross-reference stats:")
    print(f"    Total peaks analyzed: {total_peaks}")
    print(f"    Samples with transcripts: {samples_with_transcripts}")
    print(f"    Samples with transcripts (real): {int(samples_with_transcripts_real)}")
    print(f"    Samples with transcripts (fake): {int(samples_with_transcripts_fake)}")
    print(f"    Samples with freq data: {samples_with_freq_data}")
    print(f"    Temporal-Freq alignments: {temporal_freq_alignments}")
    print(f"    Temporal-Linguistic alignments: {temporal_linguistic_alignments}")
    print(f"    All-three alignments: {all_three_alignments}")
    
    if total_peaks > 0:
        cross_referenced_insights.append({
            "pattern_type": "alignment_statistics",
            "temporal_freq_alignment": float(temporal_freq_alignments / total_peaks),
            "temporal_linguistic_alignment": float(temporal_linguistic_alignments / total_peaks),
            "all_three_alignment": float(all_three_alignments / total_peaks),
            "description": f"{int(all_three_alignments/total_peaks*100)}% of temporal peaks align with both frequency anomalies and words",
            "debug_info": {
                "samples_with_transcripts": samples_with_transcripts,
                "samples_with_transcripts_real": int(samples_with_transcripts_real),
                "samples_with_transcripts_fake": int(samples_with_transcripts_fake),
                "samples_with_freq_data": samples_with_freq_data,
                "total_peaks": total_peaks
            }
        })
    else:
        cross_referenced_insights.append({
            "pattern_type": "alignment_statistics",
            "description": "No temporal peaks found for cross-reference analysis",
            "debug_info": {
                "samples_with_transcripts": samples_with_transcripts,
                "samples_with_freq_data": samples_with_freq_data,
                "total_samples_processed": len(all_sample_data)
            }
        })
    
    # Add global summary
    cross_referenced_insights.append({
        "pattern_type": "global_summary",
        "description": f"Analyzed {temporal_patterns['real']['n_samples']} real and {temporal_patterns['fake']['n_samples']} fake samples",
        "top_risk_words": [w["word"] for w in linguistic_patterns["high_risk_words"][:5]],
        "total_peaks_analyzed": int(total_peaks)
    })
    
    # Save
    results = {
        "temporal_patterns": temporal_patterns,
        "frequency_analysis": frequency_analysis,
        "linguistic_patterns": linguistic_patterns,
        "processing_stats": {
            "real": {
                "seen": int(processing_stats["real"]["seen"]),
                "missing_path": int(processing_stats["real"]["missing_path"]),
                "missing_file": int(processing_stats["real"]["missing_file"]),
                "bad_sr": int(processing_stats["real"]["bad_sr"]),
                "temporal_ok": int(processing_stats["real"]["temporal_ok"]),
                "freq_ok": int(processing_stats["real"]["freq_ok"]),
                "shap_ok": int(processing_stats["real"]["shap_ok"]),
                "ig_ok": int(processing_stats["real"]["ig_ok"]),
                "errors": {k: int(v) for k, v in processing_stats["real"]["errors"].items()},
                "first_error": processing_stats["real"]["first_error"]
            },
            "fake": {
                "seen": int(processing_stats["fake"]["seen"]),
                "missing_path": int(processing_stats["fake"]["missing_path"]),
                "missing_file": int(processing_stats["fake"]["missing_file"]),
                "bad_sr": int(processing_stats["fake"]["bad_sr"]),
                "temporal_ok": int(processing_stats["fake"]["temporal_ok"]),
                "freq_ok": int(processing_stats["fake"]["freq_ok"]),
                "shap_ok": int(processing_stats["fake"]["shap_ok"]),
                "ig_ok": int(processing_stats["fake"]["ig_ok"]),
                "errors": {k: int(v) for k, v in processing_stats["fake"]["errors"].items()},
                "first_error": processing_stats["fake"]["first_error"]
            }
        },
        "expert_shap_analysis": {
            "real": {
                "n_samples": int(expert_shap_analysis["real"]["n_samples"]),
                "baseline_pred_mean": float(np.mean(expert_shap_analysis["real"]["baseline_pred"])) if expert_shap_analysis["real"]["baseline_pred"] else None,
                "actual_pred_mean": float(np.mean(expert_shap_analysis["real"]["actual_pred"])) if expert_shap_analysis["real"]["actual_pred"] else None,
                "expert_contribution_mean": {k: float(np.mean(v)) for k, v in expert_shap_analysis["real"]["expert_contributions"].items() if v},
                "expert_contribution_std": {k: float(np.std(v)) for k, v in expert_shap_analysis["real"]["expert_contributions"].items() if v}
            },
            "fake": {
                "n_samples": int(expert_shap_analysis["fake"]["n_samples"]),
                "baseline_pred_mean": float(np.mean(expert_shap_analysis["fake"]["baseline_pred"])) if expert_shap_analysis["fake"]["baseline_pred"] else None,
                "actual_pred_mean": float(np.mean(expert_shap_analysis["fake"]["actual_pred"])) if expert_shap_analysis["fake"]["actual_pred"] else None,
                "expert_contribution_mean": {k: float(np.mean(v)) for k, v in expert_shap_analysis["fake"]["expert_contributions"].items() if v},
                "expert_contribution_std": {k: float(np.std(v)) for k, v in expert_shap_analysis["fake"]["expert_contributions"].items() if v}
            }
        },
        "integrated_gradients_analysis": {
            "real": {
                "n_samples": int(integrated_gradients_analysis["real"]["n_samples"]),
                "combined_mean_curve": (np.mean(np.asarray(integrated_gradients_analysis["real"]["combined"], dtype=float), axis=0).tolist() if integrated_gradients_analysis["real"]["combined"] else None),
                "combined_std_curve": (np.std(np.asarray(integrated_gradients_analysis["real"]["combined"], dtype=float), axis=0).tolist() if integrated_gradients_analysis["real"]["combined"] else None),
                "second_half_to_first_half_ratio_mean": float(np.mean(integrated_gradients_analysis["real"]["second_half_to_first_half_ratio"])) if integrated_gradients_analysis["real"]["second_half_to_first_half_ratio"] else None,
                "per_expert_mean_curve": {k: np.mean(np.asarray(v, dtype=float), axis=0).tolist() for k, v in integrated_gradients_analysis["real"]["per_expert"].items() if v},
                "per_expert_std_curve": {k: np.std(np.asarray(v, dtype=float), axis=0).tolist() for k, v in integrated_gradients_analysis["real"]["per_expert"].items() if v}
            },
            "fake": {
                "n_samples": int(integrated_gradients_analysis["fake"]["n_samples"]),
                "combined_mean_curve": (np.mean(np.asarray(integrated_gradients_analysis["fake"]["combined"], dtype=float), axis=0).tolist() if integrated_gradients_analysis["fake"]["combined"] else None),
                "combined_std_curve": (np.std(np.asarray(integrated_gradients_analysis["fake"]["combined"], dtype=float), axis=0).tolist() if integrated_gradients_analysis["fake"]["combined"] else None),
                "second_half_to_first_half_ratio_mean": float(np.mean(integrated_gradients_analysis["fake"]["second_half_to_first_half_ratio"])) if integrated_gradients_analysis["fake"]["second_half_to_first_half_ratio"] else None,
                "per_expert_mean_curve": {k: np.mean(np.asarray(v, dtype=float), axis=0).tolist() for k, v in integrated_gradients_analysis["fake"]["per_expert"].items() if v},
                "per_expert_std_curve": {k: np.std(np.asarray(v, dtype=float), axis=0).tolist() for k, v in integrated_gradients_analysis["fake"]["per_expert"].items() if v}
            }
        },
        "cross_referenced_insights": cross_referenced_insights,
        "summary": {
            "n_real_samples": temporal_patterns["real"]["n_samples"],
            "n_fake_samples": temporal_patterns["fake"]["n_samples"],
            "total_samples": temporal_patterns["real"]["n_samples"] + temporal_patterns["fake"]["n_samples"]
        }
    }
    
    output_file = output_dir / "global_xai_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Real: {results['summary']['n_real_samples']}")
    print(f"  Fake: {results['summary']['n_fake_samples']}")
    print(f"  High-risk words: {len(linguistic_patterns['high_risk_words'])}")
    print(f"  Problematic phonemes: {len(linguistic_patterns.get('phoneme_analysis', []))}")
    
    if linguistic_patterns['high_risk_words']:
        print("\nTop 5 High-Risk Words:")
        for i, w in enumerate(linguistic_patterns['high_risk_words'][:5], 1):
            print(f"  {i}. {w['word']} (score: {w['avg_score_when_present']:.3f}, n={w['occurrences']})")
    
    if linguistic_patterns.get('phoneme_analysis'):
        print("\nTop 5 Problematic Phonemes:")
        for i, p in enumerate(linguistic_patterns['phoneme_analysis'][:5], 1):
            print(f"  {i}. '{p['phoneme']}' - Fake: {p['mispronounced_pct_fake']:.1f}%, Real: {p['mispronounced_pct_real']:.1f}%")


if __name__ == "__main__":
    main()
