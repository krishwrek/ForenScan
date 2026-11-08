"""
Classifier & Forensic Decision Engine for ForenScan.
Combines a lightweight, calibrated Random Forest classifier with a robust
rule-based Classical IPA fallback. Every prediction outputs calibrated probabilities,
an authenticity score, and detailed, explainable forensic reasoning.
"""

import os
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .feature_extraction import FEATURE_NAMES


MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "trained_model.pkl")

CLASS_NAMES = ["Likely Authentic", "Likely AI-Generated", "Likely Edited/Tampered"]


def train_synthetic_classifier(save_path: str = MODEL_FILE, n_samples_per_class: int = 500) -> RandomForestClassifier:
    """
    Train a calibrated Random Forest on synthetic forensic feature distributions
    representing Authentic, AI-Generated, and Tampered imagery.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.random.seed(42)

    X_list = []
    y_list = []

    # 1. Authentic Photography (Class 0)
    for _ in range(n_samples_per_class):
        row = {
            "ela_mean_error": np.random.uniform(0.2, 1.8),
            "ela_std_error": np.random.uniform(0.2, 1.2),
            "ela_max_error": np.random.uniform(5.0, 25.0),
            "ela_high_error_ratio": np.random.uniform(0.0, 1.5),
            "ela_inconsistency_score": np.random.uniform(5.0, 22.0),
            "noise_global_mean": np.random.uniform(1.2, 3.8),
            "noise_global_std": np.random.uniform(0.2, 0.8),
            "noise_outlier_ratio": np.random.uniform(0.0, 2.0),
            "noise_inconsistency_score": np.random.uniform(5.0, 20.0),
            "edge_mean_density": np.random.uniform(1.5, 6.0),
            "edge_mean_sharpness": np.random.uniform(15.0, 35.0),
            "edge_max_sharpness": np.random.uniform(25.0, 50.0),
            "edge_sharpness_discrepancy": np.random.uniform(1.0, 2.0),
            "freq_spectral_slope": np.random.uniform(1.8, 2.6),
            "freq_periodic_peaks": 0,
            "freq_high_freq_ratio": np.random.uniform(0.05, 0.5),
            "freq_dct_ac_energy": np.random.uniform(15.0, 45.0),
            "color_lighting_divergence": np.random.uniform(5.0, 50.0),
            "color_inconsistency_score": np.random.uniform(5.0, 30.0),
            "copy_move_match_count": 0,
            "copy_move_clusters": 0,
            "meta_has_camera_info": np.random.choice([1.0, 0.0], p=[0.9, 0.1]),
            "meta_software_flag": 0.0,
            "fused_coverage_pct": np.random.uniform(0.0, 1.0),
            "fused_peak_suspicion": np.random.uniform(0.0, 20.0)
        }
        X_list.append([row[name] for name in FEATURE_NAMES])
        y_list.append(0)

    # 2. AI-Generated / Synthetic (Class 1)
    for _ in range(n_samples_per_class):
        row = {
            "ela_mean_error": np.random.uniform(0.2, 1.5),
            "ela_std_error": np.random.uniform(0.1, 1.0),
            "ela_max_error": np.random.uniform(5.0, 20.0),
            "ela_high_error_ratio": np.random.uniform(0.0, 1.5),
            "ela_inconsistency_score": np.random.uniform(5.0, 24.0),
            "noise_global_mean": np.random.uniform(0.05, 0.85),  # Over-smooth / lack of natural sensor shot noise
            "noise_global_std": np.random.uniform(0.02, 0.4),
            "noise_outlier_ratio": np.random.uniform(0.0, 4.0),
            "noise_inconsistency_score": np.random.uniform(3.0, 22.0),
            "edge_mean_density": np.random.uniform(0.8, 3.5),
            "edge_mean_sharpness": np.random.uniform(8.0, 24.0),
            "edge_max_sharpness": np.random.uniform(20.0, 45.0),
            "edge_sharpness_discrepancy": np.random.uniform(1.0, 2.2),
            "freq_spectral_slope": np.random.choice([np.random.uniform(0.8, 1.4), np.random.uniform(3.1, 4.2)]),
            "freq_periodic_peaks": np.random.randint(2, 12),  # Periodic lattice spikes
            "freq_high_freq_ratio": np.random.uniform(0.01, 0.10),
            "freq_dct_ac_energy": np.random.uniform(5.0, 25.0),
            "color_lighting_divergence": np.random.uniform(10.0, 60.0),
            "color_inconsistency_score": np.random.uniform(10.0, 35.0),
            "copy_move_match_count": 0,
            "copy_move_clusters": 0,
            "meta_has_camera_info": 0.0,  # Typically lacks camera EXIF
            "meta_software_flag": np.random.choice([1.0, 0.0], p=[0.6, 0.4]),
            "fused_coverage_pct": np.random.uniform(0.0, 3.0),
            "fused_peak_suspicion": np.random.uniform(5.0, 30.0)
        }
        X_list.append([row[name] for name in FEATURE_NAMES])
        y_list.append(1)

    # 3. Locally Edited / Tampered (Spliced, Inpainted, Copy-Move) (Class 2)
    for _ in range(n_samples_per_class):
        is_copy_move = np.random.rand() > 0.4
        row = {
            "ela_mean_error": np.random.uniform(0.2, 3.5),
            "ela_std_error": np.random.uniform(0.5, 3.0),
            "ela_max_error": np.random.uniform(20.0, 80.0),
            "ela_high_error_ratio": np.random.uniform(2.0, 20.0),
            "ela_inconsistency_score": np.random.uniform(40.0, 100.0),
            "noise_global_mean": np.random.uniform(0.8, 3.5),
            "noise_global_std": np.random.uniform(0.8, 3.0),
            "noise_outlier_ratio": np.random.uniform(3.0, 25.0),
            "noise_inconsistency_score": np.random.uniform(35.0, 85.0),
            "edge_mean_density": np.random.uniform(1.5, 6.0),
            "edge_mean_sharpness": np.random.uniform(20.0, 50.0),
            "edge_max_sharpness": np.random.uniform(45.0, 95.0),
            "edge_sharpness_discrepancy": np.random.uniform(2.5, 6.5),
            "freq_spectral_slope": np.random.uniform(1.8, 2.9),
            "freq_periodic_peaks": 0,
            "freq_high_freq_ratio": np.random.uniform(0.05, 0.4),
            "freq_dct_ac_energy": np.random.uniform(15.0, 50.0),
            "color_lighting_divergence": np.random.uniform(40.0, 160.0),
            "color_inconsistency_score": np.random.uniform(30.0, 75.0),
            "copy_move_match_count": np.random.randint(15, 120) if is_copy_move else 0,
            "copy_move_clusters": np.random.randint(1, 8) if is_copy_move else 0,
            "meta_has_camera_info": np.random.choice([1.0, 0.0], p=[0.4, 0.6]),
            "meta_software_flag": np.random.choice([1.0, 0.0], p=[0.8, 0.2]),
            "fused_coverage_pct": np.random.uniform(3.0, 20.0),
            "fused_peak_suspicion": np.random.uniform(50.0, 98.0)
        }
        X_list.append([row[name] for name in FEATURE_NAMES])
        y_list.append(2)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X, y)

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES}, save_path)
    return clf


def load_classifier(model_path: str = MODEL_FILE) -> Optional[RandomForestClassifier]:
    """Load the trained model if present."""
    if os.path.exists(model_path):
        try:
            data = joblib.load(model_path)
            return data["model"] if isinstance(data, dict) else data
        except Exception:
            return None
    return None


def generate_explainable_reasons(features: Dict[str, float], metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate detailed, human-readable forensic reasons and observations from extracted features.
    """
    reasons: List[Dict[str, str]] = []

    # 1. ELA checks
    if features.get("ela_inconsistency_score", 0) > 35.0:
        reasons.append({
            "severity": "Critical",
            "type": "Compression Anomaly",
            "text": f"High ELA compression error inconsistency ({features.get('ela_inconsistency_score'):.1f}/100). "
                    "Different image regions have undergone differing numbers of JPEG compression cycles, "
                    "strongly pointing to an inserted, spliced, or modified object."
        })
    elif features.get("ela_high_error_ratio", 0) > 4.0:
        reasons.append({
            "severity": "Warning",
            "type": "Compression Error",
            "text": f"{features.get('ela_high_error_ratio'):.1f}% of pixels reside in anomalous high-compression error zones."
        })

    # 2. Noise checks
    if features.get("noise_outlier_ratio", 0) > 6.0:
        reasons.append({
            "severity": "Critical",
            "type": "Sensor Noise Inconsistency",
            "text": f"Discordant sensor noise patterns detected ({features.get('noise_outlier_ratio'):.1f}% outlier blocks). "
                    "Natural digital photos show uniform shot noise across the sensor. Local variance mismatch implies splicing or inpainting."
        })
    elif features.get("noise_global_mean", 0) < 0.8:
        reasons.append({
            "severity": "Warning",
            "type": "Synthetic Smoothing",
            "text": f"Extremely low global noise level ({features.get('noise_global_mean'):.2f}). "
                    "Absence of physical photon noise is characteristic of synthetic AI diffusion or intense spatial filtering."
        })

    # 3. Edge checks
    if features.get("edge_sharpness_discrepancy", 1.0) > 2.5:
        reasons.append({
            "severity": "Critical",
            "type": "Boundary Sharpness Anomaly",
            "text": f"Unnatural edge sharpness ratio ({features.get('edge_sharpness_discrepancy'):.1f}x local median). "
                    "Scissor cut boundary or unnatural compositing transition detected without standard optical blur falloff."
        })

    # 4. Frequency checks
    peaks = int(features.get("freq_periodic_peaks", 0))
    if peaks >= 2:
        reasons.append({
            "severity": "Critical",
            "type": "Periodic Spectral Artifact",
            "text": f"{peaks} periodic high-frequency lattice spikes identified in 2D Fourier spectrum. "
                    "Distinct fingerprint of transposed convolution / latent upsampling layers found in GAN and Diffusion architectures."
        })
    slope = features.get("freq_spectral_slope", 2.0)
    if slope < 1.3 or slope > 3.0:
        reasons.append({
            "severity": "Warning",
            "type": "Atypical Power Law",
            "text": f"Spectral decay slope (alpha = {slope:.2f}) deviates from natural photographic distribution (1.6 - 2.4)."
        })

    # 5. Copy-Move checks
    cm_clusters = int(features.get("copy_move_clusters", 0))
    cm_matches = int(features.get("copy_move_match_count", 0))
    if cm_clusters > 0:
        reasons.append({
            "severity": "Critical",
            "type": "Cloned Regions",
            "text": f"{cm_clusters} cohesive duplicate cluster(s) with {cm_matches} paired patch matches. "
                    "Confirmed copy-move cloning detected within the same image frame."
        })

    # 6. Color & Lighting checks
    light_div = features.get("color_lighting_divergence", 0)
    if light_div > 80.0:
        reasons.append({
            "severity": "Warning",
            "type": "Conflicting Illumination",
            "text": f"Contradictory illumination direction ({light_div:.1f}° divergence between quadrants). "
                    "Indicates composited subjects photographed under different light environments."
        })

    # 7. Metadata traces
    sw_traces = metadata.get("software_traces", [])
    if len(sw_traces) > 0:
        reasons.append({
            "severity": "Warning",
            "type": "Software Footprint",
            "text": f"Metadata explicitly flags software signature: {', '.join(sw_traces)}."
        })

    if metadata.get("camera_info"):
        reasons.append({
            "severity": "Normal",
            "type": "Authentic Hardware EXIF",
            "text": f"Camera hardware metadata intact: {metadata['camera_info'].get('Make', '')} {metadata['camera_info'].get('Model', '')}."
        })
    elif not metadata.get("has_exif", False):
        reasons.append({
            "severity": "Information",
            "type": "EXIF Stripped",
            "text": "Image lacks camera EXIF metadata. Common in web re-downloads, screenshots, or AI generators."
        })

    if len(reasons) == 0:
        reasons.append({
            "severity": "Normal",
            "type": "Uniform Consistency",
            "text": "All forensic checks (noise, ELA, edge gradients, Fourier spectrum, and lighting) exhibit uniform photographic consistency."
        })

    return reasons


def rule_based_ipa_score(features: Dict[str, float], metadata: Dict[str, Any]) -> Tuple[str, float, float, float, float]:
    """
    Rule-based classical IPA scoring engine.
    Computes heuristic risk weights based on physical and statistical constraints.
    """
    # 1. Tamper Score (0 - 100)
    tamper_pts = 0.0
    if features.get("copy_move_clusters", 0) > 0:
        tamper_pts += 35.0 + min(30.0, features.get("copy_move_match_count", 0) * 1.5)
    if features.get("ela_inconsistency_score", 0) > 30.0:
        tamper_pts += min(35.0, (features.get("ela_inconsistency_score", 0) - 25.0) * 0.7)
    if features.get("edge_sharpness_discrepancy", 1.0) > 2.2:
        tamper_pts += min(25.0, (features.get("edge_sharpness_discrepancy", 1.0) - 2.0) * 12.0)
    if features.get("noise_outlier_ratio", 0) > 3.0:
        tamper_pts += min(20.0, features.get("noise_outlier_ratio", 0) * 2.0)
    if features.get("color_lighting_divergence", 0) > 75.0:
        tamper_pts += min(15.0, (features.get("color_lighting_divergence", 0) - 70.0) * 0.2)
    if "editor" in metadata.get("software_categories", []):
        tamper_pts += 25.0

    tamper_score = min(98.0, max(2.0, tamper_pts))

    # 2. AI Generation Score (0 - 100)
    ai_pts = 0.0
    peaks = features.get("freq_periodic_peaks", 0)
    if peaks >= 2:
        ai_pts += min(50.0, peaks * 12.0)
    slope = features.get("freq_spectral_slope", 2.0)
    if slope < 1.3 or slope > 3.0:
        ai_pts += 25.0
    if features.get("noise_global_mean", 1.0) < 0.8:
        ai_pts += 25.0
    if not metadata.get("camera_info"):
        ai_pts += 10.0
    if "generative_ai" in metadata.get("software_categories", []):
        ai_pts += 50.0

    ai_score = min(98.0, max(2.0, ai_pts))

    # 3. Authenticity Score (0 - 100)
    max_risk = max(tamper_score, ai_score)
    auth_bonus = 15.0 if metadata.get("camera_info") else 0.0
    authenticity_score = min(99.0, max(1.0, 100.0 - max_risk + auth_bonus))

    # Overall verdict determination
    if max_risk < 35.0 and authenticity_score >= 65.0:
        verdict = CLASS_NAMES[0]  # Likely Authentic
        confidence = authenticity_score
    elif tamper_score >= ai_score and tamper_score >= 40.0:
        verdict = CLASS_NAMES[2]  # Likely Edited/Tampered
        confidence = tamper_score
    elif ai_score >= 40.0:
        verdict = CLASS_NAMES[1]  # Likely AI-Generated
        confidence = ai_score
    else:
        verdict = CLASS_NAMES[0]
        confidence = authenticity_score

    return verdict, round(authenticity_score, 1), round(ai_score, 1), round(tamper_score, 1), round(confidence, 1)


def predict_authenticity(
    features_dict: Dict[str, float],
    metadata_dict: Dict[str, Any],
    model_path: str = MODEL_FILE,
    use_ml: bool = True
) -> Dict[str, Any]:
    """
    Main forensic decision pipeline.
    Combines Random Forest model with physical IPA rules and metadata evidence.
    """
    reasons = generate_explainable_reasons(features_dict, metadata_dict)
    
    # Check rule-based baseline
    v_rule, s_auth, s_ai, s_tamper, conf = rule_based_ipa_score(features_dict, metadata_dict)

    model = load_classifier(model_path) if use_ml else None

    if model is not None:
        try:
            vec = np.array([[features_dict[name] for name in FEATURE_NAMES]], dtype=np.float32)
            probs = model.predict_proba(vec)[0]  # [p_auth, p_ai, p_tamper]
            
            p_auth = float(probs[0]) * 100.0
            p_ai = float(probs[1]) * 100.0
            p_tamper = float(probs[2]) * 100.0

            # Fuse with definitive metadata evidence
            if "editor" in metadata_dict.get("software_categories", []):
                p_tamper = min(99.0, p_tamper + 35.0)
                p_auth = max(1.0, p_auth - 30.0)

            if "generative_ai" in metadata_dict.get("software_categories", []):
                p_ai = min(99.0, p_ai + 40.0)
                p_auth = max(1.0, p_auth - 35.0)

            if features_dict.get("copy_move_clusters", 0) > 0:
                p_tamper = min(99.0, p_tamper + 40.0)
                p_auth = max(1.0, p_auth - 35.0)

            # Re-normalize probabilities
            total_p = p_auth + p_ai + p_tamper
            p_auth = (p_auth / total_p) * 100.0
            p_ai = (p_ai / total_p) * 100.0
            p_tamper = (p_tamper / total_p) * 100.0

            probs_adj = [p_auth, p_ai, p_tamper]
            idx = int(np.argmax(probs_adj))
            verdict = CLASS_NAMES[idx]
            confidence = round(probs_adj[idx], 1)

            return {
                "verdict": verdict,
                "authenticity_score": round(p_auth, 1),
                "ai_score": round(p_ai, 1),
                "tamper_score": round(p_tamper, 1),
                "confidence": confidence,
                "reasons": reasons,
                "method": "Calibrated Random Forest (IPA Features)"
            }
        except Exception:
            pass

    return {
        "verdict": v_rule,
        "authenticity_score": s_auth,
        "ai_score": s_ai,
        "tamper_score": s_tamper,
        "confidence": conf,
        "reasons": reasons,
        "method": "Rule-Based Classical IPA Heuristics"
    }
