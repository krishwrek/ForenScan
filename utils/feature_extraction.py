"""
Forensic Feature Extraction module for ForenScan.
Aggregates quantitative Image Processing Analysis (IPA) metrics into a standardized
feature vector for machine-learning classification and CSV export.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "ela_mean_error",
    "ela_std_error",
    "ela_max_error",
    "ela_high_error_ratio",
    "ela_inconsistency_score",
    "noise_global_mean",
    "noise_global_std",
    "noise_outlier_ratio",
    "noise_inconsistency_score",
    "edge_mean_density",
    "edge_mean_sharpness",
    "edge_max_sharpness",
    "edge_sharpness_discrepancy",
    "freq_spectral_slope",
    "freq_periodic_peaks",
    "freq_high_freq_ratio",
    "freq_dct_ac_energy",
    "color_lighting_divergence",
    "color_inconsistency_score",
    "copy_move_match_count",
    "copy_move_clusters",
    "meta_has_camera_info",
    "meta_software_flag",
    "fused_coverage_pct",
    "fused_peak_suspicion"
]


def extract_feature_vector(
    ela_res: Dict[str, Any],
    noise_res: Dict[str, Any],
    edge_res: Dict[str, Any],
    freq_res: Dict[str, Any],
    color_res: Dict[str, Any],
    copy_move_res: Dict[str, Any],
    meta_res: Dict[str, Any],
    fused_res: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract a unified numerical feature vector and dictionary from forensic sub-analyses.
    
    Returns:
        (features_array, features_dict)
    """
    features_dict: Dict[str, float] = {
        "ela_mean_error": float(ela_res.get("mean_error", 0.0)),
        "ela_std_error": float(ela_res.get("std_error", 0.0)),
        "ela_max_error": float(ela_res.get("max_error", 0.0)),
        "ela_high_error_ratio": float(ela_res.get("high_error_ratio", 0.0)),
        "ela_inconsistency_score": float(ela_res.get("inconsistency_score", 0.0)),
        
        "noise_global_mean": float(noise_res.get("global_noise_mean", 0.0)),
        "noise_global_std": float(noise_res.get("global_noise_std", 0.0)),
        "noise_outlier_ratio": float(noise_res.get("outlier_block_ratio", 0.0)),
        "noise_inconsistency_score": float(noise_res.get("noise_inconsistency_score", 0.0)),
        
        "edge_mean_density": float(edge_res.get("mean_edge_density", 0.0)),
        "edge_mean_sharpness": float(edge_res.get("mean_sharpness", 0.0)),
        "edge_max_sharpness": float(edge_res.get("max_sharpness", 0.0)),
        "edge_sharpness_discrepancy": float(edge_res.get("sharpness_discrepancy", 1.0)),
        
        "freq_spectral_slope": float(freq_res.get("spectral_slope", 2.0)),
        "freq_periodic_peaks": float(freq_res.get("periodic_peak_count", 0)),
        "freq_high_freq_ratio": float(freq_res.get("high_freq_energy_ratio", 0.0)),
        "freq_dct_ac_energy": float(freq_res.get("dct_ac_energy", 0.0)),
        
        "color_lighting_divergence": float(color_res.get("lighting_divergence_angle", 0.0)),
        "color_inconsistency_score": float(color_res.get("color_inconsistency_score", 0.0)),
        
        "copy_move_match_count": float(copy_move_res.get("match_count", 0)),
        "copy_move_clusters": float(copy_move_res.get("cluster_count", 0)),
        
        "meta_has_camera_info": 1.0 if bool(meta_res.get("camera_info")) else 0.0,
        "meta_software_flag": 1.0 if (len(meta_res.get("software_traces", [])) > 0) else 0.0,
        
        "fused_coverage_pct": float(fused_res.get("anomaly_coverage_pct", 0.0)),
        "fused_peak_suspicion": float(fused_res.get("peak_suspicion", 0.0))
    }

    vec = np.array([features_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
    return vec, features_dict


def features_to_dataframe(features_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Format extracted forensic features into a descriptive Pandas DataFrame.
    """
    category_map = {
        "ela_": "Error Level Analysis",
        "noise_": "Sensor Noise",
        "edge_": "Edge & Sharpness",
        "freq_": "Frequency Spectrum",
        "color_": "Color & Lighting",
        "copy_move_": "Copy-Move Cloning",
        "meta_": "EXIF Metadata",
        "fused_": "Combined Anomalies"
    }

    rows = []
    for k, v in features_dict.items():
        cat = "General"
        for prefix, cname in category_map.items():
            if k.startswith(prefix):
                cat = cname
                break
        
        # Clean human label
        clean_name = k.replace("_", " ").title()
        rows.append({
            "Category": cat,
            "Feature": clean_name,
            "Value": round(float(v), 3)
        })

    return pd.DataFrame(rows)
