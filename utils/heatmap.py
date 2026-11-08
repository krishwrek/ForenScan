"""
Fused Suspicious-Region Heatmap module for ForenScan.
Fuses multi-modal forensic cues (Error Level Analysis, Noise Inconsistency,
Edge Sharpness Anomalies, and Copy-Move Duplicates) into a unified,
calibrated probability heatmap and visual overlay.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2


def generate_fused_heatmap(
    image_rgb: np.ndarray,
    ela_array: np.ndarray,
    noise_anomaly: np.ndarray,
    edge_anomaly: np.ndarray,
    copy_move_mask: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    alpha: float = 0.5,
    threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Fuse multiple forensic analysis maps into a single cohesive suspicious-region overlay.
    
    Parameters:
        image_rgb: np.ndarray uint8 of shape (H, W, 3)
        ela_array: 2D array of ELA intensity (uint8 or float)
        noise_anomaly: 2D array of noise inconsistency (uint8 or float)
        edge_anomaly: 2D array of edge boundary anomalies (uint8 or float)
        copy_move_mask: Optional 2D array of copy-move matches (uint8 0-255)
        weights: Dictionary of modality weights (ela, noise, edge, copy_move)
        alpha: Blending transparency factor (0.0 to 1.0)
        threshold: Sensitivity threshold (0.0 to 1.0) for bounding box extraction
        
    Returns:
        dict containing:
            - fused_heatmap: np.ndarray uint8 RGB (H, W, 3)
            - overlay_image: np.ndarray uint8 RGB (H, W, 3)
            - annotated_image: np.ndarray uint8 RGB with bounding boxes and labels
            - binary_mask: np.ndarray uint8 (H, W) (0 or 255)
            - bounding_boxes: list of dicts with box coords and peak confidence
            - anomaly_coverage_pct: float % of image area flagged
            - peak_suspicion: float (0 - 100)
    """
    h, w, _ = image_rgb.shape

    # Default weights
    if weights is None:
        weights = {
            "ela": 0.35,
            "noise": 0.25,
            "edge": 0.20,
            "copy_move": 0.20
        }

    # Normalize helper
    def _norm(arr):
        if arr is None or arr.size == 0:
            return np.zeros((h, w), dtype=np.float32)
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        f_arr = arr.astype(np.float32)
        max_v = np.max(f_arr)
        min_v = np.min(f_arr)
        if max_v - min_v > 1e-5:
            return (f_arr - min_v) / (max_v - min_v)
        return np.zeros((h, w), dtype=np.float32)

    n_ela = _norm(ela_array)
    n_noise = _norm(noise_anomaly)
    n_edge = _norm(edge_anomaly)
    n_copy = _norm(copy_move_mask)

    # Weighted linear combination
    w_sum = sum(weights.values())
    fused_float = (
        weights.get("ela", 0.35) * n_ela +
        weights.get("noise", 0.25) * n_noise +
        weights.get("edge", 0.20) * n_edge +
        weights.get("copy_move", 0.20) * n_copy
    ) / max(1e-4, w_sum)

    # Spatial smoothing to cluster local cues into coherent regions
    fused_smoothed = cv2.GaussianBlur(fused_float, (15, 15), 0)
    fused_smoothed = np.clip(fused_smoothed, 0.0, 1.0)

    # Generate color heatmap (JET or TURBO)
    fused_uint8 = (fused_smoothed * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(fused_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Adaptive alpha overlay: unaffected regions remain original, suspicious regions get blended
    alpha_mask = np.clip((fused_smoothed - 0.1) / 0.9, 0.0, 1.0)[:, :, np.newaxis]
    eff_alpha = alpha_mask * alpha
    
    overlay = (
        image_rgb.astype(np.float32) * (1.0 - eff_alpha) +
        heatmap_rgb.astype(np.float32) * eff_alpha
    ).astype(np.uint8)

    # Binary mask of high suspicion regions
    bin_thresh = int(threshold * 255)
    _, binary_mask = cv2.threshold(fused_uint8, bin_thresh, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Extract contours and bounding boxes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image_rgb.copy()
    bounding_boxes: List[Dict[str, Any]] = []

    min_area = (h * w) * 0.005  # Filter out tiny specks (< 0.5% image area)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Compute peak suspicion inside contour
            roi = fused_smoothed[y:y+bh, x:x+bw]
            peak_conf = float(np.max(roi)) if roi.size > 0 else 0.0

            bounding_boxes.append({
                "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                "area": int(area), "confidence": round(peak_conf * 100, 1)
            })

            # Draw glowing bounding box and label
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 30, 30), 2)
            label = f"Suspicious ({round(peak_conf * 100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, max(0, y - th - 6)), (x + tw + 6, max(th + 6, y)), (255, 30, 30), -1)
            cv2.putText(
                annotated, label, (x + 3, max(th, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

    coverage_pct = float(np.sum(binary_mask > 0) / binary_mask.size) * 100.0
    peak_suspicion = float(np.max(fused_smoothed)) * 100.0

    return {
        "fused_heatmap": heatmap_rgb,
        "overlay_image": overlay,
        "annotated_image": annotated,
        "binary_mask": binary_mask,
        "bounding_boxes": bounding_boxes,
        "anomaly_coverage_pct": round(coverage_pct, 2),
        "peak_suspicion": round(peak_suspicion, 1)
    }
