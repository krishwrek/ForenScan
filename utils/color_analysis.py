"""
Color & Lighting Consistency Analysis module for ForenScan.
Evaluates global and regional RGB/HSV distributions, illumination gradients,
and color temperature balance across quadrants to detect spliced subjects
originating from disparate lighting environments.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


def analyze_color_and_lighting(
    image_rgb: np.ndarray,
    sensitivity: float = 1.0
) -> Dict[str, Any]:
    """
    Perform color histogram, illumination gradient, and regional color distribution analysis.
    
    Parameters:
        image_rgb: np.ndarray uint8 of shape (H, W, 3)
        sensitivity: Sensitivity multiplier
        
    Returns:
        dict containing:
            - rgb_histograms: dict with 'r', 'g', 'b' frequency arrays (256 elements each)
            - hsv_histograms: dict with 'h' (180), 's' (256), 'v' (256) frequency arrays
            - quadrant_stats: list of dicts describing statistics for top-left, top-right, bottom-left, bottom-right
            - lighting_divergence_angle: float max angle difference (degrees) between quadrant light vectors
            - color_inconsistency_score: float score (0 - 100)
            - lighting_inconsistency_score: float score (0 - 100)
            - anomalies: list of detected lighting or color anomalies
    """
    h, w, _ = image_rgb.shape

    # 1. RGB Histograms
    r_hist = cv2.calcHist([image_rgb], [0], None, [256], [0, 256]).flatten()
    g_hist = cv2.calcHist([image_rgb], [1], None, [256], [0, 256]).flatten()
    b_hist = cv2.calcHist([image_rgb], [2], None, [256], [0, 256]).flatten()

    # Normalize histograms to sum = 1.0
    r_hist = r_hist / (np.sum(r_hist) + 1e-6)
    g_hist = g_hist / (np.sum(g_hist) + 1e-6)
    b_hist = b_hist / (np.sum(b_hist) + 1e-6)

    # 2. HSV Histograms
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([image_hsv], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([image_hsv], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([image_hsv], [2], None, [256], [0, 256]).flatten()

    h_hist = h_hist / (np.sum(h_hist) + 1e-6)
    s_hist = s_hist / (np.sum(s_hist) + 1e-6)
    v_hist = v_hist / (np.sum(v_hist) + 1e-6)

    # 3. Quadrant Division for Regional Lighting & Color Analysis
    mid_y, mid_x = h // 2, w // 2
    quadrants = {
        "Top-Left": image_rgb[0:mid_y, 0:mid_x],
        "Top-Right": image_rgb[0:mid_y, mid_x:w],
        "Bottom-Left": image_rgb[mid_y:h, 0:mid_x],
        "Bottom-Right": image_rgb[mid_y:h, mid_x:w]
    }

    quadrant_stats: List[Dict[str, Any]] = []
    light_angles: List[float] = []
    color_temperatures: List[float] = []
    saturation_means: List[float] = []

    for name, quad_rgb in quadrants.items():
        if quad_rgb.size == 0:
            continue
        
        # Color moments
        r_mean = float(np.mean(quad_rgb[:, :, 0]))
        g_mean = float(np.mean(quad_rgb[:, :, 1]))
        b_mean = float(np.mean(quad_rgb[:, :, 2]))

        r_std = float(np.std(quad_rgb[:, :, 0]))
        g_std = float(np.std(quad_rgb[:, :, 1]))
        b_std = float(np.std(quad_rgb[:, :, 2]))

        # HSV stats
        quad_hsv = cv2.cvtColor(quad_rgb, cv2.COLOR_RGB2HSV)
        sat_mean = float(np.mean(quad_hsv[:, :, 1]))
        val_mean = float(np.mean(quad_hsv[:, :, 2]))
        saturation_means.append(sat_mean)

        # Color temperature proxy: (R - B) / (R + B + 1e-5)
        # Positive = warm, Negative = cool
        color_temp = (r_mean - b_mean) / (r_mean + b_mean + 1e-5)
        color_temperatures.append(color_temp)

        # 4. Illumination Gradient Estimation via 2D Planar Surface Fit
        # Luminance plane: Y(x, y) = a*x + b*y + c
        # The vector (a, b) indicates the primary 2D lighting direction
        lum = cv2.cvtColor(quad_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        qh, qw = lum.shape
        
        # Subsample for fast regression
        step = max(1, min(qh, qw) // 32)
        y_grid, x_grid = np.mgrid[0:qh:step, 0:qw:step]
        z_vals = lum[0:qh:step, 0:qw:step]

        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_vals.flatten()

        # Design matrix [x, y, 1]
        A = np.column_stack([x_flat, y_flat, np.ones_like(x_flat)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
            a_dir, b_dir, _ = coeffs
            angle_deg = float(np.degrees(np.arctan2(b_dir, a_dir)))
        except Exception:
            angle_deg = 0.0

        light_angles.append(angle_deg)

        quadrant_stats.append({
            "region": name,
            "mean_rgb": [round(r_mean, 1), round(g_mean, 1), round(b_mean, 1)],
            "std_rgb": [round(r_std, 1), round(g_std, 1), round(b_std, 1)],
            "mean_saturation": round(sat_mean, 1),
            "mean_brightness": round(val_mean, 1),
            "color_temperature": round(color_temp, 3),
            "lighting_angle_deg": round(angle_deg, 1)
        })

    # 5. Calculate Divergence Between Quadrants
    # Max angular difference in lighting directions
    max_angle_diff = 0.0
    for i in range(len(light_angles)):
        for j in range(i + 1, len(light_angles)):
            diff = abs(light_angles[i] - light_angles[j])
            diff = min(diff, 360.0 - diff)
            if diff > max_angle_diff:
                max_angle_diff = diff

    # Color temperature variance across quadrants
    temp_spread = float(np.max(color_temperatures) - np.min(color_temperatures))
    sat_spread = float(np.max(saturation_means) - np.min(saturation_means))

    # Normalized scores (0 - 100)
    lighting_inconsistency_score = min(100.0, (max_angle_diff / 180.0) * 100.0 * sensitivity)
    color_inconsistency_score = min(100.0, (temp_spread * 120.0 + (sat_spread / 255.0) * 60.0) * sensitivity)

    # 6. Human-readable explainable anomalies
    anomalies: List[str] = []
    if max_angle_diff > 115.0:
        anomalies.append(
            f"Severe lighting direction conflict: {round(max_angle_diff, 1)}° angular divergence between image sectors. "
            "Suggests compositing or splicing of elements photographed under contradictory light sources."
        )
    if temp_spread > 0.45:
        anomalies.append(
            f"Substantial color temperature disparity (spread: {round(temp_spread, 2)}) between quadrants. "
            "Indicates inconsistent white balance or localized color grading/retouching."
        )

    return {
        "rgb_histograms": {
            "r": r_hist.tolist(),
            "g": g_hist.tolist(),
            "b": b_hist.tolist()
        },
        "hsv_histograms": {
            "h": h_hist.tolist(),
            "s": s_hist.tolist(),
            "v": v_hist.tolist()
        },
        "quadrant_stats": quadrant_stats,
        "lighting_divergence_angle": round(max_angle_diff, 1),
        "color_inconsistency_score": round(color_inconsistency_score, 2),
        "lighting_inconsistency_score": round(lighting_inconsistency_score, 2),
        "anomalies": anomalies
    }
