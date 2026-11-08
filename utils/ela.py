"""
Error Level Analysis (ELA) module for ForenScan.
Re-saves the image at a known JPEG quality (default 95) and computes the
recompression error. Spliced or locally edited elements typically show
distinctly higher or lower compression error compared to the original substrate.
"""

import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2


def compute_ela(
    image_rgb: np.ndarray,
    quality: int = 95,
    scale: float = 20.0,
    anomaly_threshold_std: float = 2.0
) -> Dict[str, Any]:
    """
    Perform Error Level Analysis (ELA) on an RGB image.
    
    Parameters:
        image_rgb: np.ndarray uint8 of shape (H, W, 3)
        quality: JPEG compression quality level (default: 95)
        scale: Amplification factor for visualization (default: 20.0)
        anomaly_threshold_std: Number of standard deviations above mean to flag as anomaly
        
    Returns:
        dict containing:
            - ela_image: Amplified RGB error representation (np.ndarray uint8)
            - ela_gray: Grayscale error representation (0-255)
            - ela_heatmap: Colored JET colormap of error (np.ndarray uint8 RGB)
            - anomaly_mask: Binary mask of high compression-error anomalies (0 or 255)
            - mean_error: float mean pixel difference
            - std_error: float standard deviation of pixel difference
            - max_error: float maximum pixel difference
            - high_error_ratio: percentage of pixels in high error regions
            - inconsistency_score: normalized score (0 - 100) of local error variance
    """
    # 1. Re-save original image at target JPEG quality in-memory
    pil_orig = Image.fromarray(image_rgb, mode="RGB")
    buffer = io.BytesIO()
    pil_orig.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    # 2. Re-open recompressed image
    recompressed_pil = Image.open(buffer)
    recompressed_rgb = np.array(recompressed_pil, dtype=np.float32)
    orig_rgb_float = image_rgb.astype(np.float32)

    # 3. Calculate absolute difference across color channels
    diff = np.abs(orig_rgb_float - recompressed_rgb)
    
    # Average across channels for grayscale intensity
    diff_gray = np.mean(diff, axis=2)

    # 4. Amplify difference for visual inspection
    ela_amplified = np.clip(diff * scale, 0, 255).astype(np.uint8)
    
    # Grayscale amplified
    ela_gray = np.clip(diff_gray * scale, 0, 255).astype(np.uint8)

    # 5. Generate Colormap Heatmap (JET colormap, converted to RGB)
    # Normalize diff_gray for colormap
    norm_diff = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_diff_uint8 = norm_diff.astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(norm_diff_uint8, cv2.COLORMAP_JET)
    ela_heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 6. Statistical measurements
    mean_val = float(np.mean(diff_gray))
    std_val = float(np.std(diff_gray))
    max_val = float(np.max(diff_gray))

    # Anomaly detection: pixels exceeding mean + k * std
    thresh_val = mean_val + (anomaly_threshold_std * std_val)
    anomaly_mask = np.zeros_like(diff_gray, dtype=np.uint8)
    anomaly_mask[diff_gray > thresh_val] = 255

    # Clean up isolated noise using morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

    high_error_ratio = float(np.sum(anomaly_mask > 0) / float(anomaly_mask.size))

    # Compute block-based inconsistency score
    # Divide into 32x32 blocks and calculate variance of block means
    h, w = diff_gray.shape
    block_size = 32
    block_means = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            blk = diff_gray[y:y+block_size, x:x+block_size]
            block_means.append(np.mean(blk))
    
    if len(block_means) > 0:
        block_std = float(np.std(block_means))
        # Normalize into a 0 - 100 score
        inconsistency_score = min(100.0, (block_std / (mean_val + 1e-5)) * 50.0)
    else:
        inconsistency_score = 0.0

    return {
        "ela_image": ela_amplified,
        "ela_gray": ela_gray,
        "ela_heatmap": ela_heatmap,
        "anomaly_mask": anomaly_mask,
        "mean_error": round(mean_val, 3),
        "std_error": round(std_val, 3),
        "max_error": round(max_val, 3),
        "high_error_ratio": round(high_error_ratio * 100, 2),
        "inconsistency_score": round(inconsistency_score, 2)
    }
