"""
Noise Consistency Analysis module for ForenScan.
Camera sensors produce characteristic, uniform high-frequency sensor noise.
Inpainted, spliced, or AI-generated regions display unnatural noise variances
(over-smoothed or discordant high-frequency distributions).
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import cv2


def analyze_noise_consistency(
    gray_image: np.ndarray,
    block_size: int = 32,
    sensitivity: float = 1.0,
    z_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Perform block-based noise consistency analysis on a grayscale image.
    
    Parameters:
        gray_image: np.ndarray uint8 of shape (H, W)
        block_size: Dimension of square blocks (default: 32)
        sensitivity: Sensitivity multiplier for anomaly thresholding (0.5 to 2.0)
        z_threshold: Baseline standard deviation cutoff for outlier detection
        
    Returns:
        dict containing:
            - noise_map: 2D float array (H, W) of interpolated local noise levels
            - noise_heatmap: Colored JET colormap of local noise (np.ndarray uint8 RGB)
            - anomaly_mask: Binary mask of anomalous noise regions (0 or 255)
            - global_noise_mean: float mean noise level
            - global_noise_std: float standard deviation of block noise levels
            - outlier_block_ratio: float percentage of anomalous blocks
            - noise_inconsistency_score: float score (0 - 100)
            - anomalies: list of human-readable findings
    """
    h, w = gray_image.shape
    
    # 1. High-frequency noise residual extraction
    # Using a 3x3 median filter subtraction to isolate noise while reducing structural edges
    median_filtered = cv2.medianBlur(gray_image, 3)
    noise_residual = cv2.absdiff(gray_image, median_filtered).astype(np.float32)

    # Calculate edge mask to discount structural edges from noise calculation
    edges = cv2.Canny(gray_image, 50, 150)
    dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    flat_mask = (dilated_edges == 0)

    # 2. Divide image into grid blocks and compute local noise variance & energy
    n_blocks_y = max(1, h // block_size)
    n_blocks_x = max(1, w // block_size)
    
    block_variances = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)
    block_energies = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)

    for by in range(n_blocks_y):
        y_start = by * block_size
        y_end = min(h, y_start + block_size)
        for bx in range(n_blocks_x):
            x_start = bx * block_size
            x_end = min(w, x_start + block_size)

            blk_res = noise_residual[y_start:y_end, x_start:x_end]
            blk_flat = flat_mask[y_start:y_end, x_start:x_end]

            # Use flat regions if available to prevent edge gradients from inflating noise
            if np.sum(blk_flat) > (blk_res.size * 0.2):
                valid_pixels = blk_res[blk_flat]
            else:
                valid_pixels = blk_res.flatten()

            # Robust standard deviation via Median Absolute Deviation (MAD)
            # sigma = median(|x - median(x)|) / 0.6745
            med = np.median(valid_pixels)
            mad = np.median(np.abs(valid_pixels - med))
            sigma_robust = mad / 0.6745 if mad > 1e-4 else np.std(valid_pixels)

            energy = np.mean(valid_pixels ** 2)

            block_variances[by, bx] = float(sigma_robust)
            block_energies[by, bx] = float(energy)

    # 3. Global statistics across all blocks
    global_mean = float(np.mean(block_variances))
    global_std = float(np.std(block_variances))
    global_median = float(np.median(block_variances))

    # Adjust threshold with sensitivity
    effective_thresh = (z_threshold / max(0.2, sensitivity)) * (global_std + 1e-5)

    # 4. Identify outlier blocks (both significantly higher and lower noise)
    z_scores = np.abs(block_variances - global_median) / (global_std + 1e-5)
    outlier_blocks = (np.abs(block_variances - global_median) > effective_thresh).astype(np.uint8)

    outlier_ratio = float(np.sum(outlier_blocks) / outlier_blocks.size)

    # 5. Interpolate block grid to full image size for smooth visualization
    noise_map_resized = cv2.resize(block_variances, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize for colormap
    norm_noise = cv2.normalize(noise_map_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_noise_uint8 = np.clip(norm_noise, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(norm_noise_uint8, cv2.COLORMAP_TURBO)
    noise_heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Resize outlier mask to full image
    anomaly_mask = cv2.resize(outlier_blocks * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size // 2, block_size // 2))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

    # 6. Inconsistency score (0 - 100)
    # Natural photos typically have low block variance std relative to mean
    cv_score = (global_std / (global_mean + 1e-4)) * 100.0
    inconsistency_score = min(100.0, max(0.0, cv_score * sensitivity))

    anomalies: List[str] = []
    if outlier_ratio > 0.08:
        anomalies.append(
            f"Detected {round(outlier_ratio * 100, 1)}% anomalous noise blocks. "
            "Inconsistent sensor noise across regions strongly indicates local splicing or selective inpainting."
        )
    if global_mean < 0.35:
        anomalies.append(
            f"Extremely low global noise floor ({round(global_mean, 3)}). "
            "Natural photographic sensors exhibit shot noise; near-zero noise often indicates synthetic AI generation or heavy denoising."
        )

    return {
        "noise_map": noise_map_resized,
        "noise_heatmap": noise_heatmap,
        "anomaly_mask": anomaly_mask,
        "global_noise_mean": round(global_mean, 3),
        "global_noise_std": round(global_std, 3),
        "outlier_block_ratio": round(outlier_ratio * 100, 2),
        "noise_inconsistency_score": round(inconsistency_score, 2),
        "anomalies": anomalies
    }
