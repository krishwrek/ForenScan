"""
Edge Analysis module for ForenScan.
Evaluates boundary consistency using Canny edge detection, Sobel filters,
and Laplacian operators. Detects unnaturally sharp borders (scissor pasting),
excessive smoothing/feathering halos, and localized focus/sharpness discrepancies.
"""

from typing import Dict, Any, List
import numpy as np
import cv2


def analyze_edges(
    gray_image: np.ndarray,
    block_size: int = 32,
    sensitivity: float = 1.0
) -> Dict[str, Any]:
    """
    Perform multi-operator edge and boundary sharpness analysis.
    
    Parameters:
        gray_image: np.ndarray uint8 of shape (H, W)
        block_size: Size of local analysis window (default: 32)
        sensitivity: Sensitivity modifier (0.5 to 2.0)
        
    Returns:
        dict containing:
            - canny_edges: np.ndarray uint8 (H, W)
            - sobel_magnitude: np.ndarray uint8 (H, W)
            - laplacian: np.ndarray uint8 (H, W)
            - sharpness_heatmap: np.ndarray uint8 RGB (H, W, 3)
            - anomaly_mask: np.ndarray uint8 (H, W)
            - mean_edge_density: float
            - max_sharpness: float
            - mean_sharpness: float
            - sharpness_discrepancy: float (ratio of max to median local sharpness)
            - anomalies: list of detected issues
    """
    h, w = gray_image.shape

    # 1. Canny edge detection with adaptive thresholds based on median pixel intensity
    v = np.median(gray_image)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    canny_edges = cv2.Canny(gray_image, lower_thresh, upper_thresh)

    # 2. Sobel Filter Gradients
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    norm_sobel = cv2.normalize(grad_mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    sobel_uint8 = np.clip(norm_sobel, 0, 255).astype(np.uint8)

    # 3. Laplacian Operator (Second Derivative)
    laplacian_raw = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian_raw)
    norm_laplacian = cv2.normalize(laplacian_abs, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    laplacian_uint8 = np.clip(norm_laplacian, 0, 255).astype(np.uint8)

    # 4. Regional Edge Density & Sharpness per Block
    n_by = max(1, h // block_size)
    n_bx = max(1, w // block_size)
    
    density_grid = np.zeros((n_by, n_bx), dtype=np.float32)
    sharpness_grid = np.zeros((n_by, n_bx), dtype=np.float32)

    for by in range(n_by):
        y_start = by * block_size
        y_end = min(h, y_start + block_size)
        for bx in range(n_bx):
            x_start = bx * block_size
            x_end = min(w, x_start + block_size)

            blk_canny = canny_edges[y_start:y_end, x_start:x_end]
            blk_mag = grad_mag[y_start:y_end, x_start:x_end]

            edge_count = np.sum(blk_canny > 0)
            density = float(edge_count) / float(blk_canny.size)
            density_grid[by, bx] = density

            # Sharpness is average gradient magnitude along detected edges
            if edge_count > 5:
                edge_pixels_mag = blk_mag[blk_canny > 0]
                sharpness_grid[by, bx] = float(np.mean(edge_pixels_mag))
            else:
                sharpness_grid[by, bx] = float(np.mean(blk_mag))

    # 5. Global sharpness stats & Discrepancy metric
    active_edge_blocks = sharpness_grid[density_grid > 0.02]
    if len(active_edge_blocks) > 0:
        mean_sharpness = float(np.mean(active_edge_blocks))
        med_sharpness = float(np.median(active_edge_blocks))
        max_sharpness = float(np.max(active_edge_blocks))
    else:
        mean_sharpness = float(np.mean(sharpness_grid))
        med_sharpness = float(np.median(sharpness_grid))
        max_sharpness = float(np.max(sharpness_grid))

    mean_density = float(np.mean(density_grid))

    # Sharpness discrepancy ratio relative to active edge context
    sharpness_discrepancy = max_sharpness / (max(8.0, med_sharpness))

    # 6. Anomaly Detection (Extreme local edge sharpness spikes or artificial softness)
    # Natural depth-of-field has gradual transitions. Splicing often creates abrupt sharpness jumps.
    sharpness_std = float(np.std(sharpness_grid))
    thresh_high = med_sharpness + (2.5 / max(0.2, sensitivity)) * (sharpness_std + 1e-4)

    anomalous_blocks = (sharpness_grid > thresh_high).astype(np.uint8)
    
    # Resize sharpness grid and anomaly mask to full image
    sharpness_map_resized = cv2.resize(sharpness_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    norm_sharp = cv2.normalize(sharpness_map_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    sharpness_heatmap_bgr = cv2.applyColorMap(norm_sharp.astype(np.uint8), cv2.COLORMAP_INFERNO)
    sharpness_heatmap = cv2.cvtColor(sharpness_heatmap_bgr, cv2.COLOR_BGR2RGB)

    anomaly_mask = cv2.resize(anomalous_blocks * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    anomaly_mask = cv2.morphologyEx(
        anomaly_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    anomalies: List[str] = []
    if sharpness_discrepancy > 4.5:
        anomalies.append(
            f"Unnatural edge sharpness ratio ({round(sharpness_discrepancy, 2)}x above median). "
            "Sharp transition boundary detected without gradual optical depth-of-field falloff."
        )
    if mean_density < 0.015 and mean_sharpness < 8.0:
        anomalies.append(
            "Abnormally low edge density and gradient softness throughout the image. "
            "Typical of synthetic diffusion generation, heavy AI smoothing, or aggressive bilateral filtering."
        )

    return {
        "canny_edges": canny_edges,
        "sobel_magnitude": sobel_uint8,
        "laplacian": laplacian_uint8,
        "sharpness_heatmap": sharpness_heatmap,
        "anomaly_mask": anomaly_mask,
        "mean_edge_density": round(mean_density * 100, 2),
        "mean_sharpness": round(mean_sharpness, 2),
        "max_sharpness": round(max_sharpness, 2),
        "sharpness_discrepancy": round(sharpness_discrepancy, 2),
        "anomalies": anomalies
    }
