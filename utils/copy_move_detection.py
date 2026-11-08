"""
Copy-Move Forgery Detection module for ForenScan.
Divides the image into overlapping patches, extracts color moments and texture
descriptors, identifies non-adjacent similar patches, applies shift-vector consistency
clustering, and visually highlights forged duplicates.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors


def extract_patch_features(patch_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a compact 12-dimensional feature vector from an RGB patch:
    - 6 color moments: mean & std of Y, Cr, Cb
    - 4 directional gradient energies
    - 2 high-frequency texture moments
    """
    # Color moments in YCrCb
    patch_ycrcb = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_mean, cr_mean, cb_mean = np.mean(patch_ycrcb, axis=(0, 1))
    y_std, cr_std, cb_std = np.std(patch_ycrcb, axis=(0, 1))

    # Gradients in grayscale
    patch_gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    
    g_horiz = np.mean(np.abs(gx))
    g_vert = np.mean(np.abs(gy))
    g_diag1 = np.mean(np.abs(gx + gy))
    g_diag2 = np.mean(np.abs(gx - gy))

    # Laplacian texture energy
    lap = cv2.Laplacian(patch_gray, cv2.CV_32F)
    lap_mean = np.mean(np.abs(lap))
    lap_std = np.std(lap)

    feats = np.array([
        y_mean / 255.0, cr_mean / 255.0, cb_mean / 255.0,
        y_std / 128.0, cr_std / 128.0, cb_std / 128.0,
        g_horiz / 128.0, g_vert / 128.0, g_diag1 / 180.0, g_diag2 / 180.0,
        lap_mean / 128.0, lap_std / 128.0
    ], dtype=np.float32)

    return feats


def detect_copy_move(
    image_rgb: np.ndarray,
    patch_size: int = 16,
    stride: int = 8,
    min_spatial_dist: int = 40,
    similarity_threshold: float = 0.08,
    min_cluster_size: int = 4,
    sensitivity: float = 1.0
) -> Dict[str, Any]:
    """
    Detect copy-move tampering using patch-based feature matching and shift-vector clustering.
    
    Parameters:
        image_rgb: np.ndarray uint8 of shape (H, W, 3)
        patch_size: Square patch dimension (default: 16)
        stride: Stride step between patches (default: 8)
        min_spatial_dist: Minimum pixel Euclidean distance between matching patches
        similarity_threshold: Maximum normalized feature distance to consider a match
        min_cluster_size: Minimum number of coherent pairs sharing a displacement vector
        sensitivity: Sensitivity multiplier
        
    Returns:
        dict containing:
            - visualization_image: RGB image showing matches, bounding boxes, and connection lines
            - copy_move_mask: Binary mask of duplicated regions (0 or 255)
            - match_count: int confirmed paired patch matches
            - cluster_count: int detected clone clusters
            - suspicion_score: float score (0 - 100)
            - anomalies: list of detected clone issues
    """
    h_orig, w_orig, _ = image_rgb.shape

    # For fast and robust CPU execution, work on scaled representation if image is large
    target_dim = 384
    if max(h_orig, w_orig) > target_dim:
        scale = target_dim / float(max(h_orig, w_orig))
        sw = max(1, int(w_orig * scale))
        sh = max(1, int(h_orig * scale))
        work_img = cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        sw, sh = w_orig, h_orig
        work_img = image_rgb.copy()

    # Effective parameters in scaled space
    eff_patch_size = max(12, int(patch_size * scale))
    eff_stride = max(4, int(stride * scale))
    eff_min_dist = max(20, int(min_spatial_dist * scale))
    eff_sim_thresh = similarity_threshold * max(0.5, sensitivity)

    patches_coords: List[Tuple[int, int]] = []
    features_list: List[np.ndarray] = []

    # 1. Extract features from informative patches
    gray_work = cv2.cvtColor(work_img, cv2.COLOR_RGB2GRAY)
    for y in range(0, sh - eff_patch_size + 1, eff_stride):
        for x in range(0, sw - eff_patch_size + 1, eff_stride):
            patch_gray = gray_work[y:y+eff_patch_size, x:x+eff_patch_size]
            # Skip flat low-contrast patches (sky, solid colors)
            if np.std(patch_gray) < 6.0:
                continue

            patch_rgb = work_img[y:y+eff_patch_size, x:x+eff_patch_size]
            feat = extract_patch_features(patch_rgb)
            features_list.append(feat)
            patches_coords.append((x + eff_patch_size // 2, y + eff_patch_size // 2))

    if len(features_list) < 20:
        return {
            "visualization_image": image_rgb.copy(),
            "copy_move_mask": np.zeros((h_orig, w_orig), dtype=np.uint8),
            "match_count": 0,
            "cluster_count": 0,
            "suspicion_score": 0.0,
            "anomalies": []
        }

    feats_matrix = np.array(features_list, dtype=np.float32)
    coords_array = np.array(patches_coords, dtype=np.float32)

    # 2. Nearest Neighbors search in feature space
    nbrs = NearestNeighbors(n_neighbors=min(6, len(feats_matrix)), algorithm='ball_tree')
    nbrs.fit(feats_matrix)
    distances, indices = nbrs.kneighbors(feats_matrix)

    candidate_pairs: List[Tuple[int, int, Tuple[int, int]]] = []

    for i in range(len(feats_matrix)):
        c1 = coords_array[i]
        for k in range(1, indices.shape[1]):
            j = indices[i, k]
            if j <= i:
                continue
            
            c2 = coords_array[j]
            spatial_dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
            feat_dist = distances[i, k]

            if spatial_dist >= eff_min_dist and feat_dist <= eff_sim_thresh:
                # Displacement shift vector (dx, dy)
                dx = int(round((c2[0] - c1[0]) / 8.0)) * 8
                dy = int(round((c2[1] - c1[1]) / 8.0)) * 8
                candidate_pairs.append((i, j, (dx, dy)))

    # 3. Spatial Displacement Shift-Vector Consistency Clustering
    shift_bins: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i, j, shift in candidate_pairs:
        # Standardize shift vector direction
        norm_shift = shift if shift >= (0, 0) else (-shift[0], -shift[1])
        shift_bins.setdefault(norm_shift, []).append((i, j))

    confirmed_pairs: List[Tuple[int, int]] = []
    cluster_count = 0

    for shift, pairs in shift_bins.items():
        if len(pairs) >= min_cluster_size:
            cluster_count += 1
            confirmed_pairs.extend(pairs)

    # 4. Generate visual representation and masks mapped back to original dimensions
    vis_img = image_rgb.copy()
    mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)

    palette = [
        (255, 50, 50), (50, 255, 50), (50, 100, 255),
        (255, 200, 0), (255, 0, 255), (0, 255, 255)
    ]

    for idx, (i, j) in enumerate(confirmed_pairs):
        c1 = coords_array[i]
        c2 = coords_array[j]

        # Rescale coordinates to original image size
        pt1 = (int(c1[0] / scale), int(c1[1] / scale))
        pt2 = (int(c2[0] / scale), int(c2[1] / scale))

        color = palette[idx % len(palette)]
        half_box = int(round((patch_size * 0.75)))

        # Draw source and destination boxes
        cv2.rectangle(
            vis_img,
            (pt1[0] - half_box, pt1[1] - half_box),
            (pt1[0] + half_box, pt1[1] + half_box),
            color, 2
        )
        cv2.rectangle(
            vis_img,
            (pt2[0] - half_box, pt2[1] - half_box),
            (pt2[0] + half_box, pt2[1] + half_box),
            color, 2
        )

        # Draw connecting vector line
        cv2.line(vis_img, pt1, pt2, color, 1, lineType=cv2.LINE_AA)

        # Mark in binary mask
        cv2.rectangle(
            mask_orig,
            (pt1[0] - half_box, pt1[1] - half_box),
            (pt1[0] + half_box, pt1[1] + half_box),
            255, -1
        )
        cv2.rectangle(
            mask_orig,
            (pt2[0] - half_box, pt2[1] - half_box),
            (pt2[0] + half_box, pt2[1] + half_box),
            255, -1
        )

    # Clean mask
    mask_orig = cv2.dilate(mask_orig, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    match_count = len(confirmed_pairs)
    suspicion_score = min(100.0, (cluster_count * 35.0 + match_count * 3.0) * sensitivity)

    anomalies: List[str] = []
    if cluster_count > 0:
        anomalies.append(
            f"Detected {cluster_count} coherent duplicate cluster(s) with {match_count} paired patch matches. "
            "High confidence copy-move forgery detected (identical texture duplicated with consistent displacement)."
        )

    return {
        "visualization_image": vis_img,
        "copy_move_mask": mask_orig,
        "match_count": match_count,
        "cluster_count": cluster_count,
        "suspicion_score": round(suspicion_score, 2),
        "anomalies": anomalies
    }
