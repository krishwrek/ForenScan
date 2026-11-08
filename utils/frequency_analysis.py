"""
Frequency Domain Analysis module for ForenScan.
Applies 2D Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT).
Detects periodic lattice spikes (checkerboard upsampling artifacts in GANs/Diffusion),
spectral decay slope deviations from natural 1/f^alpha photographic power laws,
and 8x8 DCT grid disruptions.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import cv2


def azimuthal_average(image: np.ndarray, center: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the azimuthally averaged radial profile of an image.
    """
    h, w = image.shape
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = center

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(np.int32)

    # Bin values by integer radius
    max_radius = min(cx, cy)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    
    radialprofile = tbin[:max_radius] / np.maximum(nr[:max_radius], 1)
    radii = np.arange(max_radius)
    return radii, radialprofile


def analyze_frequency_domain(
    gray_image: np.ndarray,
    sensitivity: float = 1.0
) -> Dict[str, Any]:
    """
    Perform 2D FFT and 2D DCT frequency domain analysis.
    
    Parameters:
        gray_image: np.ndarray uint8 of shape (H, W)
        sensitivity: Sensitivity multiplier for peak detection
        
    Returns:
        dict containing:
            - fft_spectrum: np.ndarray uint8 RGB (log magnitude spectrum display)
            - dct_spectrum: np.ndarray uint8 RGB (average 8x8 block DCT display)
            - radial_radii: list of radius integers
            - radial_profile: list of float power values
            - spectral_slope: float decay exponent
            - periodic_peak_count: int count of unnatural high-frequency spikes
            - high_freq_energy_ratio: float % of energy in high frequency band
            - dct_block_energy: float
            - anomalies: list of human-readable findings
    """
    h, w = gray_image.shape

    # 1. 2D Fast Fourier Transform (FFT)
    # Using float32 for high numerical precision
    f_transform = np.fft.fft2(gray_image.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    power_spectrum = magnitude_spectrum ** 2

    # Log magnitude for human visual interpretation
    log_magnitude = np.log1p(magnitude_spectrum)
    norm_fft = cv2.normalize(log_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_fft_uint8 = np.clip(norm_fft, 0, 255).astype(np.uint8)
    fft_spectrum_bgr = cv2.applyColorMap(norm_fft_uint8, cv2.COLORMAP_VIRIDIS)
    fft_spectrum_rgb = cv2.cvtColor(fft_spectrum_bgr, cv2.COLOR_BGR2RGB)

    # 2. Azimuthal Average (Radial Power Profile)
    radii, radial_vals = azimuthal_average(power_spectrum)

    # Fit power law log(P) = -alpha * log(r) + c (excluding DC radius 0 and 1)
    valid_mask = (radii >= 5) & (radii < len(radii) * 0.8) & (radial_vals > 0)
    if np.sum(valid_mask) > 10:
        log_r = np.log(radii[valid_mask])
        log_p = np.log(radial_vals[valid_mask])
        # Linear regression slope: p = -alpha * r + c
        poly = np.polyfit(log_r, log_p, 1)
        spectral_slope = float(-poly[0])  # Positive alpha represents decay rate
    else:
        spectral_slope = 2.0

    # 3. Detect Periodic High-Frequency Spikes (GAN / Diffusion checkerboard artifacts)
    # Remove low frequency center circle to avoid confusing natural DC luminance with periodic spikes
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Analyze middle-to-high frequency annular zone (20% to 80% of max radius)
    max_r = min(cy, cx)
    annulus_mask = (dist_from_center > (max_r * 0.20)) & (dist_from_center < (max_r * 0.80))

    # Background subtraction: median filtered log magnitude
    # Using a 25x25 median blur to isolate sharp isolated frequency spikes from broad textures
    log_mag_blurred = cv2.medianBlur(norm_fft_uint8, 25).astype(np.float32)
    residual_mag = norm_fft_uint8.astype(np.float32) - log_mag_blurred

    annulus_pixels = residual_mag[annulus_mask]
    annulus_std = float(np.std(annulus_pixels)) + 1e-5
    annulus_mean = float(np.mean(annulus_pixels))

    # High prominence threshold (must be significantly higher than local background)
    spike_thresh = max(18.0, annulus_mean + (5.5 / max(0.2, sensitivity)) * annulus_std)
    spike_candidates = (residual_mag > spike_thresh) & annulus_mask & (norm_fft_uint8 > 60)

    # Filter out single-pixel noise: require components of size >= 2 pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spike_candidates.astype(np.uint8))
    periodic_peak_count = 0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= 2:
            periodic_peak_count += 1

    # 4. High-frequency energy ratio
    high_freq_mask = dist_from_center > (max_r * 0.5)
    total_energy = float(np.sum(power_spectrum)) + 1e-5
    high_freq_energy = float(np.sum(power_spectrum[high_freq_mask]))
    high_freq_ratio = (high_freq_energy / total_energy) * 100.0

    # 5. Discrete Cosine Transform (8x8 blocks)
    # JPEG compresses in 8x8 blocks. Natural photographs recompressed maintain 8x8 grid power.
    # Uncompressed synthetic generations or spliced regions disrupt this grid.
    h_8 = (h // 8) * 8
    w_8 = (w // 8) * 8
    cropped_gray = gray_image[:h_8, :w_8].astype(np.float32)
    
    dct_accum = np.zeros((8, 8), dtype=np.float32)
    block_count = 0
    for by in range(0, h_8, 8):
        for bx in range(0, w_8, 8):
            blk = cropped_gray[by:by+8, bx:bx+8]
            dct_blk = cv2.dct(blk)
            dct_accum += np.abs(dct_blk)
            block_count += 1
            
    if block_count > 0:
        avg_dct = dct_accum / float(block_count)
    else:
        avg_dct = np.zeros((8, 8), dtype=np.float32)

    # Visual representation of average 8x8 DCT
    norm_dct = cv2.normalize(np.log1p(avg_dct), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dct_resized = cv2.resize(norm_dct.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    dct_spectrum_bgr = cv2.applyColorMap(dct_resized, cv2.COLORMAP_MAGMA)
    dct_spectrum_rgb = cv2.cvtColor(dct_spectrum_bgr, cv2.COLOR_BGR2RGB)

    dct_ac_energy = float(np.sum(avg_dct[1:, 1:]))

    # 6. Interpretive anomalies
    anomalies: List[str] = []
    if periodic_peak_count >= 4:
        anomalies.append(
            f"Detected {periodic_peak_count} anomalous periodic frequency spikes in 2D FFT spectrum. "
            "Strongly characteristic of transposed convolution artifacts found in GAN and diffusion generator architectures."
        )
    if spectral_slope < 1.1 or spectral_slope > 3.2:
        anomalies.append(
            f"Atypical spectral decay exponent (alpha = {round(spectral_slope, 2)}). "
            "Natural photographic scenes typically obey 1.5 < alpha < 2.5 power-law scaling."
        )
    if high_freq_ratio < 0.05:
        anomalies.append(
            f"Abnormally low high-frequency spectral energy ({round(high_freq_ratio, 3)}%). "
            "Indicates synthetic smoothing or suppression of natural fine-scale photographic textures."
        )

    # Subsample radial profile for clean plotting (e.g. 50 points)
    step = max(1, len(radii) // 50)
    sampled_radii = [int(r) for r in radii[::step]]
    sampled_radial_vals = [float(np.log1p(v)) for v in radial_vals[::step]]

    return {
        "fft_spectrum": fft_spectrum_rgb,
        "dct_spectrum": dct_spectrum_rgb,
        "radial_radii": sampled_radii,
        "radial_profile": sampled_radial_vals,
        "spectral_slope": round(spectral_slope, 3),
        "periodic_peak_count": int(periodic_peak_count),
        "high_freq_energy_ratio": round(high_freq_ratio, 3),
        "dct_ac_energy": round(dct_ac_energy, 2),
        "anomalies": anomalies
    }
