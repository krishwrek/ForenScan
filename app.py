"""
ForenScan: Explainable Image Forensics Web Application
A multi-modal digital image forensics suite powered by classical Image Processing
Analysis (IPA) and interpretable machine learning. Detects AI-generated content,
splicing/inpainting, and copy-move forgery with verifiable, physical explanations.
"""

import os
import io
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ForenScan Utilities
from utils.preprocessing import load_and_preprocess_image, image_to_bytes
from utils.metadata import extract_metadata
from utils.ela import compute_ela
from utils.noise_analysis import analyze_noise_consistency
from utils.edge_analysis import analyze_edges
from utils.frequency_analysis import analyze_frequency_domain
from utils.color_analysis import analyze_color_and_lighting
from utils.copy_move_detection import detect_copy_move
from utils.heatmap import generate_fused_heatmap
from utils.feature_extraction import extract_feature_vector, features_to_dataframe
from utils.classifier import predict_authenticity, load_classifier
from utils.report_generator import generate_pdf_report, generate_csv_report


# ==============================================================================
# Page Configuration & Modern SaaS Forensics Styling
# ==============================================================================
st.set_page_config(
    page_title="ForenScan - Explainable Image Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid #334155;
    }
    .hero-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.4);
        display: inline-block;
        margin-bottom: 0.8rem;
        letter-spacing: 0.05em;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        background: linear-gradient(to right, #ffffff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        text-align: left;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }
    .metric-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .metric-bar-bg {
        width: 100%;
        height: 6px;
        background: #f1f5f9;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.6rem;
    }
    .metric-bar-fill {
        height: 100%;
        border-radius: 4px;
    }

    /* Verdict Banners */
    .verdict-box {
        padding: 1.3rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-start;
        gap: 1.2rem;
    }
    .verdict-auth {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
    }
    .verdict-ai {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 6px solid #2563eb;
    }
    .verdict-tamper {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
    }

    /* Evidence Badges */
    .badge-critical {
        background: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-warning {
        background: #ffedd5;
        color: #9a3412;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-normal {
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-info {
        background: #f1f5f9;
        color: #334155;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Interactive Sample Cards */
    .sample-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper Functions for Visualizations
# ==============================================================================
def plot_radar_chart(features: Dict[str, float]) -> plt.Figure:
    """Generate normalized 6-axis forensic radar comparison against authentic baseline."""
    categories = [
        "ELA Inconsistency",
        "Sensor Noise Variance",
        "Edge Sharpness Discrepancy",
        "Fourier Lattice Peaks",
        "Lighting Divergence",
        "Copy-Move Duplicates"
    ]
    
    vals = [
        min(100.0, features.get("ela_inconsistency_score", 0)),
        min(100.0, features.get("noise_inconsistency_score", 0)),
        min(100.0, (features.get("edge_sharpness_discrepancy", 1.0) / 4.0) * 100.0),
        min(100.0, features.get("freq_periodic_peaks", 0) * 18.0),
        min(100.0, (features.get("color_lighting_divergence", 0) / 180.0) * 100.0),
        min(100.0, features.get("copy_move_match_count", 0) * 4.0)
    ]
    
    baseline = [12.0, 15.0, 25.0, 0.0, 15.0, 0.0]

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    vals += vals[:1]
    baseline += baseline[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    ax.plot(angles, baseline, color='#16a34a', linewidth=1.5, linestyle='--', label='Authentic Reference')
    ax.fill(angles, baseline, color='#16a34a', alpha=0.1)

    ax.plot(angles, vals, color='#dc2626', linewidth=2.0, label='Current Target Image')
    ax.fill(angles, vals, color='#dc2626', alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8, weight='bold', color='#1e293b')
    ax.set_ylim(0, 100)
    ax.set_rgrids([25, 50, 75, 100], labels=["25", "50", "75", "100"], color='#94a3b8', size=7)
    ax.grid(color='#cbd5e1', linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=8)
    plt.tight_layout()
    return fig


def plot_frequency_radial(radii: list, profile: list, alpha_slope: float) -> plt.Figure:
    """Plot azimuthal radial power spectrum curve and power-law decay slope."""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    ax.plot(radii, profile, color='#2563eb', linewidth=2, label='Azimuthal Power log(1 + P)')
    ax.set_xlabel('Spatial Frequency Radius (r)', fontsize=8.5, fontweight='bold', color='#334155')
    ax.set_ylabel('Log Power', fontsize=8.5, fontweight='bold', color='#334155')
    ax.set_title(f'Radial Fourier Power Spectrum (Decay Exponent alpha = {alpha_slope:.2f})', fontsize=9.5, fontweight='bold', color='#0f172a')
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_histograms(rgb_hists: dict, hsv_hists: dict) -> plt.Figure:
    """Plot RGB and HSV channel distribution density curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.8))
    fig.patch.set_facecolor('#ffffff')

    ax1.set_facecolor('#f8fafc')
    ax1.plot(rgb_hists['r'], color='#ef4444', label='Red', linewidth=1.5)
    ax1.plot(rgb_hists['g'], color='#22c55e', label='Green', linewidth=1.5)
    ax1.plot(rgb_hists['b'], color='#3b82f6', label='Blue', linewidth=1.5)
    ax1.set_title('RGB Density Distribution', fontsize=9.5, fontweight='bold', color='#0f172a')
    ax1.set_xlabel('Intensity Level (0 - 255)', fontsize=8)
    ax1.set_ylabel('Normalized Frequency', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(fontsize=8)

    ax2.set_facecolor('#f8fafc')
    ax2.plot(hsv_hists['h'], color='#8b5cf6', label='Hue (0-180)', linewidth=1.5)
    ax2.plot(hsv_hists['s'], color='#f59e0b', label='Saturation', linewidth=1.5)
    ax2.plot(hsv_hists['v'], color='#64748b', label='Value (Luminance)', linewidth=1.5)
    ax2.set_title('HSV Color & Lighting Distribution', fontsize=9.5, fontweight='bold', color='#0f172a')
    ax2.set_xlabel('Bin Index', fontsize=8)
    ax2.set_ylabel('Normalized Frequency', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ==============================================================================
# Sidebar Configuration & Controls
# ==============================================================================
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
sample_cases = {
    "Sample 1: Authentic Photo (Canon 5D Mark IV)": os.path.join(sample_dir, "sample1_authentic.jpg"),
    "Sample 2: Spliced & Inpainted (Foreign Sphere)": os.path.join(sample_dir, "sample2_spliced_tampered.jpg"),
    "Sample 3: Copy-Move Forgery (Cloned Bush)": os.path.join(sample_dir, "sample3_copy_move.jpg"),
    "Sample 4: AI Synthetic Portrait (Transposed Conv)": os.path.join(sample_dir, "sample4_ai_generated.png")
}

with st.sidebar:
    st.markdown("### 🛡️ **ForenScan Suite**")
    st.caption("Explainable Multi-Modal Image Forensics")
    st.divider()

    st.markdown("#### 📂 **Input Target**")
    input_mode = st.radio(
        "Choose Target Source:",
        ["Upload Your Own File", "Pre-Packaged Forensic Cases"],
        index=0
    )

    uploaded_file = None
    selected_sample = None

    if input_mode == "Upload Your Own File":
        uploaded_file = st.file_uploader(
            "Upload Image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Images are preprocessed with aspect-ratio preserving bounding up to 1024px."
        )
    else:
        sample_choice = st.selectbox(
            "Select Verified Forensic Benchmark:",
            list(sample_cases.keys())
        )
        selected_sample = sample_cases[sample_choice]

    st.divider()
    st.markdown("#### ⚙️ **Analysis Parameters**")
    
    sensitivity = st.slider(
        "Anomaly Sensitivity Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Modulates outlier detection thresholds across ELA, noise, and edge sharpness."
    )

    ela_quality = st.slider(
        "ELA Recompression Quality",
        min_value=85,
        max_value=98,
        value=95,
        step=1,
        help="Standard JPEG compression quality for differential Error Level Analysis."
    )

    block_size = st.select_slider(
        "Analysis Block Dimension (px)",
        options=[16, 32, 64],
        value=32,
        help="Grid window size for regional noise variance and edge sharpness analysis."
    )

    st.divider()
    st.markdown("#### 🧠 **Classification Engine**")
    classifier_mode = st.radio(
        "Decision Engine:",
        ["Calibrated Random Forest (IPA Features)", "Rule-Based Classical IPA Only"],
        index=0
    )
    use_ml = (classifier_mode == "Calibrated Random Forest (IPA Features)")

    with st.expander("🛠️ Active Forensic Checkpoints", expanded=False):
        enable_ela = st.checkbox("Error Level Analysis (ELA)", value=True)
        enable_noise = st.checkbox("Sensor Noise Consistency", value=True)
        enable_edge = st.checkbox("Multi-Operator Edge Sharpness", value=True)
        enable_freq = st.checkbox("2D FFT & DCT Frequency Analysis", value=True)
        enable_color = st.checkbox("Color & Illumination Gradients", value=True)
        enable_copymove = st.checkbox("Copy-Move Patch Clustering", value=True)


# ==============================================================================
# Header Section
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-tag">FORENSIC GRADE IMAGE INTEGRITY AUDIT // IPA ENGINE v2.0</div>
    <div class="hero-title">ForenScan: Explainable Image Forensics</div>
    <div class="hero-subtitle">
        Deep verification of digital image integrity using classical Image Processing Analysis (IPA).
        Detects AI generative synthesis (GANs, Diffusion), localized object splicing/inpainting, and copy-move forgery
        with transparent, verifiable physical evidence.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Input Image Acquisition & Quick-Start Gallery
# ==============================================================================
image_bytes = None
target_filename = "analyzed_image.png"

# Quick launcher buttons state management
if "quick_sample" not in st.session_state:
    st.session_state["quick_sample"] = None

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    target_filename = uploaded_file.name
    st.session_state["quick_sample"] = None
elif selected_sample is not None and input_mode == "Pre-Packaged Forensic Cases":
    if os.path.exists(selected_sample):
        with open(selected_sample, "rb") as f:
            image_bytes = f.read()
        target_filename = os.path.basename(selected_sample)
elif st.session_state["quick_sample"] is not None:
    path = sample_cases.get(st.session_state["quick_sample"])
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            image_bytes = f.read()
        target_filename = os.path.basename(path)

if image_bytes is None:
    st.markdown("### ⚡ **One-Click Demonstration Cases**")
    st.caption("Click any forensic scenario below to test the complete detection pipeline instantly:")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.markdown("""
        <div class="sample-card">
            <div style="font-size:1.8rem; margin-bottom:0.3rem;">📸</div>
            <div style="font-weight:700; color:#0f172a; margin-bottom:0.2rem;">Authentic Photo</div>
            <div style="font-size:0.78rem; color:#64748b; line-height:1.4; margin-bottom:0.8rem;">
                Natural landscape with physical camera sensor shot noise & Canon 5D Mark IV EXIF.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Authentic Test", use_container_width=True, key="btn_s1"):
            st.session_state["quick_sample"] = "Sample 1: Authentic Photo (Canon 5D Mark IV)"
            st.rerun()

    with col_s2:
        st.markdown("""
        <div class="sample-card">
            <div style="font-size:1.8rem; margin-bottom:0.3rem;">✂️</div>
            <div style="font-weight:700; color:#0f172a; margin-bottom:0.2rem;">Spliced & Inpainted</div>
            <div style="font-size:0.78rem; color:#64748b; line-height:1.4; margin-bottom:0.8rem;">
                Foreign object inserted with Photoshop footprint, sharp scissor edge & high ELA error.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Splicing Test", use_container_width=True, key="btn_s2"):
            st.session_state["quick_sample"] = "Sample 2: Spliced & Inpainted (Foreign Sphere)"
            st.rerun()

    with col_s3:
        st.markdown("""
        <div class="sample-card">
            <div style="font-size:1.8rem; margin-bottom:0.3rem;">👯</div>
            <div style="font-weight:700; color:#0f172a; margin-bottom:0.2rem;">Copy-Move Forgery</div>
            <div style="font-size:0.78rem; color:#64748b; line-height:1.4; margin-bottom:0.8rem;">
                Cloned texture patches duplicated across the frame with identical displacement vectors.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Copy-Move Test", use_container_width=True, key="btn_s3"):
            st.session_state["quick_sample"] = "Sample 3: Copy-Move Forgery (Cloned Bush)"
            st.rerun()

    with col_s4:
        st.markdown("""
        <div class="sample-card">
            <div style="font-size:1.8rem; margin-bottom:0.3rem;">🤖</div>
            <div style="font-weight:700; color:#0f172a; margin-bottom:0.2rem;">AI Synthetic Portrait</div>
            <div style="font-size:0.78rem; color:#64748b; line-height:1.4; margin-bottom:0.8rem;">
                Generated portrait exhibiting transposed-convolution FFT spikes & smooth noise floor.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run AI Detection Test", use_container_width=True, key="btn_s4"):
            st.session_state["quick_sample"] = "Sample 4: AI Synthetic Portrait (Transposed Conv)"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Or upload your own image** using the file uploader in the sidebar to perform a full forensic scan.")
    st.stop()
    import sys
    sys.exit(0)


# ==============================================================================
# Pipeline Execution
# ==============================================================================
with st.spinner("Analyzing image across 6 classical forensic modalities..."):
    t_start = time.time()

    # 1. Preprocessing
    orig_rgb, resized_rgb, gray_img, scale_factor, (orig_w, orig_h) = load_and_preprocess_image(
        image_bytes, max_dim=1024
    )

    # 2. Metadata Extraction
    metadata_res = extract_metadata(image_bytes)

    # 3. ELA
    if enable_ela:
        ela_res = compute_ela(resized_rgb, quality=ela_quality, scale=20.0, anomaly_threshold_std=2.0)
    else:
        ela_res = {"mean_error": 0.0, "std_error": 0.0, "max_error": 0.0, "high_error_ratio": 0.0,
                   "inconsistency_score": 0.0, "ela_image": resized_rgb, "ela_gray": gray_img,
                   "ela_heatmap": resized_rgb, "anomaly_mask": np.zeros_like(gray_img)}

    # 4. Noise Consistency
    if enable_noise:
        noise_res = analyze_noise_consistency(gray_img, block_size=block_size, sensitivity=sensitivity)
    else:
        noise_res = {"global_noise_mean": 0.0, "global_noise_std": 0.0, "outlier_block_ratio": 0.0,
                     "noise_inconsistency_score": 0.0, "noise_map": np.zeros_like(gray_img),
                     "noise_heatmap": resized_rgb, "anomaly_mask": np.zeros_like(gray_img), "anomalies": []}

    # 5. Edge Analysis
    if enable_edge:
        edge_res = analyze_edges(gray_img, block_size=block_size, sensitivity=sensitivity)
    else:
        edge_res = {"mean_edge_density": 0.0, "mean_sharpness": 0.0, "max_sharpness": 0.0,
                    "sharpness_discrepancy": 1.0, "canny_edges": np.zeros_like(gray_img),
                    "sobel_magnitude": np.zeros_like(gray_img), "laplacian": np.zeros_like(gray_img),
                    "sharpness_heatmap": resized_rgb, "anomaly_mask": np.zeros_like(gray_img), "anomalies": []}

    # 6. Frequency Domain Analysis
    if enable_freq:
        freq_res = analyze_frequency_domain(gray_img, sensitivity=sensitivity)
    else:
        freq_res = {"fft_spectrum": resized_rgb, "dct_spectrum": resized_rgb, "radial_radii": [],
                    "radial_profile": [], "spectral_slope": 2.0, "periodic_peak_count": 0,
                    "high_freq_energy_ratio": 0.0, "dct_ac_energy": 0.0, "anomalies": []}

    # 7. Color & Lighting Consistency
    if enable_color:
        color_res = analyze_color_and_lighting(resized_rgb, sensitivity=sensitivity)
    else:
        color_res = {"rgb_histograms": {'r': [], 'g': [], 'b': []}, "hsv_histograms": {'h': [], 's': [], 'v': []},
                     "quadrant_stats": [], "lighting_divergence_angle": 0.0, "color_inconsistency_score": 0.0,
                     "lighting_inconsistency_score": 0.0, "anomalies": []}

    # 8. Copy-Move Forgery Detection
    if enable_copymove:
        copy_move_res = detect_copy_move(resized_rgb, patch_size=16, stride=8, sensitivity=sensitivity)
    else:
        copy_move_res = {"visualization_image": resized_rgb, "copy_move_mask": np.zeros_like(gray_img),
                         "match_count": 0, "cluster_count": 0, "suspicion_score": 0.0, "anomalies": []}

    # 9. Fused Heatmap
    fused_res = generate_fused_heatmap(
        image_rgb=resized_rgb,
        ela_array=ela_res.get("ela_gray"),
        noise_anomaly=noise_res.get("anomaly_mask"),
        edge_anomaly=edge_res.get("anomaly_mask"),
        copy_move_mask=copy_move_res.get("copy_move_mask"),
        alpha=0.55,
        threshold=0.35
    )

    # 10. Feature Aggregation
    feat_vec, feat_dict = extract_feature_vector(
        ela_res, noise_res, edge_res, freq_res, color_res, copy_move_res, metadata_res, fused_res
    )

    # 11. Classifier Inference & Explainable Reasoning
    verdict_info = predict_authenticity(
        features_dict=feat_dict,
        metadata_dict=metadata_res,
        use_ml=use_ml
    )

    elapsed_time = round(time.time() - t_start, 2)


# ==============================================================================
# Top Verdict Display & Metric Gauges
# ==============================================================================
verdict = verdict_info["verdict"]
auth_score = verdict_info["authenticity_score"]
tamper_score = verdict_info["tamper_score"]
ai_score = verdict_info["ai_score"]
conf = verdict_info["confidence"]

case_hash = hashlib.sha256(image_bytes).hexdigest()[:12].upper()

if "Authentic" in verdict:
    st.markdown(f"""
    <div class="verdict-box verdict-auth">
        <div style="font-size:2.2rem;">🛡️</div>
        <div>
            <div style="font-size:0.75rem; font-weight:700; color:#15803d; letter-spacing:0.05em; text-transform:uppercase;">
                OFFICIAL DETERMINATION // CASE #{case_hash}
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:#14532d; margin:0.1rem 0 0.3rem 0;">
                {verdict.upper()}
            </div>
            <div style="font-size:0.92rem; color:#166534; line-height:1.4;">
                Uniform sensor shot noise, authentic EXIF camera metadata, consistent ELA compression history, and intact Fourier power-law decay.
                Analysis finalized in {elapsed_time}s via <b>{verdict_info['method']}</b>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif "AI" in verdict:
    st.markdown(f"""
    <div class="verdict-box verdict-ai">
        <div style="font-size:2.2rem;">🤖</div>
        <div>
            <div style="font-size:0.75rem; font-weight:700; color:#1d4ed8; letter-spacing:0.05em; text-transform:uppercase;">
                OFFICIAL DETERMINATION // CASE #{case_hash}
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:#1e3a8a; margin:0.1rem 0 0.3rem 0;">
                {verdict.upper()}
            </div>
            <div style="font-size:0.92rem; color:#1e40af; line-height:1.4;">
                Periodic Fourier lattice spikes identified (fingerprint of transposed convolution / latent upsamplers), unnatural high-frequency decay, and absence of natural sensor noise.
                Analysis finalized in {elapsed_time}s via <b>{verdict_info['method']}</b>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="verdict-box verdict-tamper">
        <div style="font-size:2.2rem;">⚠️</div>
        <div>
            <div style="font-size:0.75rem; font-weight:700; color:#b91c1c; letter-spacing:0.05em; text-transform:uppercase;">
                OFFICIAL DETERMINATION // CASE #{case_hash}
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:#7f1d1d; margin:0.1rem 0 0.3rem 0;">
                {verdict.upper()}
            </div>
            <div style="font-size:0.92rem; color:#991b1b; line-height:1.4;">
                Discordant compression cycles detected across regions, unnatural boundary sharpness discrepancies, discordant sensor noise blocks, or duplicated copy-move patches.
                Analysis finalized in {elapsed_time}s via <b>{verdict_info['method']}</b>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Metric Gauge Cards
g1, g2, g3, g4 = st.columns(4)

with g1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-header">
            <span class="metric-label">Authenticity Score</span>
            <span style="font-size:1.1rem;">🛡️</span>
        </div>
        <div class="metric-val" style="color:#16a34a;">{auth_score}%</div>
        <div style="font-size:0.75rem; color:#64748b;">Photographic Integrity</div>
        <div class="metric-bar-bg">
            <div class="metric-bar-fill" style="width:{auth_score}%; background:#16a34a;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with g2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-header">
            <span class="metric-label">Tamper Probability</span>
            <span style="font-size:1.1rem;">✂️</span>
        </div>
        <div class="metric-val" style="color:#dc2626;">{tamper_score}%</div>
        <div style="font-size:0.75rem; color:#64748b;">Splicing / Inpainting Risk</div>
        <div class="metric-bar-bg">
            <div class="metric-bar-fill" style="width:{tamper_score}%; background:#dc2626;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with g3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-header">
            <span class="metric-label">AI Generation Risk</span>
            <span style="font-size:1.1rem;">🤖</span>
        </div>
        <div class="metric-val" style="color:#2563eb;">{ai_score}%</div>
        <div style="font-size:0.75rem; color:#64748b;">Diffusion / GAN Fingerprint</div>
        <div class="metric-bar-bg">
            <div class="metric-bar-fill" style="width:{ai_score}%; background:#2563eb;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with g4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-header">
            <span class="metric-label">Decision Confidence</span>
            <span style="font-size:1.1rem;">🎯</span>
        </div>
        <div class="metric-val" style="color:#4f46e5;">{conf}%</div>
        <div style="font-size:0.75rem; color:#64748b;">Statistical Certainty</div>
        <div class="metric-bar-bg">
            <div class="metric-bar-fill" style="width:{conf}%; background:#4f46e5;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# Comprehensive Forensic Tabs
# ==============================================================================
tab_overview, tab_ela_noise, tab_edge_freq, tab_color, tab_copymove, tab_reasons, tab_metadata, tab_features, tab_report = st.tabs([
    "🔍 Visual Overview",
    "🔬 Compression & Noise",
    "📐 Boundary & Frequency",
    "🎨 Color & Lighting",
    "🔎 Copy-Move Visualizer",
    "📋 Explainable Findings",
    "🏷️ Metadata Audit",
    "📊 Feature Radar",
    "📑 Export Dossier"
])

# ------------------------------------------------------------------------------
# TAB 1: Visual Overview & Triage
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown("#### Primary Visual Forensic Triage")
    st.caption("Direct visual comparison of input target, Error Level Analysis (ELA), and combined multi-modal suspicious-region overlay.")

    ov_c1, ov_c2, ov_c3 = st.columns(3)
    with ov_c1:
        st.image(resized_rgb, caption=f"Original Target ({orig_w}x{orig_h}px)", use_container_width=True)
    with ov_c2:
        st.image(ela_res["ela_heatmap"], caption=f"ELA Heatmap (JPEG Quality {ela_quality})", use_container_width=True)
    with ov_c3:
        st.image(fused_res["annotated_image"], caption="Fused Suspicious Regions (Peak Confidence Overlay)", use_container_width=True)

    if fused_res["bounding_boxes"]:
        st.markdown(f"**Identified Suspicious Regions ({len(fused_res['bounding_boxes'])} isolated):**")
        bb_df = pd.DataFrame(fused_res["bounding_boxes"])
        st.dataframe(bb_df, use_container_width=True)
    else:
        st.info("No concentrated high-suspicion clusters isolated at current sensitivity.")

# ------------------------------------------------------------------------------
# TAB 2: Compression & Sensor Noise
# ------------------------------------------------------------------------------
with tab_ela_noise:
    st.markdown("#### Error Level Analysis (ELA) & Sensor Noise Consistency")
    
    col_en1, col_en2 = st.columns(2)
    with col_en1:
        st.markdown("##### Error Level Analysis (ELA)")
        st.image(ela_res["ela_heatmap"], caption="Amplified JPEG Recompression Error", use_container_width=True)
        st.markdown(f"""
        - **Mean Recompression Error:** `{ela_res['mean_error']}`
        - **Error Standard Deviation:** `{ela_res['std_error']}`
        - **High-Error Pixel Coverage:** `{ela_res['high_error_ratio']}%`
        - **Local Error Inconsistency Score:** `{ela_res['inconsistency_score']}/100`
        """)
        st.caption("When an image is saved in JPEG format, regions modified after the initial save exhibit distinct error levels upon recompression.")

    with col_en2:
        st.markdown("##### Physical Sensor Noise Residual")
        st.image(noise_res["noise_heatmap"], caption=f"Local Noise Variance Map ({block_size}x{block_size}px Blocks)", use_container_width=True)
        st.markdown(f"""
        - **Global Noise Floor (MAD):** `{noise_res['global_noise_mean']}`
        - **Block Noise Deviation:** `{noise_res['global_noise_std']}`
        - **Statistically Discordant Blocks:** `{noise_res['outlier_block_ratio']}%`
        - **Noise Inconsistency Score:** `{noise_res['noise_inconsistency_score']}/100`
        """)
        st.caption("Authentic camera sensors produce uniform Poisson/Gaussian shot noise. Mismatched variance indicates foreign object splicing or synthetic smoothing.")

# ------------------------------------------------------------------------------
# TAB 3: Boundary & Frequency Transforms
# ------------------------------------------------------------------------------
with tab_edge_freq:
    st.markdown("#### Boundary Sharpness & 2D Frequency Transforms")

    ed_c1, ed_c2, ed_c3 = st.columns(3)
    with ed_c1:
        st.markdown("##### Canny Edge Detection")
        st.image(edge_res["canny_edges"], caption=f"Mean Edge Density: {edge_res['mean_edge_density']}%", use_container_width=True)
    with ed_c2:
        st.markdown("##### Sobel Gradient Magnitude")
        st.image(edge_res["sobel_magnitude"], caption=f"Mean Gradient Sharpness: {edge_res['mean_sharpness']}", use_container_width=True)
    with ed_c3:
        st.markdown("##### Regional Sharpness Heatmap")
        st.image(edge_res["sharpness_heatmap"], caption=f"Sharpness Discrepancy Ratio: {edge_res['sharpness_discrepancy']}x", use_container_width=True)

    st.divider()

    fq_c1, fq_c2, fq_c3 = st.columns([1, 1, 1.2])
    with fq_c1:
        st.markdown("##### 2D FFT Magnitude Spectrum")
        st.image(freq_res["fft_spectrum"], caption=f"Periodic High-Freq Spikes: {freq_res['periodic_peak_count']}", use_container_width=True)
        st.caption("Bright lattice dots away from the DC center reveal transposed convolution upsampling grids common in GAN & Diffusion architectures.")
    with fq_c2:
        st.markdown("##### 8x8 Block DCT Grid")
        st.image(freq_res["dct_spectrum"], caption=f"DCT AC Energy: {freq_res['dct_ac_energy']}", use_container_width=True)
        st.caption("Natural photographs recompressed with JPEG adhere to standard 8x8 block DCT structures; synthetic elements disrupt this grid.")
    with fq_c3:
        st.markdown("##### Radial Power Profile")
        if freq_res["radial_radii"] and freq_res["radial_profile"]:
            fig_rad = plot_frequency_radial(
                freq_res["radial_radii"], freq_res["radial_profile"], freq_res["spectral_slope"]
            )
            st.pyplot(fig_rad)
        else:
            st.caption("Radial profile unavailable.")

# ------------------------------------------------------------------------------
# TAB 4: Color & Lighting Consistency
# ------------------------------------------------------------------------------
with tab_color:
    st.markdown("#### Color Distribution & Illumination Gradient Consistency")

    if color_res["rgb_histograms"]['r']:
        fig_hist = plot_histograms(color_res["rgb_histograms"], color_res["hsv_histograms"])
        st.pyplot(fig_hist)

    st.markdown("##### Quadrant Illumination & Color Temperature Audit")
    q_cols = st.columns(4)
    for idx, q in enumerate(color_res.get("quadrant_stats", [])):
        with q_cols[idx % 4]:
            st.markdown(f"**{q['region']}**")
            st.markdown(f"""
            - **Mean RGB:** `{q['mean_rgb']}`
            - **Color Temp:** `{q['color_temperature']}`
            - **Lighting Vector:** `{q['lighting_angle_deg']}°`
            """)

    st.info(f"**Max Illumination Divergence:** `{color_res.get('lighting_divergence_angle', 0)}°` across quadrants (Score: `{color_res.get('lighting_inconsistency_score', 0)}/100`). Divergence above 80° indicates compositing from conflicting illumination environments.")

# ------------------------------------------------------------------------------
# TAB 5: Copy-Move Forgery Detection
# ------------------------------------------------------------------------------
with tab_copymove:
    st.markdown("#### Copy-Move & Texture Cloning Visualizer")
    st.caption("Identifies non-adjacent, highly similar texture patches sharing identical spatial displacement vectors.")

    cm_col1, cm_col2 = st.columns([1.5, 1])
    with cm_col1:
        st.image(copy_move_res["visualization_image"], caption="Matched Duplicated Patches & Displacement Vectors", use_container_width=True)
    with cm_col2:
        st.markdown(f"""
        ##### Cloning Statistics
        - **Confirmed Patch Matches:** `{copy_move_res['match_count']}`
        - **Coherent Duplicate Clusters:** `{copy_move_res['cluster_count']}`
        - **Copy-Move Suspicion Score:** `{copy_move_res['suspicion_score']}/100`
        """)
        if copy_move_res['cluster_count'] > 0:
            st.error("⚠️ **Cloned content detected:** High confidence copy-move forgery confirmed. Colored bounding boxes and connecting vectors mark duplicated patches.")
        else:
            st.success("✅ **No copy-move cloning detected:** No significant non-adjacent duplicate patch clusters observed.")

# ------------------------------------------------------------------------------
# TAB 6: Explainable Findings & Reasons
# ------------------------------------------------------------------------------
with tab_reasons:
    st.markdown("#### Transparent Forensic Evidence Findings")
    st.caption("Verifiable, physical explanations connecting mathematical image processing anomalies directly to the verdict.")

    reasons_list = verdict_info.get("reasons", [])
    if reasons_list:
        for r in reasons_list:
            sev = r.get("severity", "Info")
            cat = r.get("type", "General")
            txt = r.get("text", "")

            if sev == "Critical":
                badge_html = '<span class="badge-critical">CRITICAL</span>'
            elif sev == "Warning":
                badge_html = '<span class="badge-warning">WARNING</span>'
            elif sev == "Normal":
                badge_html = '<span class="badge-normal">NORMAL</span>'
            else:
                badge_html = '<span class="badge-info">INFO</span>'

            st.markdown(f"""
            <div style="padding: 0.9rem 1.1rem; border-radius: 10px; background: white; border: 1px solid #e2e8f0; margin-bottom: 0.6rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                {badge_html} &nbsp; <b>{cat}:</b> {txt}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No anomalies identified.")

# ------------------------------------------------------------------------------
# TAB 7: Metadata Audit
# ------------------------------------------------------------------------------
with tab_metadata:
    st.markdown("#### Hardware, GPS & Software Metadata Audit")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("##### Camera Hardware")
        cam = metadata_res.get("camera_info", {})
        if cam:
            st.json(cam)
        else:
            st.warning("No camera hardware tags present in metadata.")

        st.markdown("##### File Attributes")
        st.json(metadata_res.get("basic_info", {}))

    with m_col2:
        st.markdown("##### Software Footprints & AI Signatures")
        sw = metadata_res.get("software_traces", [])
        if sw:
            st.error(f"Software signatures detected: **{', '.join(sw)}**")
        else:
            st.success("No editing or AI software fingerprints recorded.")

        st.markdown("##### Geolocation")
        gps = metadata_res.get("gps_info", {})
        if gps:
            st.json(gps)
        else:
            st.caption("No GPS geotag metadata embedded.")

    if metadata_res.get("raw_tags"):
        with st.expander("Inspect All Raw EXIF Tags"):
            st.dataframe(pd.DataFrame(list(metadata_res["raw_tags"].items()), columns=["Tag", "Value"]), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 8: Feature Radar
# ------------------------------------------------------------------------------
with tab_features:
    st.markdown("#### Quantitative IPA Feature Manifest & Radar Comparison")
    st.caption("Standardized 25-dimensional feature vector extracted across all classical modalities.")

    feat_df = features_to_dataframe(feat_dict)
    
    fm_c1, fm_c2 = st.columns([1.2, 1])
    with fm_c1:
        st.dataframe(feat_df, use_container_width=True, height=420)
    with fm_c2:
        st.markdown("##### Forensic Radar Comparison")
        fig_radar = plot_radar_chart(feat_dict)
        st.pyplot(fig_radar)

# ------------------------------------------------------------------------------
# TAB 9: Export Dossier
# ------------------------------------------------------------------------------
with tab_report:
    st.markdown("#### Export Forensic Dossier")
    st.caption("Generate publication-ready PDF examination reports and CSV feature manifests.")

    orig_bytes = image_to_bytes(resized_rgb, format="JPEG")
    ela_bytes = image_to_bytes(ela_res["ela_heatmap"], format="JPEG")
    overlay_bytes = image_to_bytes(fused_res["annotated_image"], format="JPEG")

    pdf_bytes = generate_pdf_report(
        orig_img_bytes=orig_bytes,
        ela_img_bytes=ela_bytes,
        overlay_img_bytes=overlay_bytes,
        verdict_info=verdict_info,
        metadata_dict=metadata_res,
        features_dict=feat_dict,
        filename=target_filename
    )

    csv_str = generate_csv_report(
        features_dict=feat_dict,
        metadata_dict=metadata_res,
        verdict_info=verdict_info,
        filename=target_filename
    )

    r_c1, r_c2 = st.columns(2)
    with r_c1:
        st.download_button(
            label="📄 Download Official Forensic PDF Report",
            data=pdf_bytes,
            file_name=f"forenscan_report_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.caption("Includes case summary, visual exhibits, explainable evidence table, and hardware audit.")

    with r_c2:
        st.download_button(
            label="📊 Download Quantitative IPA CSV Manifest",
            data=csv_str.encode('utf-8'),
            file_name=f"forenscan_features_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Complete CSV data logging of all 25 numerical IPA metrics and metadata parameters.")
