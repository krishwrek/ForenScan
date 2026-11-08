"""
ForenScan: Explainable Image Forensics Web Application
A multi-modal digital image forensics suite powered by classical Image Processing
Analysis (IPA) and interpretable machine learning. Detects AI-generated content,
splicing/inpainting, and copy-move forgery with physical explanations.
"""

import os
import io
import time
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
# Page Setup & Styling
# ==============================================================================
st.set_page_config(
    page_title="ForenScan - Explainable Image Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern forensic dashboard styling
st.markdown("""
<style>
    /* Global styles */
    .reportview-container {
        background: #0f172a;
    }
    .main-title {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        color: #1e293b;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }
    /* Verdict badge banners */
    .verdict-banner-auth {
        background: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 1.1rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .verdict-banner-ai {
        background: #eff6ff;
        border-left: 6px solid #3b82f6;
        padding: 1.1rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .verdict-banner-tamper {
        background: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 1.1rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    /* Severity badges */
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .badge-warning {
        background-color: #ffedd5;
        color: #9a3412;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .badge-info {
        background-color: #f1f5f9;
        color: #334155;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Sidebar Controls & Configuration
# ==============================================================================
with st.sidebar:
    st.markdown("## 🔍 **ForenScan IPA**")
    st.caption("Explainable Classical Image Forensics Suite")
    st.divider()

    st.markdown("### 📥 **Image Source**")
    source_choice = st.radio(
        "Select Image Input Mode:",
        ["Upload Your Own Image", "Load Demo Sample Case"],
        index=0
    )

    uploaded_file = None
    selected_sample_path = None
    selected_sample_label = "Custom Image"

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
    sample_options = {
        "Sample 1: Authentic Photo (Clean Sensor & EXIF)": os.path.join(sample_dir, "sample1_authentic.jpg"),
        "Sample 2: Spliced & Inpainted (Foreign Sphere inserted)": os.path.join(sample_dir, "sample2_spliced_tampered.jpg"),
        "Sample 3: Copy-Move Forgery (Cloned Bush/Texture)": os.path.join(sample_dir, "sample3_copy_move.jpg"),
        "Sample 4: AI Synthetic Portrait (Transposed Conv Peaks)": os.path.join(sample_dir, "sample4_ai_generated.png"),
    }

    if source_choice == "Upload Your Own Image":
        uploaded_file = st.file_uploader(
            "Upload Target Image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Images are preprocessed up to 1024px maximum dimension."
        )
    else:
        sample_choice = st.selectbox(
            "Choose a Verified Forensic Sample:",
            list(sample_options.keys())
        )
        selected_sample_path = sample_options[sample_choice]
        selected_sample_label = sample_choice

    st.divider()
    st.markdown("### ⚙️ **Forensic Parameters**")
    
    sensitivity = st.slider(
        "Anomaly Sensitivity Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls the z-score threshold for outlier detection across ELA, noise, and edge sharpness."
    )

    ela_quality = st.slider(
        "ELA Recompression Quality",
        min_value=85,
        max_value=98,
        value=95,
        step=1,
        help="Standard quality for baseline Error Level Analysis recompression (default: 95)."
    )

    block_size = st.select_slider(
        "Analysis Block Size (px)",
        options=[16, 32, 64],
        value=32,
        help="Block dimension for local noise and edge sharpness variance analysis."
    )

    st.divider()
    st.markdown("### 🤖 **Decision Engine**")
    classifier_mode = st.radio(
        "Classification Engine:",
        ["Random Forest ML (Trained on IPA features)", "Rule-Based Classical IPA Only"],
        index=0
    )
    use_ml = (classifier_mode == "Random Forest ML (Trained on IPA features)")

    st.divider()
    st.markdown("### 🔬 **Active Forensic Checks**")
    enable_ela = st.checkbox("Error Level Analysis (ELA)", value=True)
    enable_noise = st.checkbox("Sensor Noise Consistency", value=True)
    enable_edge = st.checkbox("Edge & Boundary Sharpness", value=True)
    enable_freq = st.checkbox("2D FFT & DCT Frequency Spectrum", value=True)
    enable_color = st.checkbox("Color & Illumination Vectors", value=True)
    enable_copymove = st.checkbox("Copy-Move Patch Matching", value=True)


# ==============================================================================
# Helper Functions for Radar and Chart Plotting
# ==============================================================================
def plot_radar_chart(features: Dict[str, float]) -> plt.Figure:
    """
    Generate a normalized forensic radar comparison against authentic baseline.
    """
    categories = [
        "ELA Error",
        "Noise Variance",
        "Edge Sharpness Jump",
        "FFT Peaks",
        "Light Divergence",
        "Copy-Move Matches"
    ]
    
    # Normalize features roughly into [0, 100] scale
    vals = [
        min(100.0, features.get("ela_inconsistency_score", 0)),
        min(100.0, features.get("noise_inconsistency_score", 0)),
        min(100.0, features.get("edge_sharpness_discrepancy", 1.0) * 15.0),
        min(100.0, features.get("freq_periodic_peaks", 0) * 15.0),
        min(100.0, (features.get("color_lighting_divergence", 0) / 180.0) * 100.0),
        min(100.0, features.get("copy_move_match_count", 0) * 8.0)
    ]
    
    # Authentic baseline reference (low values for all anomalies)
    baseline = [15.0, 18.0, 20.0, 0.0, 15.0, 0.0]

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    vals += vals[:1]
    baseline += baseline[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    # Plot baseline
    ax.plot(angles, baseline, color='#10b981', linewidth=1.5, linestyle='--', label='Authentic Baseline')
    ax.fill(angles, baseline, color='#10b981', alpha=0.1)

    # Plot target image
    ax.plot(angles, vals, color='#ef4444', linewidth=2.0, label='Target Image')
    ax.fill(angles, vals, color='#ef4444', alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8.5, weight='bold')
    ax.set_ylim(0, 100)
    ax.set_rgrids([25, 50, 75, 100], labels=["25", "50", "75", "100"], color='#94a3b8', size=7)
    ax.grid(color='#cbd5e1', linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=8)
    plt.tight_layout()
    return fig


def plot_frequency_radial(radii: list, profile: list, alpha_slope: float) -> plt.Figure:
    """
    Plot radial power spectrum curve and power-law fit.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    ax.plot(radii, profile, color='#3b82f6', linewidth=2, label='Azimuthal Power log(P)')
    ax.set_xlabel('Spatial Frequency Radius (r)', fontsize=9, fontweight='bold', color='#334155')
    ax.set_ylabel('Log Power', fontsize=9, fontweight='bold', color='#334155')
    ax.set_title(f'Radial Fourier Power Spectrum (Fitted slope alpha = {alpha_slope:.2f})', fontsize=10, fontweight='bold', color='#0f172a')
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_histograms(rgb_hists: dict, hsv_hists: dict) -> plt.Figure:
    """
    Plot RGB and HSV channel distribution curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    fig.patch.set_facecolor('#ffffff')

    # RGB
    ax1.set_facecolor('#f8fafc')
    ax1.plot(rgb_hists['r'], color='#ef4444', label='Red', linewidth=1.5)
    ax1.plot(rgb_hists['g'], color='#22c55e', label='Green', linewidth=1.5)
    ax1.plot(rgb_hists['b'], color='#3b82f6', label='Blue', linewidth=1.5)
    ax1.set_title('RGB Channel Distribution', fontsize=10, fontweight='bold', color='#0f172a')
    ax1.set_xlabel('Intensity (0 - 255)', fontsize=8)
    ax1.set_ylabel('Normalized Frequency', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(fontsize=8)

    # HSV
    ax2.set_facecolor('#f8fafc')
    ax2.plot(hsv_hists['h'], color='#8b5cf6', label='Hue (0-180)', linewidth=1.5)
    ax2.plot(hsv_hists['s'], color='#f59e0b', label='Saturation', linewidth=1.5)
    ax2.plot(hsv_hists['v'], color='#64748b', label='Value (Brightness)', linewidth=1.5)
    ax2.set_title('HSV Color & Lighting Distribution', fontsize=10, fontweight='bold', color='#0f172a')
    ax2.set_xlabel('Bin Index', fontsize=8)
    ax2.set_ylabel('Normalized Frequency', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ==============================================================================
# Main Application Flow
# ==============================================================================
st.markdown('<div class="main-title">ForenScan: Explainable Image Forensics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Multi-Modal Classical Image Processing Analysis (IPA) & Interpretable Anomaly Detection</div>',
    unsafe_allow_html=True
)

# Acquire input image data
image_input = None
input_filename = "image.png"

if source_choice == "Upload Your Own Image" and uploaded_file is not None:
    image_input = uploaded_file.getvalue()
    input_filename = uploaded_file.name
elif source_choice == "Load Demo Sample Case" and selected_sample_path is not None:
    if os.path.exists(selected_sample_path):
        with open(selected_sample_path, "rb") as f:
            image_input = f.read()
        input_filename = os.path.basename(selected_sample_path)
    else:
        st.error(f"Sample file not found at {selected_sample_path}. Please re-run demo asset generator.")

if image_input is None:
    st.info("👈 **Get Started:** Please upload an image or select a sample forensic case from the sidebar.")
    
    # Feature overview cards
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **1. Error Level Analysis (ELA)**
        Re-saves target at quality 95 to amplify compression quantization differences. Spliced elements reveal mismatched compression error.
        """)
        st.markdown("""
        **2. Sensor Noise Consistency**
        Evaluates physical sensor photon shot noise variance across grid blocks. Flagged outliers indicate inpainting or compositing.
        """)
    with col_b:
        st.markdown("""
        **3. Multi-Operator Edge Sharpness**
        Analyzes Canny, Sobel, and Laplacian gradients to identify scissor borders, feathering halos, and localized focus jumps.
        """)
        st.markdown("""
        **4. 2D FFT & DCT Frequency Analysis**
        Identifies periodic checkerboard lattice spikes left by GAN/Diffusion transposed convolutions and unnatural spectral rolloff.
        """)
    with col_c:
        st.markdown("""
        **5. Copy-Move Forgery Detection**
        Detects duplicated textures using patch descriptors, nearest-neighbor matching, and shift-vector spatial clustering.
        """)
        st.markdown("""
        **6. Explainable Forensic Reports**
        Combines cues into a calibrated heatmap, explainable evidence list, downloadable PDF dossier, and CSV metrics.
        """)
    st.stop()
    import sys
    sys.exit(0)


# ==============================================================================
# Processing Pipeline Execution
# ==============================================================================
with st.spinner("Executing Classical Image Processing Analysis (IPA) pipeline..."):
    t_start = time.time()

    # 1. Preprocessing (Resize max 1024, RGB, Gray)
    orig_rgb, resized_rgb, gray_img, scale_factor, (orig_w, orig_h) = load_and_preprocess_image(
        image_input, max_dim=1024
    )

    # 2. EXIF & Metadata Extraction
    metadata_res = extract_metadata(image_input)

    # 3. Error Level Analysis (ELA)
    if enable_ela:
        ela_res = compute_ela(resized_rgb, quality=ela_quality, scale=20.0, anomaly_threshold_std=2.0)
    else:
        ela_res = {"mean_error": 0.0, "std_error": 0.0, "max_error": 0.0, "high_error_ratio": 0.0,
                   "inconsistency_score": 0.0, "ela_image": resized_rgb, "ela_gray": gray_img,
                   "ela_heatmap": resized_rgb, "anomaly_mask": np.zeros_like(gray_img)}

    # 4. Noise Consistency Analysis
    if enable_noise:
        noise_res = analyze_noise_consistency(gray_img, block_size=block_size, sensitivity=sensitivity)
    else:
        noise_res = {"global_noise_mean": 0.0, "global_noise_std": 0.0, "outlier_block_ratio": 0.0,
                     "noise_inconsistency_score": 0.0, "noise_map": np.zeros_like(gray_img),
                     "noise_heatmap": resized_rgb, "anomaly_mask": np.zeros_like(gray_img), "anomalies": []}

    # 5. Multi-scale Edge Analysis
    if enable_edge:
        edge_res = analyze_edges(gray_img, block_size=block_size, sensitivity=sensitivity)
    else:
        edge_res = {"mean_edge_density": 0.0, "mean_sharpness": 0.0, "max_sharpness": 0.0,
                    "sharpness_discrepancy": 1.0, "canny_edges": np.zeros_like(gray_img),
                    "sobel_magnitude": np.zeros_like(gray_img), "laplacian": np.zeros_like(gray_img),
                    "sharpness_heatmap": resized_rgb, "anomaly_mask": np.zeros_like(gray_img), "anomalies": []}

    # 6. Frequency Domain Analysis (FFT & DCT)
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

    # 9. Fused Suspicious-Region Heatmap
    fused_res = generate_fused_heatmap(
        image_rgb=resized_rgb,
        ela_array=ela_res.get("ela_gray"),
        noise_anomaly=noise_res.get("anomaly_mask"),
        edge_anomaly=edge_res.get("anomaly_mask"),
        copy_move_mask=copy_move_res.get("copy_move_mask"),
        alpha=0.55,
        threshold=0.35
    )

    # 10. Quantitative Feature Aggregation
    feat_vec, feat_dict = extract_feature_vector(
        ela_res, noise_res, edge_res, freq_res, color_res, copy_move_res, metadata_res, fused_res
    )

    # 11. Classifier Decision Engine & Explainable Inference
    verdict_info = predict_authenticity(
        features_dict=feat_dict,
        metadata_dict=metadata_res,
        use_ml=use_ml
    )

    processing_time = round(time.time() - t_start, 2)


# ==============================================================================
# Top Summary Metrics & Verdict Banner
# ==============================================================================
verdict = verdict_info["verdict"]
auth_score = verdict_info["authenticity_score"]
tamper_score = verdict_info["tamper_score"]
ai_score = verdict_info["ai_score"]
conf = verdict_info["confidence"]

# Dynamic Verdict Banner
if "Authentic" in verdict:
    st.markdown(f"""
    <div class="verdict-banner-auth">
        <h3 style="margin:0; color:#065f46;">🛡️ VERDICT: {verdict.upper()}</h3>
        <p style="margin:0.2rem 0 0 0; color:#047857; font-size:0.95rem;">
            Uniform sensor noise, natural Fourier power-law decay, consistent ELA compression history, and intact optical characteristics.
            Analysis completed in {processing_time}s using <b>{verdict_info['method']}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
elif "AI" in verdict:
    st.markdown(f"""
    <div class="verdict-banner-ai">
        <h3 style="margin:0; color:#1e40af;">🤖 VERDICT: {verdict.upper()}</h3>
        <p style="margin:0.2rem 0 0 0; color:#1d4ed8; font-size:0.95rem;">
            Periodic Fourier lattice spikes detected (transposed convolution artifacts), absence of physical sensor shot noise, and unnatural spectral decay.
            Analysis completed in {processing_time}s using <b>{verdict_info['method']}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="verdict-banner-tamper">
        <h3 style="margin:0; color:#991b1b;">⚠️ VERDICT: {verdict.upper()}</h3>
        <p style="margin:0.2rem 0 0 0; color:#b91c1c; font-size:0.95rem;">
            Localized compression anomalies, discordant sensor noise blocks, unnatural boundary sharpness, or duplicate copy-move patches detected.
            Analysis completed in {processing_time}s using <b>{verdict_info['method']}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 4 Key Quantitative Metrics Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Authenticity Score</div>
        <div class="metric-value" style="color:#10b981;">{auth_score}%</div>
        <div class="metric-sub">Photographic Integrity</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Tampering Probability</div>
        <div class="metric-value" style="color:#ef4444;">{tamper_score}%</div>
        <div class="metric-sub">Splicing / Inpainting Risk</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">AI Generation Risk</div>
        <div class="metric-value" style="color:#3b82f6;">{ai_score}%</div>
        <div class="metric-sub">Diffusion / GAN Fingerprint</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Verdict Confidence</div>
        <div class="metric-value" style="color:#6366f1;">{conf}%</div>
        <div class="metric-sub">Statistical Certainty</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# Detailed Interactive Forensic Tabs
# ==============================================================================
tab_overview, tab_ela_noise, tab_edge_freq, tab_color, tab_copymove, tab_reasons, tab_metadata, tab_features, tab_report = st.tabs([
    "🔍 Visual Overview",
    "🔬 ELA & Noise",
    "📐 Edges & Frequency",
    "🎨 Color & Lighting",
    "🔎 Copy-Move",
    "📋 Explainable Findings",
    "🏷️ EXIF Metadata",
    "📊 Feature Manifest",
    "📑 Download Reports"
])

# ------------------------------------------------------------------------------
# TAB 1: Visual Overview & Triage
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown("#### Primary Visual Forensic Triage")
    st.caption("Side-by-side inspection of original input, Error Level Analysis (ELA), and combined multi-modal suspicious-region overlay.")

    ov_c1, ov_c2, ov_c3 = st.columns(3)
    with ov_c1:
        st.image(resized_rgb, caption=f"Original Input ({orig_w}x{orig_h}px)", use_container_width=True)
    with ov_c2:
        st.image(ela_res["ela_heatmap"], caption=f"ELA Heatmap (Q={ela_quality})", use_container_width=True)
    with ov_c3:
        st.image(fused_res["annotated_image"], caption="Suspicious Region Heatmap Overlay", use_container_width=True)

    if fused_res["bounding_boxes"]:
        st.markdown(f"**Identified Suspicious Regions ({len(fused_res['bounding_boxes'])} detected):**")
        bb_df = pd.DataFrame(fused_res["bounding_boxes"])
        st.dataframe(bb_df, use_container_width=True)
    else:
        st.info("No concentrated high-suspicion clusters isolated at current sensitivity.")

# ------------------------------------------------------------------------------
# TAB 2: ELA & Sensor Noise Deep-Dive
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
        - **High-Error Pixel Ratio:** `{ela_res['high_error_ratio']}%`
        - **Local Inconsistency Score:** `{ela_res['inconsistency_score']}/100`
        """)
        st.caption("Spliced or inpainted objects that have not undergone the same JPEG compression cycle as the host canvas exhibit distinct brightness in the ELA heatmap.")

    with col_en2:
        st.markdown("##### Sensor Noise Consistency")
        st.image(noise_res["noise_heatmap"], caption=f"Block-wise Noise Residual Map ({block_size}x{block_size}px)", use_container_width=True)
        st.markdown(f"""
        - **Global Noise Floor (MAD):** `{noise_res['global_noise_mean']}`
        - **Noise Variance Deviation:** `{noise_res['global_noise_std']}`
        - **Outlier Noise Blocks:** `{noise_res['outlier_block_ratio']}%`
        - **Noise Inconsistency Score:** `{noise_res['noise_inconsistency_score']}/100`
        """)
        st.caption("Physical digital camera sensors produce uniform shot noise. Discordant high/low noise regions point to selective filtering, inpainting, or composite pasting.")

# ------------------------------------------------------------------------------
# TAB 3: Edge & Frequency Domain Analysis
# ------------------------------------------------------------------------------
with tab_edge_freq:
    st.markdown("#### Boundary Sharpness & 2D Frequency Transforms")

    ed_c1, ed_c2, ed_c3 = st.columns(3)
    with ed_c1:
        st.markdown("##### Canny Edge Map")
        st.image(edge_res["canny_edges"], caption=f"Edge Density: {edge_res['mean_edge_density']}%", use_container_width=True)
    with ed_c2:
        st.markdown("##### Sobel Gradient Magnitude")
        st.image(edge_res["sobel_magnitude"], caption=f"Mean Sharpness: {edge_res['mean_sharpness']}", use_container_width=True)
    with ed_c3:
        st.markdown("##### Regional Sharpness Heatmap")
        st.image(edge_res["sharpness_heatmap"], caption=f"Sharpness Discrepancy: {edge_res['sharpness_discrepancy']}x", use_container_width=True)

    st.divider()

    fq_c1, fq_c2, fq_c3 = st.columns([1, 1, 1.2])
    with fq_c1:
        st.markdown("##### 2D FFT Magnitude Spectrum")
        st.image(freq_res["fft_spectrum"], caption=f"Periodic Spikes Detected: {freq_res['periodic_peak_count']}", use_container_width=True)
        st.caption("Bright lattice dots away from center reveal transposed convolution upsampling grids common in GAN & Diffusion architectures.")
    with fq_c2:
        st.markdown("##### 8x8 DCT Power Grid")
        st.image(freq_res["dct_spectrum"], caption=f"DCT AC Energy: {freq_res['dct_ac_energy']}", use_container_width=True)
        st.caption("Natural JPEG images adhere to standard 8x8 block DCT distributions; uncompressed synthetic elements disrupt this grid.")
    with fq_c3:
        st.markdown("##### Azimuthal Radial Power Profile")
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

    st.info(f"**Max Illumination Divergence:** `{color_res.get('lighting_divergence_angle', 0)}°` across quadrants (Score: `{color_res.get('lighting_inconsistency_score', 0)}/100`). Angular differences > 90° suggest compositing from different light sources.")

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
            st.error("⚠️ **Cloned content detected:** High confidence copy-move forgery confirmed. Colored boxes and connecting lines mark duplicated regions.")
        else:
            st.success("✅ **No copy-move cloning detected:** No significant non-adjacent duplicate patch clusters observed.")

# ------------------------------------------------------------------------------
# TAB 6: Explainable Findings & Reasons
# ------------------------------------------------------------------------------
with tab_reasons:
    st.markdown("#### Explainable Forensic Evidence Findings")
    st.caption("Transparent justifications connecting physical image processing anomalies to the final verdict.")

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
            <div style="padding: 0.8rem 1rem; border-radius: 8px; background: white; border: 1px solid #e2e8f0; margin-bottom: 0.6rem;">
                {badge_html} &nbsp; <b>{cat}:</b> {txt}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No anomalies identified.")

# ------------------------------------------------------------------------------
# TAB 7: EXIF & Hardware Metadata Inspector
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

        st.markdown("##### GPS Coordinates")
        gps = metadata_res.get("gps_info", {})
        if gps:
            st.json(gps)
        else:
            st.caption("No GPS geotag metadata embedded.")

    if metadata_res.get("raw_tags"):
        with st.expander("Inspect All Raw EXIF Tags"):
            st.dataframe(pd.DataFrame(list(metadata_res["raw_tags"].items()), columns=["Tag", "Value"]), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 8: Feature Manifest & Radar Chart
# ------------------------------------------------------------------------------
with tab_features:
    st.markdown("#### Quantitative IPA Feature Manifest")
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
# TAB 9: Downloadable Forensic Examination Reports
# ------------------------------------------------------------------------------
with tab_report:
    st.markdown("#### Export Forensic Dossier")
    st.caption("Generate publication-ready PDF examination reports and CSV feature manifests.")

    # Generate buffer bytes for visual exhibits
    orig_bytes = image_to_bytes(resized_rgb, format="JPEG")
    ela_bytes = image_to_bytes(ela_res["ela_heatmap"], format="JPEG")
    overlay_bytes = image_to_bytes(fused_res["annotated_image"], format="JPEG")

    # Generate PDF
    pdf_bytes = generate_pdf_report(
        orig_img_bytes=orig_bytes,
        ela_img_bytes=ela_bytes,
        overlay_img_bytes=overlay_bytes,
        verdict_info=verdict_info,
        metadata_dict=metadata_res,
        features_dict=feat_dict,
        filename=input_filename
    )

    # Generate CSV
    csv_str = generate_csv_report(
        features_dict=feat_dict,
        metadata_dict=metadata_res,
        verdict_info=verdict_info,
        filename=input_filename
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
        st.caption("Includes case summary, high-resolution visual exhibits, explainable evidence table, and hardware audit.")

    with r_c2:
        st.download_button(
            label="📊 Download Quantitative IPA CSV Manifest",
            data=csv_str.encode('utf-8'),
            file_name=f"forenscan_features_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Complete CSV data logging of all 25 numerical IPA metrics and metadata parameters.")
