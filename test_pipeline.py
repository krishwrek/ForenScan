"""
Verification test script for ForenScan pipeline.
Runs all forensic sub-modules, feature extraction, classifier, and PDF/CSV generation.
"""

import os
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
from utils.classifier import predict_authenticity
from utils.report_generator import generate_pdf_report, generate_csv_report


def test_full_pipeline():
    samples = [
        "sample_images/sample1_authentic.jpg",
        "sample_images/sample2_spliced_tampered.jpg",
        "sample_images/sample3_copy_move.jpg",
        "sample_images/sample4_ai_generated.png"
    ]

    for sample_path in samples:
        print(f"\n--- Testing: {sample_path} ---")
        assert os.path.exists(sample_path), f"File {sample_path} does not exist!"

        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        # 1. Preprocessing
        orig_rgb, resized_rgb, gray_img, scale, dims = load_and_preprocess_image(img_bytes, max_dim=1024)
        print(f"Loaded: orig dims {dims}, resized to {resized_rgb.shape}, scale {scale:.2f}")

        # 2. Metadata
        meta_res = extract_metadata(img_bytes)
        print(f"Metadata: has_exif={meta_res['has_exif']}, software_traces={meta_res['software_traces']}")

        # 3. ELA
        ela_res = compute_ela(resized_rgb, quality=95)
        print(f"ELA: mean_error={ela_res['mean_error']}, inconsistency={ela_res['inconsistency_score']}")

        # 4. Noise
        noise_res = analyze_noise_consistency(gray_img)
        print(f"Noise: global_mean={noise_res['global_noise_mean']}, outliers={noise_res['outlier_block_ratio']}%")

        # 5. Edges
        edge_res = analyze_edges(gray_img)
        print(f"Edges: mean_sharpness={edge_res['mean_sharpness']}, discrepancy={edge_res['sharpness_discrepancy']}x")

        # 6. Frequency
        freq_res = analyze_frequency_domain(gray_img)
        print(f"Freq: peaks={freq_res['periodic_peak_count']}, slope={freq_res['spectral_slope']}")

        # 7. Color & Lighting
        color_res = analyze_color_and_lighting(resized_rgb)
        print(f"Color: light_div={color_res['lighting_divergence_angle']} deg")

        # 8. Copy-Move
        cm_res = detect_copy_move(resized_rgb)
        print(f"Copy-Move: matches={cm_res['match_count']}, clusters={cm_res['cluster_count']}")

        # 9. Fused Heatmap
        fused_res = generate_fused_heatmap(
            image_rgb=resized_rgb,
            ela_array=ela_res["ela_gray"],
            noise_anomaly=noise_res["anomaly_mask"],
            edge_anomaly=edge_res["anomaly_mask"],
            copy_move_mask=cm_res["copy_move_mask"]
        )
        print(f"Fused: coverage={fused_res['anomaly_coverage_pct']}%, boxes={len(fused_res['bounding_boxes'])}")

        # 10. Features
        vec, feat_dict = extract_feature_vector(
            ela_res, noise_res, edge_res, freq_res, color_res, cm_res, meta_res, fused_res
        )
        assert len(vec) == 25, f"Expected 25 features, got {len(vec)}"
        print(f"Features: extracted {len(vec)} features successfully")

        # 11. Classifier
        pred = predict_authenticity(feat_dict, meta_res, use_ml=True)
        print(f"Verdict: {pred['verdict']} | Authenticity: {pred['authenticity_score']}% | Tamper: {pred['tamper_score']}% | AI: {pred['ai_score']}%")

        # 12. Report Generation
        orig_b = image_to_bytes(resized_rgb)
        ela_b = image_to_bytes(ela_res["ela_heatmap"])
        overlay_b = image_to_bytes(fused_res["annotated_image"])

        pdf_bytes = generate_pdf_report(
            orig_img_bytes=orig_b,
            ela_img_bytes=ela_b,
            overlay_img_bytes=overlay_b,
            verdict_info=pred,
            metadata_dict=meta_res,
            features_dict=feat_dict,
            filename=os.path.basename(sample_path)
        )
        assert len(pdf_bytes) > 1000, "PDF generation produced empty or corrupted output"
        print(f"PDF generated: {len(pdf_bytes)} bytes")

        csv_str = generate_csv_report(feat_dict, meta_res, pred, filename=os.path.basename(sample_path))
        assert len(csv_str) > 100, "CSV generation produced empty output"
        print(f"CSV generated: {len(csv_str)} chars")

    print("\n✅ All 4 test cases passed successfully!")


if __name__ == "__main__":
    test_full_pipeline()
