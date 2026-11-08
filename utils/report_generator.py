"""
Forensic Report Generator module for ForenScan.
Generates comprehensive, court-admissible-grade PDF forensic examination reports
via ReportLab, and exportable CSV feature manifests for forensic data logging.
"""

import io
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.units import inch


def generate_csv_report(
    features_dict: Dict[str, float],
    metadata_dict: Dict[str, Any],
    verdict_info: Dict[str, Any],
    filename: str = "image.png"
) -> str:
    """
    Generate a detailed CSV string containing forensic findings, EXIF metadata, and features.
    """
    lines = []
    lines.append("# FORENSCAN DIGITAL IMAGE FORENSIC REPORT")
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Target File: {filename}")
    lines.append("")
    
    # Executive Verdict
    lines.append("SECTION,PARAMETER,VALUE")
    lines.append(f"Verdict,Final Determination,{verdict_info.get('verdict', 'Unknown')}")
    lines.append(f"Verdict,Authenticity Score,{verdict_info.get('authenticity_score', 0)}%")
    lines.append(f"Verdict,Tampering Probability,{verdict_info.get('tamper_score', 0)}%")
    lines.append(f"Verdict,AI Generation Probability,{verdict_info.get('ai_score', 0)}%")
    lines.append(f"Verdict,Confidence Level,{verdict_info.get('confidence', 0)}%")
    lines.append(f"Verdict,Analysis Method,{verdict_info.get('method', 'Classical IPA')}")
    lines.append("")

    # Metadata
    lines.append("SECTION,METADATA_FIELD,VALUE")
    for k, v in metadata_dict.get("basic_info", {}).items():
        lines.append(f"Basic Metadata,{k},{v}")
    for k, v in metadata_dict.get("camera_info", {}).items():
        lines.append(f"Camera Hardware,{k},{v}")
    sw_traces = ", ".join(metadata_dict.get("software_traces", [])) or "None Detected"
    lines.append(f"Software Traces,Detected Tools,\"{sw_traces}\"")
    lines.append("")

    # Anomalies
    lines.append("SECTION,SEVERITY,ANOMALY_DESCRIPTION")
    for r in verdict_info.get("reasons", []):
        sev = r.get("severity", "Info")
        desc = r.get("text", "").replace('"', '""')
        lines.append(f"Detected Anomaly,{sev},\"{desc}\"")
    lines.append("")

    # Quantitative IPA Features
    lines.append("SECTION,IPA_FEATURE_NAME,MEASURED_VALUE")
    for feat_name, feat_val in features_dict.items():
        lines.append(f"Quantitative Metric,{feat_name},{round(feat_val, 4)}")

    return "\n".join(lines)


def generate_pdf_report(
    orig_img_bytes: bytes,
    ela_img_bytes: bytes,
    overlay_img_bytes: bytes,
    verdict_info: Dict[str, Any],
    metadata_dict: Dict[str, Any],
    features_dict: Dict[str, float],
    filename: str = "analyzed_image.png"
) -> bytes:
    """
    Generate an authoritative, professional PDF forensic report using ReportLab.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748b')
    )
    
    heading2_style = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11
    )

    cell_normal = ParagraphStyle(
        'CellNormal',
        parent=body_style,
        fontSize=8,
        leading=11
    )

    story = []

    # Header Banner
    story.append(Paragraph("FORENSCAN // DIGITAL IMAGE FORENSICS REPORT", title_style))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    file_sha256 = hashlib.sha256(orig_img_bytes).hexdigest()[:24] + "..."
    story.append(Paragraph(
        f"<b>Analysis Date:</b> {now_str} &nbsp;|&nbsp; <b>File:</b> {filename} &nbsp;|&nbsp; <b>SHA-256:</b> {file_sha256}",
        subtitle_style
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=8))

    # Executive Verdict Banner Box
    verdict = verdict_info.get("verdict", "Unknown")
    auth_score = verdict_info.get("authenticity_score", 0.0)
    tamper_score = verdict_info.get("tamper_score", 0.0)
    ai_score = verdict_info.get("ai_score", 0.0)
    conf = verdict_info.get("confidence", 0.0)

    if "Authentic" in verdict:
        box_bg = colors.HexColor('#f0fdf4')
        box_border = colors.HexColor('#22c55e')
        verdict_color = "#15803d"
    elif "AI" in verdict:
        box_bg = colors.HexColor('#eff6ff')
        box_border = colors.HexColor('#3b82f6')
        verdict_color = "#1d4ed8"
    else:
        box_bg = colors.HexColor('#fef2f2')
        box_border = colors.HexColor('#ef4444')
        verdict_color = "#b91c1c"

    verdict_html = f"""
    <font size="13" color="{verdict_color}"><b>EXECUTIVE DETERMINATION: {verdict.upper()}</b></font><br/>
    <font size="9" color="#334155">
    Authenticity Score: <b>{auth_score}%</b> &nbsp;|&nbsp; 
    Tamper Probability: <b>{tamper_score}%</b> &nbsp;|&nbsp; 
    AI Generation Risk: <b>{ai_score}%</b> &nbsp;|&nbsp; 
    Confidence: <b>{conf}%</b>
    </font>
    """
    
    verdict_table = Table([[Paragraph(verdict_html, body_style)]], colWidths=[540])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), box_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, box_border),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # Visual Forensic Exhibits Side-by-Side
    story.append(Paragraph("Visual Forensic Exhibits", heading2_style))
    
    img_w = 2.4 * inch
    img_h = 1.8 * inch

    def make_rl_image(img_b):
        buf = io.BytesIO(img_b)
        return RLImage(buf, width=img_w, height=img_h)

    try:
        im1 = make_rl_image(orig_img_bytes)
        im2 = make_rl_image(ela_img_bytes)
        im3 = make_rl_image(overlay_img_bytes)

        exhibit_table = Table([
            [Paragraph("<b>Exhibit A: Original</b>", cell_bold),
             Paragraph("<b>Exhibit B: ELA Heatmap</b>", cell_bold),
             Paragraph("<b>Exhibit C: Fused Suspicion Overlay</b>", cell_bold)],
            [im1, im2, im3]
        ], colWidths=[180, 180, 180])
        exhibit_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(exhibit_table)
    except Exception:
        story.append(Paragraph("[Visual exhibits omitted due to rendering constraint]", body_style))

    story.append(Spacer(1, 10))

    # Explainable Forensic Evidence Findings Table
    story.append(Paragraph("Explainable Findings & Detected Anomalies", heading2_style))
    reasons = verdict_info.get("reasons", [])
    
    findings_rows = [[
        Paragraph("<b>Severity</b>", cell_bold),
        Paragraph("<b>Category</b>", cell_bold),
        Paragraph("<b>Observed Forensic Evidence & Explanation</b>", cell_bold)
    ]]

    for r in reasons:
        sev = r.get("severity", "Info")
        t = r.get("type", "General")
        desc = r.get("text", "")
        
        if sev == "Critical":
            sev_badge = f'<font color="#b91c1c"><b>CRITICAL</b></font>'
        elif sev == "Warning":
            sev_badge = f'<font color="#c2410c"><b>WARNING</b></font>'
        elif sev == "Normal":
            sev_badge = f'<font color="#15803d"><b>NORMAL</b></font>'
        else:
            sev_badge = f'<font color="#475569"><b>INFO</b></font>'

        findings_rows.append([
            Paragraph(sev_badge, cell_normal),
            Paragraph(f"<b>{t}</b>", cell_normal),
            Paragraph(desc, cell_normal)
        ])

    findings_table = Table(findings_rows, colWidths=[65, 120, 355])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 10))

    # Metadata & Hardware Audit
    story.append(Paragraph("EXIF & Hardware Metadata Audit", heading2_style))
    cam = metadata_dict.get("camera_info", {})
    sw_traces = metadata_dict.get("software_traces", [])
    
    meta_rows = [
        [Paragraph("<b>Parameter</b>", cell_bold), Paragraph("<b>Audit Value</b>", cell_bold),
         Paragraph("<b>Parameter</b>", cell_bold), Paragraph("<b>Audit Value</b>", cell_bold)],
        [Paragraph("Camera Make", cell_normal), Paragraph(str(cam.get("Make", "Not recorded")), cell_normal),
         Paragraph("Original Dimensions", cell_normal), Paragraph(f"{metadata_dict.get('basic_info', {}).get('width')} x {metadata_dict.get('basic_info', {}).get('height')}", cell_normal)],
        [Paragraph("Camera Model", cell_normal), Paragraph(str(cam.get("Model", "Not recorded")), cell_normal),
         Paragraph("Image Format", cell_normal), Paragraph(str(metadata_dict.get('basic_info', {}).get('format', 'Unknown')), cell_normal)],
        [Paragraph("Software Traces", cell_normal), Paragraph(", ".join(sw_traces) if sw_traces else "None detected", cell_normal),
         Paragraph("GPS Data", cell_normal), Paragraph("Present" if metadata_dict.get("gps_info") else "None", cell_normal)],
    ]
    meta_table = Table(meta_rows, colWidths=[110, 160, 110, 160])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Quantitative IPA Feature Metrics Table (Key Indicators)
    story.append(Paragraph("Key Quantitative Forensic Metrics", heading2_style))
    metrics_rows = [
        [Paragraph("<b>Metric Name</b>", cell_bold), Paragraph("<b>Value</b>", cell_bold),
         Paragraph("<b>Metric Name</b>", cell_bold), Paragraph("<b>Value</b>", cell_bold)],
        [Paragraph("ELA Error Inconsistency", cell_normal), Paragraph(f"{features_dict.get('ela_inconsistency_score', 0):.2f}", cell_normal),
         Paragraph("Spectral Decay Exponent (alpha)", cell_normal), Paragraph(f"{features_dict.get('freq_spectral_slope', 0):.2f}", cell_normal)],
        [Paragraph("Sensor Noise Outliers (%)", cell_normal), Paragraph(f"{features_dict.get('noise_outlier_ratio', 0):.2f}%", cell_normal),
         Paragraph("Periodic FFT Spikes Count", cell_normal), Paragraph(f"{int(features_dict.get('freq_periodic_peaks', 0))}", cell_normal)],
        [Paragraph("Edge Sharpness Discrepancy", cell_normal), Paragraph(f"{features_dict.get('edge_sharpness_discrepancy', 0):.2f}x", cell_normal),
         Paragraph("Quadrant Light Divergence", cell_normal), Paragraph(f"{features_dict.get('color_lighting_divergence', 0):.1f}°", cell_normal)],
        [Paragraph("Copy-Move Duplicate Clusters", cell_normal), Paragraph(f"{int(features_dict.get('copy_move_clusters', 0))}", cell_normal),
         Paragraph("Fused Anomaly Coverage", cell_normal), Paragraph(f"{features_dict.get('fused_coverage_pct', 0):.2f}%", cell_normal)],
    ]
    metrics_table = Table(metrics_rows, colWidths=[170, 100, 170, 100])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # Methodology & Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#e2e8f0'), spaceBefore=4, spaceAfter=6))
    footer_text = (
        "<b>Forensic Methodology:</b> ForenScan conducts explainable non-destructive forensic analysis via Error Level Analysis (ELA), "
        "multiscale sensor noise residual modeling, Fourier azimuthal power-spectrum decay, edge gradient consistency, and spatial shift-vector "
        "patch clustering. Results are intended for forensic triage and screening. Generated by ForenScan IPA Engine."
    )
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=subtitle_style, fontSize=7.5, leading=10)))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
