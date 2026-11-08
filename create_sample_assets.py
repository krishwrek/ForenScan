"""
Asset & Demo Generator script for ForenScan.
1. Generates 4 representative test images with known ground-truth forensic properties:
   - sample1_authentic.jpg: Natural scene with authentic sensor noise, uniform JPEG compression, and valid Canon camera EXIF.
   - sample2_spliced_tampered.jpg: Image with an inserted/spliced object having discordant noise, sharp edges, distinct compression history, and Adobe Photoshop signature.
   - sample3_copy_move.jpg: Scene with an explicit cloned/duplicated pattern at another coordinate.
   - sample4_ai_generated.png: Synthetic image with checkerboard upsampling artifacts and smooth noise.
2. Trains and saves the Random Forest classifier to models/trained_model.pkl.
"""

import os
import io
import numpy as np
from PIL import Image, ExifTags, PngImagePlugin
import cv2
from utils.classifier import train_synthetic_classifier


def create_sample_assets(output_dir: str = "sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    width, height = 640, 480

    # ==========================================
    # Sample 1: Authentic Camera Photo
    # ==========================================
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Natural landscape background
    sky = np.stack([
        135 - 50 * yy,
        185 - 40 * yy,
        235 - 30 * yy
    ], axis=2)
    
    ground = np.stack([
        34 + 60 * xx,
        120 + 40 * xx,
        45 + 30 * yy
    ], axis=2)
    
    img1 = np.where(yy[:, :, np.newaxis] < 0.55, sky, ground).astype(np.float32)
    img1 += (25 * np.sin(xx * 6) * np.cos(yy * 4))[:, :, np.newaxis]
    
    # Authentic sensor noise
    sensor_noise = np.random.normal(0, 3.5, img1.shape)
    img1 = np.clip(img1 + sensor_noise, 0, 255).astype(np.uint8)
    
    pil_img1 = Image.fromarray(img1, mode="RGB")
    
    # Add authentic camera EXIF
    exif1 = Image.Exif()
    exif1[ExifTags.Base.Make] = "Canon"
    exif1[ExifTags.Base.Model] = "Canon EOS 5D Mark IV"
    exif1[ExifTags.Base.DateTime] = "2024:06:14 11:24:05"
    exif1[ExifTags.Base.Software] = "Canon EOS Firmware 1.3.3"
    
    sample1_path = os.path.join(output_dir, "sample1_authentic.jpg")
    pil_img1.save(sample1_path, "JPEG", quality=92, exif=exif1)
    print(f"Created: {sample1_path}")

    # ==========================================
    # Sample 2: Spliced & Inpainted Tampered Photo
    # ==========================================
    base_img = img1.copy()
    base_pil = Image.fromarray(base_img, mode="RGB")
    buf = io.BytesIO()
    base_pil.save(buf, "JPEG", quality=70)
    buf.seek(0)
    base_compressed = np.array(Image.open(buf), dtype=np.uint8)

    # Insert a foreign red floating sphere / satellite near the upper right
    spliced_img = base_compressed.copy()
    cx, cy, radius = 460, 160, 55

    y_grid, x_grid = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    sphere_mask = dist_from_center <= radius

    norm_dx = (x_grid - (cx - 20)) / float(radius)
    norm_dy = (y_grid - (cy - 20)) / float(radius)
    sphere_shade = np.clip(1.0 - np.sqrt(norm_dx**2 + norm_dy**2), 0.1, 1.0)
    
    sphere_rgb = np.stack([
        np.clip(240 * sphere_shade, 0, 255),
        np.clip(40 * sphere_shade, 0, 255),
        np.clip(40 * sphere_shade, 0, 255)
    ], axis=2).astype(np.uint8)

    spliced_img[sphere_mask] = sphere_rgb[sphere_mask]
    
    # Metadata indicating Photoshop editing
    exif2 = Image.Exif()
    exif2[ExifTags.Base.Software] = "Adobe Photoshop 25.4 (Windows)"
    exif2[ExifTags.Base.DateTime] = "2024:06:15 22:10:48"

    sample2_path = os.path.join(output_dir, "sample2_spliced_tampered.jpg")
    Image.fromarray(spliced_img).save(sample2_path, "JPEG", quality=96, exif=exif2)
    print(f"Created: {sample2_path}")

    # ==========================================
    # Sample 3: Copy-Move Forged Image
    # ==========================================
    cm_img = img1.copy()
    def draw_texture_badge(target_arr, bx, by):
        cv2.circle(target_arr, (bx, by), 35, (30, 90, 30), -1)
        cv2.circle(target_arr, (bx, by), 25, (80, 160, 60), -1)
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            px = int(bx + 30 * np.cos(rad))
            py = int(by + 30 * np.sin(rad))
            cv2.line(target_arr, (bx, by), (px, py), (200, 220, 100), 3)

    # Source patch
    draw_texture_badge(cm_img, 150, 300)
    
    # Exact duplicate cloned patch at (420, 300)
    patch = cm_img[260:340, 110:190].copy()
    cm_img[260:340, 380:460] = patch

    sample3_path = os.path.join(output_dir, "sample3_copy_move.jpg")
    Image.fromarray(cm_img).save(sample3_path, "JPEG", quality=94)
    print(f"Created: {sample3_path}")

    # ==========================================
    # Sample 4: AI-Generated Synthetic Portrait
    # ==========================================
    ai_img = np.zeros((height, width, 3), dtype=np.float32)
    ai_img[:, :, 0] = 70 + 40 * np.sin(xx * 3)
    ai_img[:, :, 1] = 60 + 50 * np.cos(yy * 3)
    ai_img[:, :, 2] = 110 + 60 * np.sin((xx + yy) * 2)

    face_mask = (((xx - 0.5) / 0.22)**2 + ((yy - 0.48) / 0.32)**2) <= 1.0
    ai_img[face_mask, 0] = 230
    ai_img[face_mask, 1] = 190
    ai_img[face_mask, 2] = 165

    eye_l = (((xx - 0.42) / 0.04)**2 + ((yy - 0.43) / 0.025)**2) <= 1.0
    eye_r = (((xx - 0.58) / 0.04)**2 + ((yy - 0.43) / 0.025)**2) <= 1.0
    ai_img[eye_l | eye_r, :] = [40, 80, 140]

    # Extreme smoothing (no camera sensor noise)
    ai_img = cv2.GaussianBlur(ai_img, (7, 7), 0)

    # Periodic 16x16 grid artifact (classic transposed conv upsampling signature)
    checker = (np.sin(xx * width * (np.pi / 8)) * np.cos(yy * height * (np.pi / 8)))
    ai_img += 16.0 * checker[:, :, np.newaxis]

    ai_uint8 = np.clip(ai_img, 0, 255).astype(np.uint8)
    
    # Metadata with Midjourney / AI trace
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("Software", "Midjourney v6.0")
    png_info.add_text("Description", "Cyberpunk portrait rendered via generative diffusion")

    sample4_path = os.path.join(output_dir, "sample4_ai_generated.png")
    Image.fromarray(ai_uint8).save(sample4_path, "PNG", pnginfo=png_info)
    print(f"Created: {sample4_path}")

    # ==========================================
    # Train and Save Model
    # ==========================================
    print("Training Random Forest model on calibrated IPA feature space...")
    train_synthetic_classifier(save_path=os.path.join("models", "trained_model.pkl"))
    print("Model trained and saved to models/trained_model.pkl")


if __name__ == "__main__":
    create_sample_assets()
