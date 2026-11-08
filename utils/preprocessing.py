"""
Preprocessing module for ForenScan.
Handles image loading, color space conversions, resizing to max 1024px,
and standardized image representations for forensic pipeline analysis.
"""

import io
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageOps
import cv2


def load_and_preprocess_image(
    image_source: Union[str, bytes, io.BytesIO, Image.Image],
    max_dim: int = 1024
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Load an image from various input types and prepare it for forensic analysis.
    
    Parameters:
        image_source: File path, byte buffer, or PIL.Image
        max_dim: Maximum dimension (width or height) allowed (default: 1024)
        
    Returns:
        orig_rgb: np.ndarray (H, W, 3) in uint8, original dimensions RGB
        resized_rgb: np.ndarray (H', W', 3) in uint8, bounded by max_dim
        gray: np.ndarray (H', W') in uint8, grayscale of resized image
        scale_factor: float scaling ratio applied
        original_dims: (orig_width, orig_height)
    """
    if isinstance(image_source, Image.Image):
        pil_img = image_source
    elif isinstance(image_source, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, io.BytesIO):
        image_source.seek(0)
        pil_img = Image.open(image_source)
    elif isinstance(image_source, str):
        pil_img = Image.open(image_source)
    else:
        raise ValueError(f"Unsupported image_source type: {type(image_source)}")

    # Handle EXIF orientation if present to ensure proper display
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # Ensure RGB format (strip alpha channel safely if present)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    orig_w, orig_h = pil_img.size
    orig_rgb = np.array(pil_img, dtype=np.uint8)

    # Calculate scale factor for max_dim bounding
    max_side = max(orig_w, orig_h)
    if max_side > max_dim:
        scale_factor = max_dim / float(max_side)
        new_w = max(1, int(round(orig_w * scale_factor)))
        new_h = max(1, int(round(orig_h * scale_factor)))
        resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_rgb = np.array(resized_pil, dtype=np.uint8)
    else:
        scale_factor = 1.0
        resized_rgb = orig_rgb.copy()

    # Convert to grayscale for single-channel analyses (Sobel, Laplacian, FFT, Noise)
    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)

    return orig_rgb, resized_rgb, gray, scale_factor, (orig_w, orig_h)


def image_to_bytes(img: Union[np.ndarray, Image.Image], format: str = "PNG") -> bytes:
    """
    Convert a numpy array (RGB or Grayscale) or PIL Image to binary bytes.
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            pil_img = Image.fromarray(img, mode="L")
        elif img.ndim == 3 and img.shape[2] == 3:
            pil_img = Image.fromarray(img, mode="RGB")
        elif img.ndim == 3 and img.shape[2] == 4:
            pil_img = Image.fromarray(img, mode="RGBA")
        else:
            raise ValueError(f"Invalid array shape for image conversion: {img.shape}")
    else:
        pil_img = img

    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()
