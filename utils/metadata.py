"""
Metadata & EXIF analysis module for ForenScan.
Extracts hardware, camera, GPS, timestamp information, and scans for
digital editing software or generative AI software signatures.
"""

import io
from typing import Dict, Any, List, Optional
from PIL import Image, ExifTags


KNOWN_EDITING_SIGNATURES = [
    "photoshop", "gimp", "canva", "lightroom", "affinity", "paint.net",
    "snapseed", "pixlr", "photoscape", "coreldraw", "capture one",
    "pixelmator", "procreate", "vsco", "facetune", "picsart"
]

KNOWN_AI_SIGNATURES = [
    "midjourney", "stable diffusion", "stablediffusion", "dall-e", "dalle",
    "novelai", "comfyui", "automatic1111", "fooocus", "invokeai",
    "adobe firefly", "firefly", "leonardo.ai", "flux.1", "flux",
    "sdxl", "dreamstudio", "bing image creator", "copilot designer"
]


def _convert_to_degrees(value) -> Optional[float]:
    """Helper function to convert GPS rational tuples (degrees, minutes, seconds) to decimal."""
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def extract_metadata(image_source) -> Dict[str, Any]:
    """
    Extract and analyze EXIF metadata from an image.
    
    Parameters:
        image_source: File path, bytes, or PIL Image.
        
    Returns:
        Dictionary containing:
            - has_exif: bool
            - basic_info: dict (format, size, mode)
            - camera_info: dict (make, model, lens, exposure, iso)
            - date_info: dict (original date, modify date)
            - gps_info: dict (latitude, longitude, altitude)
            - software_traces: list of detected editing or AI software names
            - software_categories: list of categories ('editor', 'generative_ai')
            - anomalies: list of human-readable warnings
            - raw_tags: dict of stringified tag name -> tag value
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

    result: Dict[str, Any] = {
        "has_exif": False,
        "basic_info": {
            "format": pil_img.format or "Unknown",
            "width": pil_img.width,
            "height": pil_img.height,
            "mode": pil_img.mode
        },
        "camera_info": {},
        "date_info": {},
        "gps_info": {},
        "software_traces": [],
        "software_categories": [],
        "anomalies": [],
        "raw_tags": {}
    }

    tag_dict = {}

    # Check for PNG/TIFF text metadata chunks in pil_img.info
    if hasattr(pil_img, "info") and isinstance(pil_img.info, dict):
        for info_k, info_v in pil_img.info.items():
            if isinstance(info_k, str):
                tag_dict[info_k] = str(info_v)

    try:
        exif_data = pil_img.getexif()
    except Exception:
        exif_data = None

    if exif_data and len(exif_data) > 0:
        result["has_exif"] = True
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            tag_dict[tag_name] = value

        # Check for EXIF sub-IFD (detailed photo parameters)
        try:
            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                try:
                    ifd_data = exif_data.get_ifd(ifd_id)
                    for tag_id, value in ifd_data.items():
                        sub_name = (ExifTags.GPSTAGS.get(tag_id, str(tag_id))
                                    if ifd_id == ExifTags.IFD.GPSInfo
                                    else ExifTags.TAGS.get(tag_id, str(tag_id)))
                        tag_dict[sub_name] = value
                except Exception:
                    pass
        except Exception:
            pass
    else:
        result["anomalies"].append(
            "No standard EXIF hardware metadata found. Digital camera images normally retain camera hardware and sensor data. "
            "Absence of EXIF is characteristic of AI generation, web downloads, or intentional scrubbing."
        )

    # Populate stringified raw tags
    for k, v in tag_dict.items():
        try:
            # Filter out giant binary blobs
            str_val = str(v)
            if len(str_val) > 200:
                str_val = str_val[:200] + "... [truncated]"
            result["raw_tags"][str(k)] = str_val
        except Exception:
            pass

    # Extract Camera Info
    make = tag_dict.get("Make")
    model = tag_dict.get("Model")
    lens = tag_dict.get("LensModel") or tag_dict.get("LensInfo")
    exposure = tag_dict.get("ExposureTime")
    fnumber = tag_dict.get("FNumber")
    iso = tag_dict.get("ISOSpeedRatings") or tag_dict.get("PhotographicSensitivity")
    focal_length = tag_dict.get("FocalLength")

    if make or model:
        result["camera_info"]["Make"] = str(make).strip() if make else "Unknown"
        result["camera_info"]["Model"] = str(model).strip() if model else "Unknown"
    if lens:
        result["camera_info"]["Lens"] = str(lens).strip()
    if exposure:
        result["camera_info"]["Exposure"] = str(exposure)
    if fnumber:
        result["camera_info"]["F-Number"] = f"f/{fnumber}"
    if iso:
        result["camera_info"]["ISO"] = str(iso)
    if focal_length:
        result["camera_info"]["Focal Length"] = f"{focal_length}mm"

    # Extract Dates
    date_orig = tag_dict.get("DateTimeOriginal")
    date_digitized = tag_dict.get("DateTimeDigitized")
    date_modified = tag_dict.get("DateTime")

    if date_orig:
        result["date_info"]["Date Original"] = str(date_orig)
    if date_digitized:
        result["date_info"]["Date Digitized"] = str(date_digitized)
    if date_modified:
        result["date_info"]["Date Modified"] = str(date_modified)

    # Check for software traces
    software_val = str(tag_dict.get("Software", "")).lower()
    description_val = str(tag_dict.get("ImageDescription", "")).lower()
    comment_val = str(tag_dict.get("UserComment", "")).lower()
    combined_metadata_text = f"{software_val} {description_val} {comment_val}"

    # Search for known editing tools
    for sig in KNOWN_EDITING_SIGNATURES:
        if sig in combined_metadata_text:
            result["software_traces"].append(sig.title())
            if "editor" not in result["software_categories"]:
                result["software_categories"].append("editor")
            result["anomalies"].append(
                f"Editing software signature detected in metadata: '{sig.title()}'. "
                "Indicates potential post-processing, re-saving, or manipulation."
            )

    # Search for known AI generation tools
    for sig in KNOWN_AI_SIGNATURES:
        if sig in combined_metadata_text:
            result["software_traces"].append(sig.title())
            if "generative_ai" not in result["software_categories"]:
                result["software_categories"].append("generative_ai")
            result["anomalies"].append(
                f"Synthetic / Generative AI signature detected in metadata: '{sig.title()}'."
            )

    # Extract GPS if present
    try:
        gps_lat = tag_dict.get("GPSLatitude")
        gps_lat_ref = tag_dict.get("GPSLatitudeRef")
        gps_lon = tag_dict.get("GPSLongitude")
        gps_lon_ref = tag_dict.get("GPSLongitudeRef")

        if gps_lat and gps_lon:
            lat = _convert_to_degrees(gps_lat)
            lon = _convert_to_degrees(gps_lon)
            if lat is not None and lon is not None:
                if gps_lat_ref and str(gps_lat_ref).upper() == "S":
                    lat = -lat
                if gps_lon_ref and str(gps_lon_ref).upper() == "W":
                    lon = -lon
                result["gps_info"]["Latitude"] = round(lat, 6)
                result["gps_info"]["Longitude"] = round(lon, 6)
    except Exception:
        pass

    # Check for anomalies
    if not result["camera_info"]:
        result["anomalies"].append(
            "EXIF present but lacks standard camera manufacturer and model tags. "
            "Common in synthetic renders or software exports."
        )

    if date_orig and date_modified and date_orig != date_modified:
        result["anomalies"].append(
            f"Timestamp mismatch: Image original creation ({date_orig}) differs from file modification ({date_modified})."
        )

    return result
