from __future__ import annotations
import base64
import io
import os
from typing import Tuple

from PIL import Image, ExifTags

from schema_models import FileData, CameraData


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path).convert('RGB')
    return img


def image_to_b64(img: Image.Image, format: str = 'JPEG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def extract_filedata(img: Image.Image, path: str) -> FileData:
    stat = os.stat(path)
    width, height = img.size
    dpi = img.info.get('dpi', (72, 72))
    return FileData(
        fileSizeBytes=stat.st_size,
        width=width,
        height=height,
        dimensions=f"{width} x {height}",
        aspectRatio=round(width / height, 2) if height else 0,
        horizontalResolutionDPI=int(dpi[0]),
        verticalResolutionDPI=int(dpi[1]),
        bitDepth=8 * len(img.getbands()),
        colorMode=img.mode,
        format=img.format or 'JPEG',
        imageQuality='High',
    )


def extract_cameradata(img: Image.Image) -> CameraData:
    exif = img._getexif() or {}
    exif_data = {ExifTags.TAGS.get(k): v for k, v in exif.items() if k in ExifTags.TAGS}
    time = exif_data.get('DateTime', '')
    camera = exif_data.get('Model', 'Unknown')
    return CameraData(
        position='Terrestrial',
        isProfessionalPhoto=False,
        cameraType='Unknown',
        technique='unknown',
        framing='unknown',
        timeOfDay='Morning',
        lighting='Natural',
    )
