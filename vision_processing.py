from __future__ import annotations
from typing import List, Dict

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from config import ARCH_LABELS, get_settings
from sam2_client import SAM2Mask


def filter_masks(masks: List[SAM2Mask], settings=None) -> List[SAM2Mask]:
    settings = settings or get_settings()
    filtered = []
    for m in masks:
        if (
            m.predicted_iou >= settings.min_iou
            and m.stability_score >= settings.min_stability
            and m.area >= settings.min_area_px
        ):
            filtered.append(m)
    return filtered


def crop_from_mask(img: Image.Image, mask: SAM2Mask) -> Image.Image:
    bbox = mask.bbox
    x, y, w, h = bbox
    crop = img.crop((x, y, x + w, y + h))
    mask_img = Image.fromarray((mask.mask_np[y:y+h, x:x+w] * 255).astype('uint8'))
    crop.putalpha(mask_img)
    return crop


def top_colors_from_mask(img: Image.Image, mask: SAM2Mask, k: int = 3) -> List[str]:
    x, y, w, h = mask.bbox
    region = np.array(img.crop((x, y, x + w, y + h)))
    mask_region = mask.mask_np[y:y+h, x:x+w]
    pixels = region[mask_region]
    if len(pixels) == 0:
        return []
    km = KMeans(n_clusters=min(k, len(pixels)))
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)
    return [f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in centers]


def classify_masks(masks: List[SAM2Mask], img: Image.Image, settings=None) -> List[Dict]:
    results = []
    for m in masks:
        colors = top_colors_from_mask(img, m)
        results.append({
            'label': 'object',
            'confidence': 0.5,
            'category': 'unknown',
            'bbox': m.bbox,
            'colors': colors,
            'leafCount': 1,
            'isGlazed': False,
        })
    return results
