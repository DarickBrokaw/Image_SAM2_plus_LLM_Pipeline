from __future__ import annotations
import json
import requests
from dataclasses import dataclass
from typing import List
import numpy as np

from config import get_settings
from image_io import image_to_b64


@dataclass
class SAM2Mask:
    mask_np: np.ndarray
    bbox: List[int]
    predicted_iou: float
    stability_score: float
    area: int


def call_sam2_everything(img, settings=None) -> List[SAM2Mask]:
    settings = settings or get_settings()
    payload = {
        'inputs': image_to_b64(img, format='PNG'),
        'parameters': {'mode': 'everything'}
    }
    headers = {
        'Authorization': f'Bearer {settings.hf_token}',
        'Content-Type': 'application/json'
    }
    resp = requests.post(settings.sam2_endpoint_url, headers=headers, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    masks = []
    for m in data.get('masks', []):
        mask_array = np.array(m['segmentation']).astype(bool)
        bbox = m.get('bbox')
        iou = m.get('predicted_iou', 0.0)
        stability = m.get('stability_score', 0.0)
        area = int(mask_array.sum())
        if not bbox:
            ys, xs = np.where(mask_array)
            if ys.size and xs.size:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())]
            else:
                bbox = [0, 0, 0, 0]
        masks.append(SAM2Mask(mask_array, bbox, iou, stability, area))
    return masks
