from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import openai

from config import get_settings
from image_io import load_image, extract_filedata, extract_cameradata
from sam2_client import call_sam2_everything
from vision_processing import filter_masks, classify_masks
from schema_models import ImageSchema

app = FastAPI()

class AnalyzeRequest(BaseModel):
    imagePath: str

@app.post('/analyze-image')
def analyze_image_endpoint(req: AnalyzeRequest):
    settings = get_settings()
    openai.api_key = settings.openai_api_key
    img = load_image(req.imagePath)
    filedata = extract_filedata(img, req.imagePath)
    cameradata = extract_cameradata(img)
    masks = call_sam2_everything(img, settings)
    masks = filter_masks(masks, settings)
    classifications = classify_masks(masks, img, settings)

    schema = ImageSchema.empty_schema()
    schema.fileData = filedata
    schema.cameraData = cameradata
    schema.tags = []
    schema.colors = []
    schema.doorDetails = []
    schema.balconyDetails = []
    schema.roofDescription = ''
    schema.description = ''

    # Simple OpenAI call for description (placeholder)
    prompt = 'Describe the exterior architectural image.'
    resp = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}]
    )
    schema.description = resp.choices[0].message['content']

    return schema.to_json_dict()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    args = parser.parse_args()
    req = AnalyzeRequest(imagePath=args.image_path)
    result = analyze_image_endpoint(req)
    print(json.dumps(result, indent=2))

    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    output_path = desktop / f"analysis_{Path(args.image_path).stem}.txt"
    with output_path.open('w') as f:
        f.write(json.dumps(result, indent=2))
    print(f"Saved analysis to {output_path}")
