# Image SAM2 + LLM Pipeline

This project demonstrates a proof-of-concept pipeline combining Segment Anything Model 2 (SAM2) with LLM-powered analysis. The service exposes a `/analyze-image` endpoint that returns detailed JSON describing an exterior architectural image.

## Setup

Requirements:

- Python 3.8+
- `pip install -r requirements.txt`

Environment variables:

- `HF_TOKEN` – Hugging Face token for SAM2 endpoint
- `SAM2_ENDPOINT_URL` – URL of SAM2 everything mode endpoint (default provided)
- `OPENAI_API_KEY` – OpenAI API key

## Usage

Run the FastAPI server:

```bash
uvicorn orchestrator_service:app --reload
```

Or run analysis via CLI:

```bash
python orchestrator_service.py /path/to/image.jpg
```

The script prints the analysis JSON and also saves it to your Desktop in a file
named `analysis_<image-name>.txt`.

The API returns JSON conforming to the schema defined in `schema_models.py`.
