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

The API returns JSON conforming to the schema defined in `schema_models.py`.

## Pipeline overview

1. **Segmentation** – the image is sent to the configured SAM2 endpoint in
   "everything" mode to obtain candidate masks.
2. **Filtering** – masks are filtered by IOU, stability, and area thresholds
   defined in `config.py` (overridable via environment variables).
3. **Classification & attributes** – each remaining mask is analyzed to derive
   coarse labels, dominant colors and simple geometric properties.
4. **LLM enrichment** – OpenAI is queried with the structured data to produce a
   narrative description and any missing attributes.

## Example request

Start the server as shown above and send a request using `curl`:

```bash
curl -X POST http://localhost:8000/analyze-image \
  -H 'Content-Type: application/json' \
  -d '{"imagePath": "/path/to/photo.jpg"}'
```

The response is a JSON object following the specification. See
`schema_models.py` for field definitions. If inappropriate content is detected
the response will contain `{ "contentFlagged": true }`.
