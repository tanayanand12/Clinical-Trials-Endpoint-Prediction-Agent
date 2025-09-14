# Endpoint Prediction Agent v2

This project provides a FastAPI-based API for processing medical research papers, clinical trials data, and predicting endpoint timelines using hybrid retrieval and vector search.

## Features
- Upload and index PDFs for semantic search
- Index and query clinical trials CSV data
- Predict clinical trial endpoints using hybrid retrieval
- GCP integration for index storage
- Dockerized deployment

## Quickstart

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd endpoint_prediction_agent_v2
```

### 2. Set up environment variables
Copy `.env.example` to `.env` and fill in your values:
```sh
cp .env.example .env
```

### 3. Build and run with Docker
```sh
docker build -t endpoint-prediction-agent .
docker run -p 8359:8359 --env-file .env endpoint-prediction-agent
```

### 4. API Usage
- The API will be available at `http://localhost:8359`
- See `/docs` for interactive Swagger UI

## Development
- Python 3.11+
- Install dependencies: `pip install -r requirements.txt`
- Run locally: `python -m src.enhanced_api_module`

## File Structure
- `src/` - Main source code
- `scripts/` - Utility scripts
- `indexes/`, `gcp-indexes/`, `csv-indexes/` - Index storage
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker build instructions

## Ignore Patterns
- `.gitignore` excludes `.env`, `.csv`, and other sensitive or large files

## License
MIT
