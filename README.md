
# Endpoint Prediction Agent

A powerful, production-ready FastAPI application for hybrid retrieval, semantic search, and endpoint prediction on clinical trial data and research documents.  
**Supports PDF and CSV ingestion, GCP integration, and LLM-powered endpoint prediction.**

---

## 🌟 Features

- **Semantic Search**: Upload and index PDFs for advanced semantic search.
- **CSV Indexing**: Index and query clinical trial CSV data with embeddings.
- **Hybrid Retrieval**: Combine document and tabular data for robust answers.
- **Endpoint Prediction**: Predict clinical trial endpoints and timelines using LLM orchestration.
- **Google Cloud Integration**: Store and retrieve indexes from GCP buckets.
- **Dockerized**: Easy deployment with Docker.
- **Modular & Extensible**: Clean architecture for research and production.

---

## 🏗️ Architecture Overview

```
endpoint-prediction-agent/
├── src/
│   ├── enhanced_api_module.py      # FastAPI server, all endpoints
│   ├── enhanced_vectorization.py   # Embedding logic
│   ├── enhanced_faiss_db_manager.py# FAISS vector DB management
│   ├── rag_module.py               # RAG answer generation
│   ├── gcp_storage_adapter.py      # GCP storage integration
│   ├── csv_utils/                  # CSV processing and storage
│   ├── retrieval/                  # Hybrid retriever, timepoint parser
│   └── prediction/                 # Endpoint prediction logic
├── scripts/                        # CLI utilities
├── indexes/, csv-indexes/, gcp-indexes/ # Local & GCP index storage
├── data/                           # Sample data
├── service_account_credentials.json# GCP service account (see below)
├── .env.example                    # Environment variable template
├── Dockerfile                      # Containerization
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quickstart

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/endpoint-prediction-agent.git
cd endpoint-prediction-agent
```

### 2. Set Up Environment Variables

Copy the example file and fill in your values:

```sh
cp .env.example .env
```

**Required variables:**
- `GCP_BUCKET`: Your Google Cloud Storage bucket name.
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your GCP service account JSON (default: `service_account_credentials.json`).

### 3. Add GCP Service Account Credentials

- Download your GCP service account key as JSON.
- Place it in the project root as `service_account_credentials.json`.
- **Never commit this file to public repositories!**

### 4. Install Python Dependencies

```sh
pip install -r requirements.txt
```

### 5. Run the API Locally

```sh
python -m uvicorn src.enhanced_api_module:app --host 0.0.0.0 --port 8000
```

Or use Docker:

```sh
docker build -t endpoint-prediction-agent .
docker run -p 8000:8000 --env-file .env -v $(pwd):/app endpoint-prediction-agent
```

---

## 🔑 Credentials & Environment

- **.env**: Store all secrets and configuration here. See `.env.example` for required keys.
- **service_account_credentials.json**: GCP service account for accessing Google Cloud Storage.  
	- Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to its path.
- **GCP Bucket**: Must exist and be accessible by the service account.

---

## 🧩 API Endpoints

- `POST /upload`: Upload and index PDFs.
- `POST /index_csv`: Index a CSV file for tabular search.
- `POST /query`: Query indexed documents.
- `POST /query_csv`: Query indexed CSV data.
- `POST /predict`: Predict clinical trial endpoints using hybrid retrieval.
- `GET /indexes`, `GET /csv_indexes`: List available indexes.
- `GET /healthz`: Health check.

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🗂️ Data & Index Storage

- **Local**: `indexes/`, `csv-indexes/`, `gcp-indexes/`
- **GCP**:  
	```
	gs://<bucket>/
		├── indexes/<model_id>/
		└── csv-indexes/<model_id>/
	```
- **Sample Data**: Provided in `data/` and `sample_data_seven_percent.csv`

---

## 🛠️ Development

- Python 3.10+
- All code in `src/`
- Utility scripts in `scripts/`
- Tests in `tests/`
- Modular design for easy extension

---

## 📝 Example .env

```env
GCP_BUCKET=your-gcp-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=service_account_credentials.json
# Add other environment variables as needed
```

---

## 🧪 Testing

- Run unit tests (if available) with your preferred test runner, e.g.:
	```sh
	pytest tests/
	```

---

## 🏢 Deployment

- **Docker**: See above for build/run instructions.
- **Cloud**: Deploy on GCP, AWS, Azure, or any platform supporting Docker.
- **Environment**: Always set up `.env` and credentials on your deployment target.

---

## 📚 Documentation

- See `architecture.txt` for a high-level overview.
- API documentation available at `/docs` when running the server.
- For advanced usage, see scripts and source code comments.

---

## ⚠️ Security & Best Practices

- **Never commit secrets**: Add `.env` and `service_account_credentials.json` to `.gitignore`.
- **Rotate credentials** regularly.
- **Restrict GCP service account** permissions to only required resources.

---

## 🤝 Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

---

## 📄 License

MIT License

---

**Beautiful, robust, and ready for research or production.**  
*Built with ❤️ for clinical trial innovation.*

---
