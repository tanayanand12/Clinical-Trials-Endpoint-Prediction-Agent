# enhanced_api_module.py
from fastapi import FastAPI, UploadFile, File, HTTPException #type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel, Field # type: ignore
from typing import List, Dict, Any, Optional
import os
import shutil
import uvicorn # type: ignore
import logging
from datetime import datetime
import traceback
from dotenv import load_dotenv # type: ignore
load_dotenv()

# Import existing modules
from .pdf_processor import PDFProcessor
from .enhanced_vectorization import VectorizationModule
from .enhanced_faiss_db_manager import EnhancedFaissVectorDB
from .rag_module import RAGModule
from .gcp_storage_adapter import GCPStorageAdapter

# Import new modules for endpoint prediction
from src.csv_utils.csv_processor import CSVProcessor
from src.csv_utils.csv_storage_adapter import CSVStorageAdapter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.prediction.endpoint_predictor import EndpointPredictor
from src.retrieval.timepoint_parser import TimepointParser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trials RAG & Endpoint Prediction API",
    description="API for processing medical research papers, clinical trials data, and predicting endpoint timelines",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
vectorizer = VectorizationModule()
vector_db = EnhancedFaissVectorDB()
rag = RAGModule()

# Initialize new components for endpoint prediction
csv_processor = CSVProcessor()
csv_storage = CSVStorageAdapter(
    bucket_name=os.getenv("GCP_BUCKET", "intraintel-cloudrun-clinical-volume"),
    credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account_credentials.json")
)
hybrid_retriever = HybridRetriever()
endpoint_predictor = EndpointPredictor()
timepoint_parser = TimepointParser()

# Existing request/response models
class QueryRequest(BaseModel):
    query: str
    model_id: str
    top_k: int = 5

class GenerateEmbeddingsRequest(BaseModel):
    model_id: str
    max_chars: int = 3000
    overlap: float = 0.2

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]

# New request/response models for endpoint prediction
class CSVIndexRequest(BaseModel):
    csv_file_path: str
    model_id: str
    batch_size: int = 256
    embedding_model: str = "text-embedding-3-large"

class CSVQueryRequest(BaseModel):
    query: str
    model_id: str
    top_k: int = 10

class EndpointPredictionRequest(BaseModel):
    query: str
    csv_model_id: str
    docs_model_id: str
    top_k_docs: int = Field(default=8, ge=1, le=20)
    top_k_csv: int = Field(default=12, ge=1, le=30)
    return_trials: int = Field(default=6, ge=1, le=15)

class EndpointPredictionResponse(BaseModel):
    primary_endpoint: Optional[str] = None
    secondary_endpoint: Optional[str] = None
    similarity: List[Dict[str, Any]] = Field(default_factory=list)

    # predicted_primary_time_days: Optional[int]
    # predicted_secondary_time_days: Optional[int]
    # time_window_days: Optional[int]
    # rationale: str
    # supporting_trials: List[str]  # Changed from Dict to List[str] for NCT numbers
    # confidence_score: Optional[float]

# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# === EXISTING ENDPOINTS (PRESERVED) ===

@app.post("/generate_embeddings")
async def generate_embeddings(request: GenerateEmbeddingsRequest):
    """Generate embeddings for a given model ID."""
    try:
        from .pipeline import RAGPipeline
        pipeline = RAGPipeline()

        success = pipeline.process_pdfs(
            request.model_id,
            request.model_id,
            request.max_chars,
            request.overlap
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")

        return {"message": f"Embeddings generated successfully for model ID: {request.model_id}"}

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    index_name: str = "default"
):
    """Upload PDFs and create searchable index."""
    try:
        # Create temporary directory for PDFs
        os.makedirs("temp_pdfs", exist_ok=True)
        
        # Save uploaded files
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are accepted")
            
            file_path = f"temp_pdfs/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Process PDFs
        chunks = pdf_processor.process_folder("temp_pdfs")
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content extracted from PDFs")
        
        # Create embeddings
        embedded_chunks = vectorizer.embed_chunks(chunks)
        if not embedded_chunks:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        
        # Add to vector database
        if not vector_db.add_documents(embedded_chunks):
            raise HTTPException(status_code=500, detail="Failed to add documents to vector database")
        
        # Save vector database
        os.makedirs("indexes", exist_ok=True)
        if not vector_db.save(f"indexes/{index_name}"):
            raise HTTPException(status_code=500, detail="Failed to save vector database")
        
        # Upload to GCS
        try:
            gcp_storage = GCPStorageAdapter(
                bucket_name="intraintel-cloudrun-clinical-volume",
                credentials_path="service_account_credentials.json"
            )
            gcp_storage.upload_index_to_model_id(index_name, f"indexes/{index_name}")
            logger.info(f"Docs index uploaded to GCS for model ID: {index_name}")
        except Exception as upload_error:
            logger.warning(f"Failed to upload docs to GCS: {upload_error}")
        
        # Cleanup
        shutil.rmtree("temp_pdfs")
        
        return {"message": f"Successfully processed {len(files)} files and created index '{index_name}'"}
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """Query the RAG system for answers."""
    try:
        gcp_storage = GCPStorageAdapter(
            bucket_name="intraintel-cloudrun-clinical-volume",    
            credentials_path="service_account_credentials.json"
        )

        logger.info(f"Downloading index for model ID: {request.model_id}")
        index_path = os.path.join("gcp-indexes", request.model_id)
        status = gcp_storage.download_index_using_model_id(
            model_id=request.model_id,
            local_path=index_path
        )
        if not status:
            raise ValueError(f"Failed to download index: {request.model_id}")
        logger.info(f"Index downloaded to {index_path}")

        # Load vector database
        if not vector_db.load(f"gcp-indexes/{request.model_id}"):
            raise HTTPException(status_code=404, detail=f"Index '{request.model_id}' not found")

        # Create query embedding
        query_embedding = vectorizer.embed_query(request.query)
        
        # Search for relevant documents
        results, scores = vector_db.similarity_search(
            query_embedding,
            k=request.top_k
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Convert to LangChain documents
        documents = vector_db.get_langchain_documents(results)
        
        # Generate answer
        response = rag.generate_answer(request.query, documents)
        
        return {
            "answer": response["answer"],
            "citations": response["citations"]
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexes")
async def list_indexes():
    """List available indexes."""
    try:
        indexes = []
        if os.path.exists("indexes"):
            for file in os.listdir("indexes"):
                if file.endswith(".index"):
                    indexes.append(file[:-6])  # Remove .index extension
        return {"indexes": indexes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/indexes/{index_name}")
async def delete_index(index_name: str):
    """Delete an index."""
    try:
        index_path = f"indexes/{index_name}"
        if os.path.exists(f"{index_path}.index"):
            os.remove(f"{index_path}.index")
            os.remove(f"{index_path}.documents")
            return {"message": f"Index '{index_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === NEW ENDPOINTS FOR ENDPOINT PREDICTION ===

@app.post("/index_csv")
async def index_csv(request: CSVIndexRequest):
    """Create embeddings index from clinical trials CSV data."""
    try:
        logger.info(f"Starting CSV indexing for model ID: {request.model_id}")
        
        # Process CSV file
        success = csv_processor.process_csv_file(
            csv_file_path=request.csv_file_path,
            model_id=request.model_id,
            batch_size=request.batch_size,
            embedding_model=request.embedding_model
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process CSV file")
        
        # Upload to GCS if configured
        try:
            csv_storage.upload_csv_index(request.model_id)
            logger.info(f"CSV index uploaded to GCS for model ID: {request.model_id}")
        except Exception as upload_error:
            logger.warning(f"Failed to upload to GCS: {upload_error}")
        
        return {
            "message": f"CSV index created successfully for model ID: {request.model_id}",
            "embedding_model": request.embedding_model
        }
        
    except Exception as e:
        logger.error(f"Error indexing CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_csv")
async def query_csv(request: CSVQueryRequest):
    """Query CSV index directly for debugging/inspection."""
    try:
        # Download CSV index from GCS
        csv_storage.download_csv_index(request.model_id)
        
        # Load CSV vector database
        csv_vector_db = EnhancedFaissVectorDB()
        csv_index_path = f"csv-indexes/{request.model_id}/{request.model_id}"
        
        if not csv_vector_db.load(csv_index_path):
            raise HTTPException(status_code=404, detail=f"CSV index '{request.model_id}' not found")
        
        # Create query embedding with auto-dimension detection
        index_dim = csv_vector_db.get_dimension()
        query_embedding = vectorizer.get_query_vector_auto_dim(request.query, index_dim)
        
        # Search CSV database
        results, scores = csv_vector_db.similarity_search(
            query_embedding,
            k=request.top_k
        )
        
        return {
            "results": results,
            "scores": scores,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error querying CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=EndpointPredictionResponse)
async def predict_endpoints(request: EndpointPredictionRequest):
    """Full pipeline for endpoint prediction using hybrid retrieval."""
    try:
        logger.info(f"Starting endpoint prediction for query: {request.query[:100]}...")
        
        # Step 1: Hybrid retrieval (docs + CSV)
        retrieval_results = await hybrid_retriever.retrieve(
            query=request.query,
            docs_model_id=request.docs_model_id,
            csv_model_id=request.csv_model_id,
            top_k_docs=request.top_k_docs,
            top_k_csv=request.top_k_csv
        )
        
        # Step 2: Parse timepoints from outcomes
        parsed_timepoints = timepoint_parser.parse_multiple_outcomes(
            retrieval_results["csv_results"]
        )
        
        # Step 3: Prepare evidence for LLM
        evidence = endpoint_predictor.prepare_evidence(
            docs_results=retrieval_results["docs_results"],
            csv_results=retrieval_results["csv_results"],
            parsed_timepoints=parsed_timepoints,
            max_trials=request.return_trials
        )
        
        # Step 4: Generate prediction using LLM
        prediction = await endpoint_predictor.predict_endpoint_timing(
            query=request.query,
            evidence=evidence
        )

        mapped_prediction = {
            "primary_endpoint": prediction.get("primary_endpoint", ""),
            "secondary_endpoint": (
                prediction.get("secondary_endpoint", [""])[0]
                if isinstance(prediction.get("secondary_endpoint"), list)
                else prediction.get("secondary_endpoint", "")
            ),
            "similarity": prediction.get("similarity", []),
        }
        
        return EndpointPredictionResponse(**mapped_prediction)
        
    except Exception as e:
        logger.error(f"Error in endpoint prediction: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/csv_indexes")
async def list_csv_indexes():
    """List available CSV indexes."""
    try:
        indexes = []
        csv_root = "csv-indexes"
        if os.path.exists(csv_root):
            for model_dir in os.listdir(csv_root):
                model_path = os.path.join(csv_root, model_dir)
                if os.path.isdir(model_path):
                    index_file = os.path.join(model_path, f"{model_dir}.index")
                    if os.path.exists(index_file):
                        indexes.append(model_dir)
        return {"csv_indexes": indexes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8359)