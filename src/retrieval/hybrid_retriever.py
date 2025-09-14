# hybrid_retriever.py
import logging
from typing import Dict, List, Any
import os
import asyncio

from src.enhanced_vectorization import VectorizationModule
from src.enhanced_faiss_db_manager import EnhancedFaissVectorDB
from src.gcp_storage_adapter import GCPStorageAdapter
from src.csv_utils.csv_storage_adapter import CSVStorageAdapter

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retrieval system combining docs and CSV outcomes."""
    
    def __init__(self):
        """Initialize hybrid retriever."""
        self.vectorizer = VectorizationModule()
        self.docs_storage = GCPStorageAdapter(
            bucket_name=os.getenv("GCP_BUCKET", "intraintel-cloudrun-clinical-volume"),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account_credentials.json")
        )
        self.csv_storage = CSVStorageAdapter(
            bucket_name=os.getenv("GCP_BUCKET", "intraintel-cloudrun-clinical-volume"),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account_credentials.json")
        )
    
    async def retrieve(
        self,
        query: str,
        docs_model_id: str,
        csv_model_id: str,
        top_k_docs: int = 8,
        top_k_csv: int = 12
    ) -> Dict[str, Any]:
        """
        Perform hybrid retrieval from both docs and CSV indexes.
        
        Args:
            query: User query
            docs_model_id: Model ID for docs index
            csv_model_id: Model ID for CSV index
            top_k_docs: Number of doc results to retrieve
            top_k_csv: Number of CSV results to retrieve
            
        Returns:
            Combined retrieval results
        """
        try:
            # Run both retrievals concurrently
            docs_task = asyncio.create_task(
                self._retrieve_docs(query, docs_model_id, top_k_docs)
            )
            csv_task = asyncio.create_task(
                self._retrieve_csv(query, csv_model_id, top_k_csv)
            )
            
            docs_results, csv_results = await asyncio.gather(docs_task, csv_task)
            
            # Build enriched query based on docs context
            enriched_query = self._build_enriched_query(query, docs_results)
            
            # Re-retrieve CSV with enriched query if different
            if enriched_query != query:
                logger.info("Re-retrieving CSV with enriched query")
                csv_results = await self._retrieve_csv(enriched_query, csv_model_id, top_k_csv)
            
            return {
                "original_query": query,
                "enriched_query": enriched_query,
                "docs_results": docs_results,
                "csv_results": csv_results,
                "total_docs": len(docs_results),
                "total_csv": len(csv_results)
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return {
                "original_query": query,
                "enriched_query": query,
                "docs_results": [],
                "csv_results": [],
                "total_docs": 0,
                "total_csv": 0,
                "error": str(e)
            }
    
    async def _retrieve_docs(self, query: str, model_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from docs index."""
        try:
            # Download docs index
            docs_index_path = f"gcp-indexes/{model_id}"
            if not self.docs_storage.download_index_using_model_id(model_id, docs_index_path):
                logger.error(f"Failed to download docs index: {model_id}")
                return []
            
            # Load docs vector database
            docs_db = EnhancedFaissVectorDB()
            if not docs_db.load(docs_index_path):
                logger.error(f"Failed to load docs index: {docs_index_path}")
                return []
            
            # Get query embedding with matching dimension
            docs_dim = docs_db.get_dimension()
            query_embedding = self.vectorizer.get_query_vector_auto_dim(query, docs_dim)
            
            # Search docs
            results, scores = docs_db.similarity_search(query_embedding, k=top_k)
            
            # Add scores to results
            for i, result in enumerate(results):
                result['similarity_score'] = float(scores[i]) if i < len(scores) else 0.0
                result['source_type'] = 'docs'
            
            logger.info(f"Retrieved {len(results)} docs results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving docs: {e}")
            return []
    
    async def _retrieve_csv(self, query: str, model_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from CSV index."""
        # try:
        #     # Download CSV index
        #     csv_index_path = f"csv-indexes/{model_id}/{model_id}"
        #     if not self.csv_storage.download_csv_index(model_id):
        #         logger.error(f"Failed to download CSV index: {model_id}")
        #         return []
            
        #     # Load CSV vector database
        #     csv_db = EnhancedFaissVectorDB()
        #     if not csv_db.load(csv_index_path):
        #         logger.error(f"Failed to load CSV index: {csv_index_path}")
        #         return []

        try:
            csv_index_path = f"csv-indexes/{model_id}/{model_id}"
            
            # Check if exists locally first
            if not (os.path.exists(f"{csv_index_path}.index") and 
                    (os.path.exists(f"{csv_index_path}.documents") or 
                    os.path.exists(f"{csv_index_path}.documents.gz"))):
                # Download only if not local
                if not self.csv_storage.download_csv_index(model_id):
                    logger.error(f"Failed to download CSV index: {model_id}")
                    return []
            
            # Load CSV vector database
            csv_db = EnhancedFaissVectorDB()
            if not csv_db.load(csv_index_path):
                logger.error(f"Failed to load CSV index: {csv_index_path}")
                return []

            # Get query embedding with matching dimension
            csv_dim = csv_db.get_dimension()
            query_embedding = self.vectorizer.get_query_vector_auto_dim(query, csv_dim)
            
            # Search CSV
            results, scores = csv_db.similarity_search(query_embedding, k=top_k)
            
            # Add scores to results
            for i, result in enumerate(results):
                result['similarity_score'] = float(scores[i]) if i < len(scores) else 0.0
                result['source_type'] = 'csv'
            
            logger.info(f"Retrieved {len(results)} CSV results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving CSV: {e}")
            return []
    
    def _build_enriched_query(self, original_query: str, docs_results: List[Dict[str, Any]]) -> str:
        """
        Build enriched query based on docs context.
        
        Args:
            original_query: Original user query
            docs_results: Results from docs retrieval
            
        Returns:
            Enriched query string
        """
        if not docs_results:
            return original_query
        
        try:
            # Extract key terms from top docs
            enrichment_terms = []
            
            for result in docs_results[:3]:  # Use top 3 docs
                text = result.get('text', '')
                
                # Extract potential clinical terms
                clinical_terms = self._extract_clinical_terms(text)
                enrichment_terms.extend(clinical_terms)
            
            # Remove duplicates and limit
            unique_terms = list(set(enrichment_terms))[:5]
            
            if unique_terms:
                enriched_query = f"{original_query} {' '.join(unique_terms)}"
                logger.info(f"Enriched query with terms: {unique_terms}")
                return enriched_query
            
            return original_query
            
        except Exception as e:
            logger.error(f"Error building enriched query: {e}")
            return original_query
    
    def _extract_clinical_terms(self, text: str) -> List[str]:
        """
        Extract relevant clinical terms from text.
        
        Args:
            text: Input text
            
        Returns:
            List of clinical terms
        """
        # Simple term extraction - can be enhanced with NLP
        clinical_keywords = [
            'efficacy', 'safety', 'dosage', 'treatment', 'therapy', 'intervention',
            'placebo', 'control', 'randomized', 'blinded', 'phase', 'trial',
            'primary endpoint', 'secondary endpoint', 'outcome', 'mortality',
            'adverse events', 'biomarker', 'progression', 'response', 'remission'
        ]
        
        text_lower = text.lower()
        found_terms = []
        
        for keyword in clinical_keywords:
            if keyword in text_lower:
                found_terms.append(keyword)
        
        return found_terms[:3]  # Limit to top 3 terms