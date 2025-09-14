# csv_processor.py
import pandas as pd # type: ignore
import os
import logging
import tiktoken # type: ignore
from typing import List, Dict, Any, Optional
from tqdm import tqdm # type: ignore
import json

from src.enhanced_vectorization import VectorizationModule
from src.enhanced_faiss_db_manager import EnhancedFaissVectorDB

logger = logging.getLogger(__name__)

class CSVProcessor:
    """Module for processing clinical trials CSV data and creating embeddings."""
    
    def __init__(self):
        """Initialize the CSV processor."""
        self.vectorizer = VectorizationModule()
        self.vector_db = EnhancedFaissVectorDB()
    
    # def process_csv_file(
    #     self,
    #     csv_file_path: str,
    #     model_id: str,
    #     batch_size: int = 256,
    #     embedding_model: str = "text-embedding-3-small",
    #     chunk_size: int = 1000
    # ) -> bool:
    #     """
    #     Process CSV file and create FAISS index.
        
    #     Args:
    #         csv_file_path: Path to the CSV file
    #         model_id: Unique identifier for this index
    #         batch_size: Batch size for embedding generation
    #         embedding_model: OpenAI embedding model to use
    #         chunk_size: Number of rows to process at once
            
    #     Returns:
    #         Success status
    #     """
    #     try:
    #         logger.info(f"Processing CSV file: {csv_file_path}")
            
    #         # Validate file exists
    #         if not os.path.exists(csv_file_path):
    #             raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
            
    #         # Initialize vectorizer with specified model
    #         self.vectorizer = VectorizationModule(embedding_model=embedding_model)
            
    #         # Get total rows for progress tracking
    #         # total_rows = sum(1 for _ in open(csv_file_path)) - 1  # Subtract header
    #         # total_rows = len(pd.read_csv(csv_file_path, dtype=str, keep_default_na=False)) #working
    #         total_rows = 0
    #         for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
    #             try:
    #                 total_rows = len(pd.read_csv(csv_file_path, dtype=str, keep_default_na=False, encoding=encoding))
    #                 break
    #             except UnicodeDecodeError:
    #                 continue
    #         logger.info(f"Total rows to process: {total_rows}")
            
    #         # Process CSV in chunks with encoding handling
    #         processed_count = 0
    #         all_embeddings = []
    #         all_documents = []
            
    #         # Try different encodings
    #         for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
    #             try:
    #                 chunk_iter = pd.read_csv(csv_file_path, chunksize=chunk_size, dtype=str, keep_default_na=False)
    #                 logger.info(f"Successfully opened CSV with encoding: {encoding}")
    #                 break
    #             except UnicodeDecodeError:
    #                 continue
    #         else:
    #             raise ValueError(f"Could not decode CSV file with any common encoding")
            
    #         for chunk_idx, chunk in enumerate(chunk_iter):
    #             logger.info(f"Processing chunk {chunk_idx + 1}")
                
    #             # Convert chunk to text documents
    #             chunk_docs = self._chunk_to_documents(chunk)
                
    #             if not chunk_docs:
    #                 continue
                
    #             # Create embeddings for this chunk
    #             chunk_texts = [doc['text'] for doc in chunk_docs]
    #             embeddings = self.vectorizer.get_batch_embeddings(
    #                 chunk_texts, 
    #                 batch_size=batch_size
    #             )
                
    #             # Add embeddings to documents
    #             for doc, embedding in zip(chunk_docs, embeddings):
    #                 doc['embedding'] = embedding
                
    #             all_embeddings.extend(embeddings)
    #             all_documents.extend(chunk_docs)
    #             processed_count += len(chunk_docs)
                
    #             logger.info(f"Processed {processed_count}/{total_rows} rows")
            
    #         # Create FAISS index
    #         logger.info("Creating FAISS index...")
    #         success = self.vector_db.add_documents(all_documents)
    #         if not success:
    #             raise RuntimeError("Failed to create FAISS index")
            
    #         # Save index
    #         index_path = f"csv-indexes/{model_id}/{model_id}"
    #         os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
    #         if not self.vector_db.save(index_path):
    #             raise RuntimeError("Failed to save FAISS index")
            
    #         logger.info(f"Successfully processed {processed_count} rows and created index: {index_path}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error processing CSV file: {e}")
    #         return False

    # def process_csv_file( #fails on large amounts of data
    # self,
    # csv_file_path: str,
    # model_id: str,
    # batch_size: int = 256,
    # embedding_model: str = "text-embedding-3-small",
    # chunk_size: int = 1000
    # ) -> bool:
    #     """Process CSV file and create FAISS index with encoding handling."""
    #     try:
    #         logger.info(f"Processing CSV file: {csv_file_path}")
            
    #         if not os.path.exists(csv_file_path):
    #             raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
            
    #         self.vectorizer = VectorizationModule(embedding_model=embedding_model)
            
    #         # Try different encodings for row counting
    #         total_rows = 0
    #         detected_encoding = 'utf-8'
    #         for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
    #             try:
    #                 total_rows = len(pd.read_csv(csv_file_path, dtype=str, keep_default_na=False, encoding=encoding))
    #                 detected_encoding = encoding
    #                 logger.info(f"Successfully opened CSV with encoding: {encoding}")
    #                 break
    #             except UnicodeDecodeError:
    #                 continue
    #         else:
    #             raise ValueError("Could not decode CSV file with any common encoding")
            
    #         logger.info(f"Total rows to process: {total_rows}")
            
    #         # Process CSV in chunks
    #         processed_count = 0
    #         all_documents = []
            
    #         chunk_iter = pd.read_csv(
    #             csv_file_path, 
    #             chunksize=chunk_size, 
    #             dtype=str, 
    #             keep_default_na=False, 
    #             encoding=detected_encoding
    #         )
            
    #         for chunk_idx, chunk in enumerate(chunk_iter):
    #             logger.info(f"Processing chunk {chunk_idx + 1}")
                
    #             chunk_docs = self._chunk_to_documents(chunk)
    #             if not chunk_docs:
    #                 continue
                
    #             chunk_texts = [doc['text'] for doc in chunk_docs]
    #             embeddings = self.vectorizer.get_batch_embeddings(chunk_texts, batch_size=batch_size)
                
    #             for doc, embedding in zip(chunk_docs, embeddings):
    #                 doc['embedding'] = embedding
                
    #             all_documents.extend(chunk_docs)
    #             processed_count += len(chunk_docs)
                
    #             logger.info(f"Processed {processed_count} documents")
            
    #         # Create FAISS index
    #         logger.info("Creating FAISS index...")
    #         success = self.vector_db.add_documents(all_documents)
    #         if not success:
    #             raise RuntimeError("Failed to create FAISS index")
            
    #         # Save index
    #         index_path = f"csv-indexes/{model_id}/{model_id}"
    #         os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
    #         if not self.vector_db.save(index_path):
    #             raise RuntimeError("Failed to save FAISS index")
            
    #         logger.info(f"Successfully processed {processed_count} documents and created index: {index_path}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error processing CSV file: {e}")
    #         return False


    def process_csv_file(
    self,
    csv_file_path: str,
    model_id: str,
    batch_size: int = 256,
    embedding_model: str = "text-embedding-3-large",
    chunk_size: int = 1000,
    faiss_batch_size: int = 5000  # Process 5k docs at a time for FAISS
) -> bool:
        """Process CSV file with incremental FAISS index building."""
        try:
            logger.info(f"Processing CSV file: {csv_file_path}")
            
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
            
            self.vectorizer = VectorizationModule(embedding_model=embedding_model)
            
            # Detect encoding
            detected_encoding = 'utf-8'
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    total_rows = len(pd.read_csv(csv_file_path, dtype=str, keep_default_na=False, encoding=encoding))
                    detected_encoding = encoding
                    logger.info(f"Using encoding: {encoding}, {total_rows} rows")
                    break
                except UnicodeDecodeError:
                    continue
            
            # Initialize FAISS DB early
            index_path = f"csv-indexes/{model_id}/{model_id}"
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            processed_count = 0
            total_docs = 0
            
            # Process in chunks
            chunk_iter = pd.read_csv(
                csv_file_path, 
                chunksize=chunk_size, 
                dtype=str, 
                keep_default_na=False, 
                encoding=detected_encoding
            )
            
            batch_docs = []
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                logger.info(f"Processing chunk {chunk_idx + 1}")
                
                chunk_docs = self._chunk_to_documents(chunk)
                if not chunk_docs:
                    continue
                
                # Add to batch
                batch_docs.extend(chunk_docs)
                
                # Process batch when it reaches faiss_batch_size
                if len(batch_docs) >= faiss_batch_size:
                    success = self._process_and_add_batch(batch_docs, batch_size)
                    if not success:
                        raise RuntimeError("Failed to process batch")
                    
                    processed_count += len(batch_docs)
                    total_docs += len(batch_docs)
                    logger.info(f"Added {len(batch_docs)} docs to FAISS. Total: {total_docs}")
                    
                    # Save checkpoint every 50k docs
                    if total_docs % 50000 == 0:
                        self.vector_db.save(index_path)
                        logger.info(f"Checkpoint saved at {total_docs} docs")
                    
                    batch_docs = []
            
            # Process remaining docs
            if batch_docs:
                success = self._process_and_add_batch(batch_docs, batch_size)
                if not success:
                    raise RuntimeError("Failed to process final batch")
                total_docs += len(batch_docs)
            
            # Final save
            if not self.vector_db.save(index_path):
                raise RuntimeError("Failed to save final FAISS index")
            
            logger.info(f"Successfully processed {total_docs} documents and created index: {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            return False

    def _process_and_add_batch(self, docs: List[Dict[str, Any]], batch_size: int) -> bool:
        """Process embeddings and add to FAISS in memory-efficient way."""
        try:
            # Get embeddings
            texts = [doc['text'] for doc in docs]
            embeddings = self.vectorizer.get_batch_embeddings(texts, batch_size=batch_size)
            
            # Add embeddings to docs
            embedded_docs = []
            for doc, embedding in zip(docs, embeddings):
                doc_copy = doc.copy()
                doc_copy['embedding'] = embedding
                embedded_docs.append(doc_copy)
            
            # Add to FAISS
            success = self.vector_db.add_documents(embedded_docs)
            
            # Clear memory
            del embedded_docs, embeddings, texts
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return False
    
    # def _chunk_to_documents(self, chunk: pd.DataFrame) -> List[Dict[str, Any]]:
    #     """
    #     Convert DataFrame chunk to document format.
        
    #     Args:
    #         chunk: DataFrame chunk
            
    #     Returns:
    #         List of document dictionaries
    #     """
    #     documents = []
        
    #     for idx, row in chunk.iterrows():
    #         try:
    #             # Extract key columns (already strings from dtype=str)
    #             nct_number = str(row.get('NCT Number', '')).strip()
    #             primary_outcomes = str(row.get('Primary Outcome Measures', '')).strip()
    #             secondary_outcomes = str(row.get('Secondary Outcome Measures', '')).strip()
    #             other_outcomes = str(row.get('Other Outcome Measures', '')).strip()
                
    #             # Skip rows with missing NCT number or empty outcomes
    #             if not nct_number or nct_number in ['nan', '', 'NaN']:
    #                 continue
                
    #             # Build text representation (skip empty fields)
    #             text_parts = [f"NCT Number: {nct_number}"]
                
    #             if primary_outcomes and primary_outcomes not in ['nan', '', 'NaN']:
    #                 text_parts.append(f"Primary Outcomes: {primary_outcomes}")
                
    #             if secondary_outcomes and secondary_outcomes not in ['nan', '', 'NaN']:
    #                 text_parts.append(f"Secondary Outcomes: {secondary_outcomes}")
                
    #             if other_outcomes and other_outcomes not in ['nan', '', 'NaN']:
    #                 text_parts.append(f"Other Outcomes: {other_outcomes}")
                
    #             # Skip if only NCT number (no outcomes)
    #             if len(text_parts) == 1:
    #                 continue
                
    #             text = "\n".join(text_parts)
                
    #             # Create document
    #             document = {
    #                 'text': text,
    #                 'nct_number': nct_number,
    #                 'primary_outcomes': primary_outcomes,
    #                 'secondary_outcomes': secondary_outcomes,
    #                 'other_outcomes': other_outcomes,
    #                 'source': 'clinical_trials_csv',
    #                 'row_index': idx
    #             }
                
    #             documents.append(document)
                
    #         except Exception as e:
    #             logger.warning(f"Error processing row {idx}: {e}")
    #             continue
        
    #     return documents

    # def _chunk_to_documents(self, chunk: pd.DataFrame) -> List[Dict[str, Any]]: # working baseline with skipping mechaniusm
    #     """Convert DataFrame chunk to document format with length checking."""

    #     enc = tiktoken.get_encoding("cl100k_base")
    #     documents = []
        
    #     for idx, row in chunk.iterrows():
    #         try:
    #             nct_number = str(row.get('NCT Number', '')).strip()
    #             primary_outcomes = str(row.get('Primary Outcome Measures', '')).strip()
    #             secondary_outcomes = str(row.get('Secondary Outcome Measures', '')).strip()
    #             other_outcomes = str(row.get('Other Outcome Measures', '')).strip()
                
    #             if not nct_number or nct_number in ['nan', '', 'NaN']:
    #                 continue
                
    #             outcome_data = [
    #                 ('primary', primary_outcomes),
    #                 ('secondary', secondary_outcomes), 
    #                 ('other', other_outcomes)
    #             ]
                
    #             for outcome_type, content in outcome_data:
    #                 if content and content not in ['nan', '', 'NaN']:
    #                     text = f"NCT Number: {nct_number}\n{outcome_type.title()} Outcomes: {content}"
                        
    #                     # Skip if too long (>6000 tokens to be safe)
    #                     if len(enc.encode(text)) > 6000:
    #                         logger.warning(f"Skipping {nct_number} {outcome_type} - too long")
    #                         continue
                        
    #                     document = {
    #                         'text': text,
    #                         'nct_number': nct_number,
    #                         'outcome_type': outcome_type,
    #                         'outcome_content': content,
    #                         'source': 'clinical_trials_csv',
    #                         'row_index': idx
    #                     }
                        
    #                     documents.append(document)
                
    #         except Exception as e:
    #             logger.warning(f"Error processing row {idx}: {e}")
    #             continue
        
    #     return documents

    def _chunk_to_documents(self, chunk: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame chunk with contextual splitting for long texts."""
        enc = tiktoken.get_encoding("cl100k_base")
        documents = []
        
        # def split_long_text(text: str, max_tokens: int = 6000) -> List[str]:
        #     """Split text by sentences, keeping under token limit."""
        #     if len(enc.encode(text)) <= max_tokens:
        #         return [text]
            
        #     # Split by common delimiters
        #     sentences = []
        #     for delimiter in ['. ', '| ', '; ', '\n']:
        #         if delimiter in text:
        #             sentences = text.split(delimiter)
        #             break
        #     else:
        #         # Fallback: split by spaces
        #         words = text.split(' ')
        #         sentences = [' '.join(words[i:i+100]) for i in range(0, len(words), 100)]
            
        #     # Recombine staying under token limit
        #     chunks = []
        #     current = []
        #     current_tokens = 0
            
        #     for sentence in sentences:
        #         sentence_tokens = len(enc.encode(sentence))
        #         if current_tokens + sentence_tokens > max_tokens and current:
        #             chunks.append('. '.join(current))
        #             current = [sentence]
        #             current_tokens = sentence_tokens
        #         else:
        #             current.append(sentence)
        #             current_tokens += sentence_tokens
            
        #     if current:
        #         chunks.append('. '.join(current))
            
        #     return chunks

        def split_long_text(text: str, max_tokens: int = 5000) -> List[str]:
            """Aggressively split text with hard token limits."""
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return [text]
            
            # Split into smaller token chunks
            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i+max_tokens]
                chunk_text = enc.decode(chunk_tokens)
                chunks.append(chunk_text)
            
            return chunks
        
        for idx, row in chunk.iterrows():
            try:
                nct_number = str(row.get('NCT Number', '')).strip()
                if not nct_number or nct_number in ['nan', '', 'NaN']:
                    continue
                
                outcome_data = [
                    ('primary', str(row.get('Primary Outcome Measures', '')).strip()),
                    ('secondary', str(row.get('Secondary Outcome Measures', '')).strip()),
                    ('other', str(row.get('Other Outcome Measures', '')).strip())
                ]
                
                for outcome_type, content in outcome_data:
                    if content and content not in ['nan', '', 'NaN']:
                        content_chunks = split_long_text(content)
                        
                        for i, chunk_content in enumerate(content_chunks):
                            text = f"NCT Number: {nct_number}\n{outcome_type.title()} Outcomes: {chunk_content}"
                            
                            document = {
                                'text': text,
                                'nct_number': nct_number,
                                'outcome_type': outcome_type,
                                'outcome_content': chunk_content,
                                'chunk_index': i,
                                'source': 'clinical_trials_csv',
                                'row_index': idx
                            }
                            
                            documents.append(document)
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        return documents