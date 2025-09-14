# csv_storage_adapter.py
import os
import logging
from typing import Optional
from google.cloud import storage
from src.gcp_storage_adapter import GCPStorageAdapter

logger = logging.getLogger(__name__)

class CSVStorageAdapter(GCPStorageAdapter):
    """Specialized adapter for CSV indexes in Google Cloud Storage."""
    
    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        """
        Initialize CSV storage adapter.
        
        Args:
            bucket_name: GCS bucket name
            credentials_path: Path to service account credentials
        """
        super().__init__(bucket_name, credentials_path)
        self.csv_prefix = "csv-indexes"
    
    def upload_csv_index(self, model_id: str) -> bool:
        """
        Upload CSV index files to GCS.
        
        Args:
            model_id: Model ID for the CSV index
            
        Returns:
            Success status
        """
        try:
            local_index_path = f"csv-indexes/{model_id}/{model_id}"
            gcs_path = f"{self.csv_prefix}/{model_id}/{model_id}"
            
            return self.upload_index(local_index_path, gcs_path)
            
        except Exception as e:
            logger.error(f"Error uploading CSV index for model {model_id}: {e}")
            return False
    
    def download_csv_index(self, model_id: str) -> bool:
        """
        Download CSV index files from GCS.
        
        Args:
            model_id: Model ID for the CSV index
            
        Returns:
            Success status
        """
        try:
            local_index_path = f"csv-indexes/{model_id}/{model_id}"
            gcs_path = f"{self.csv_prefix}/{model_id}/{model_id}"
            
            return self.download_index(gcs_path, local_index_path)
            
        except Exception as e:
            logger.error(f"Error downloading CSV index for model {model_id}: {e}")
            return False
    
    def list_csv_indexes(self) -> list:
        """
        List all CSV indexes in GCS.
        
        Returns:
            List of CSV index model IDs
        """
        try:
            blobs = self.bucket.list_blobs(prefix=f"{self.csv_prefix}/")
            model_ids = set()
            
            for blob in blobs:
                if blob.name.endswith('.index'):
                    # Extract model_id from path: csv-indexes/model_id/model_id.index
                    path_parts = blob.name.split('/')
                    if len(path_parts) >= 3:
                        model_ids.add(path_parts[1])
            
            return list(model_ids)
            
        except Exception as e:
            logger.error(f"Error listing CSV indexes: {e}")
            return []
    
    def upload_csv_file(self, local_csv_path: str, model_id: str) -> bool:
        """
        Upload original CSV file to GCS for backup.
        
        Args:
            local_csv_path: Local path to CSV file
            model_id: Model ID
            
        Returns:
            Success status
        """
        try:
            gcs_path = f"{self.csv_prefix}/{model_id}/original.csv"
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_csv_path)
            
            logger.info(f"Uploaded CSV file to gs://{self.bucket_name}/{gcs_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading CSV file: {e}")
            return False
    
    def check_csv_index_exists(self, model_id: str) -> bool:
        """
        Check if CSV index exists in GCS.
        
        Args:
            model_id: Model ID to check
            
        Returns:
            True if index exists
        """
        try:
            index_blob_path = f"{self.csv_prefix}/{model_id}/{model_id}.index"
            docs_blob_path = f"{self.csv_prefix}/{model_id}/{model_id}.documents"
            docs_gz_blob_path = f"{self.csv_prefix}/{model_id}/{model_id}.documents.gz"
            
            index_blob = self.bucket.blob(index_blob_path)
            docs_blob = self.bucket.blob(docs_blob_path)
            docs_gz_blob = self.bucket.blob(docs_gz_blob_path)
            
            return index_blob.exists() and (docs_blob.exists() or docs_gz_blob.exists())
            
        except Exception as e:
            logger.error(f"Error checking CSV index existence: {e}")
            return False