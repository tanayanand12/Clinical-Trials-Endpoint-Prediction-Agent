#!/usr/bin/env python3
# scripts/build_csv_index.py
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.csv_utils.csv_processor import CSVProcessor

def main():
    parser = argparse.ArgumentParser(description="Build CSV embeddings index")
    parser.add_argument("--csv_path", "-c", required=True, help="Path to CSV file")
    parser.add_argument("--model_id", "-m", required=True, help="Model ID for index")
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Batch size")
    parser.add_argument("--embedding_model", "-e", default="text-embedding-3-small", help="Embedding model")
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Initialize processor
    processor = CSVProcessor()
    
    # Validate CSV format
    if not processor.validate_csv_format(args.csv_path):
        print("Error: Invalid CSV format")
        sys.exit(1)
    
    print(f"Building CSV index for: {args.csv_path}")
    print(f"Model ID: {args.model_id}")
    print(f"Embedding model: {args.embedding_model}")
    
    # Process CSV
    success = processor.process_csv_file(
        csv_file_path=args.csv_path,
        model_id=args.model_id,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model
    )
    
    if success:
        print(f"✓ Successfully created CSV index: csv-indexes/{args.model_id}")
    else:
        print("✗ Failed to create CSV index")
        sys.exit(1)

if __name__ == "__main__":
    main()