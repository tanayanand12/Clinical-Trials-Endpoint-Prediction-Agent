#!/usr/bin/env python3
# scripts/test_prediction.py
import requests #type: ignore
import json
import argparse

def test_prediction(base_url, query, docs_model_id, csv_model_id):
    """Test the endpoint prediction system."""
    
    url = f"{base_url}/predict"
    payload = {
        "query": query,
        "csv_model_id": csv_model_id,
        "docs_model_id": docs_model_id,
        "top_k_docs": 8,
        "top_k_csv": 12,
        "return_trials": 6
    }
    
    print(f"Testing endpoint prediction...")
    print(f"Query: {query}")
    print(f"URL: {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Success!")
            print(f"Primary endpoint: {result.get('predicted_primary_time_days')} days")
            print(f"Secondary endpoint: {result.get('predicted_secondary_time_days')} days")
            print(f"Confidence: {result.get('confidence_score', 0):.2f}")
            print(f"Supporting trials: {len(result.get('supporting_trials', []))}")
            print(f"\nRationale: {result.get('rationale', '')[:200]}...")
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"✗ Request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test endpoint prediction")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--query", default="Phase 2 oncology trial for breast cancer treatment", help="Test query")
    parser.add_argument("--docs_model_id", default="test_docs", help="Docs model ID")
    parser.add_argument("--csv_model_id", default="test_csv", help="CSV model ID")
    
    args = parser.parse_args()
    
    test_prediction(args.url, args.query, args.docs_model_id, args.csv_model_id)

if __name__ == "__main__":
    main()