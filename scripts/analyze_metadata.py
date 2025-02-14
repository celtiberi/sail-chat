from chromadb import PersistentClient
from pathlib import Path
from collections import defaultdict
import json
from pprint import pprint

# Get vectordb path
VECTORDB_PATH = Path("../../forum-scrape/vectordb")

def analyze_collection_metadata(collection_name: str):
    """Analyze metadata for a specific collection."""
    client = PersistentClient(path=str(VECTORDB_PATH))
    collection = client.get_collection(collection_name)
    
    # Get all metadata
    result = collection.get()
    
    # Analyze metadata keys and their unique values
    metadata_analysis = defaultdict(set)
    
    for metadata in result['metadatas']:
        if metadata:  # Some might be None
            for key, value in metadata.items():
                metadata_analysis[key].add(str(value))
    
    # Convert sets to sorted lists for better display
    metadata_summary = {
        key: sorted(list(values))
        for key, values in metadata_analysis.items()
    }
    
    return metadata_summary

def main():
    # Analyze both collections
    collections = ["forum_content", "forum_hierarchies"]
    
    for collection in collections:
        print(f"\n=== Metadata Analysis for {collection} ===")
        metadata_summary = analyze_collection_metadata(collection)
        pprint(metadata_summary)
        
        # Save to file
        output_file = f"{collection}_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata_summary, f, indent=2)
        print(f"\nMetadata summary saved to {output_file}")

if __name__ == "__main__":
    main() 