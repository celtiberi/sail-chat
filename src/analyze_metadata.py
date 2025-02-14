from chromadb import PersistentClient
from pathlib import Path
import json

VECTORDB_PATH = Path("../forum-scrape/vectordb")

def analyze_collection(name: str):
    client = PersistentClient(path=str(VECTORDB_PATH))
    collection = client.get_collection(name)
    
    # Get all documents
    result = collection.get()
    
    print(f"\n=== Collection: {name} ===")
    print(f"Total documents: {len(result['documents'])}")
    print("\nSample metadata:")
    for meta in result['metadatas'][:5]:
        print(meta)
    
    # Count documents per topic
    topics = {}
    for meta in result['metadatas']:
        if meta and 'topic' in meta:
            topics[meta['topic']] = topics.get(meta['topic'], 0) + 1
    
    print("\nDocuments per topic:")
    for topic, count in sorted(topics.items()):
        print(f"{topic}: {count}")

if __name__ == "__main__":
    analyze_collection("forum_content") 