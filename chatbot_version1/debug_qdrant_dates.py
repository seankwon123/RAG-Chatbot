#!/usr/bin/env python3
"""
Debug script to check what date data is stored in Qdrant
"""

from data_processing import DataProcessor
from qdrant_manager import QdrantManager

def debug_qdrant_dates():
    """Check what date information is stored in Qdrant"""
    print("Checking date information in Qdrant...")
    
    data_processor = DataProcessor()
    qdrant_manager = QdrantManager()
    
    # Test search for a sample query
    query = "latest blog post"
    query_embedding = data_processor.embedding_model.encode([query])[0].tolist()
    
    # Search with higher limit to see more results
    results = qdrant_manager.search_similar_articles(
        query_embedding=query_embedding,
        limit=10,
        score_threshold=0.0
    )
    
    print(f"Found {len(results)} articles")
    print("\nSample articles with their stored date information:")
    print("=" * 80)
    
    for i, article in enumerate(results[:5], 1):
        print(f"\nArticle {i}:")
        print(f"ID: {article['id']}")
        print(f"Title: {article['title'][:60]}...")
        print(f"Published Date (raw): {repr(article.get('published_date'))}")
        print(f"Published Date (type): {type(article.get('published_date'))}")
        print(f"Author: {article.get('author', 'Unknown')}")
        print(f"Score: {article['score']:.3f}")
        
        # Check if the date is being converted to string properly
        pub_date = article.get('published_date')
        if pub_date:
            print(f"Date as string: '{str(pub_date)}'")
            print(f"Date != 'None': {pub_date != 'None'}")
            print(f"Date != None: {pub_date != None}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    debug_qdrant_dates()