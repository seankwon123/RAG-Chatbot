#!/usr/bin/env python3
"""
Debug script to test search functionality
"""

from data_processing import DataProcessor
from qdrant_manager import QdrantManager

def debug_search():
    """Debug the search functionality"""
    print("Debugging search functionality...")
    
    # Initialize
    data_processor = DataProcessor()
    qdrant_manager = QdrantManager()
    
    # Get collection info
    info = qdrant_manager.get_collection_info()
    print(f"Collection has {info.get('points_count', 0)} articles")
    
    # Test different queries and thresholds
    test_queries = [
        "technology innovation",
        "business strategy", 
        "software development",
        "data analysis"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        # Generate embedding
        embedding = data_processor.embedding_model.encode([query])[0].tolist()
        
        # Test with different score thresholds
        thresholds = [0.0, 0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            results = qdrant_manager.search_similar_articles(
                embedding, 
                limit=3, 
                score_threshold=threshold
            )
            print(f"  Threshold {threshold}: {len(results)} results")
            
            if results:
                for i, result in enumerate(results[:2], 1):
                    print(f"    {i}. {result['title'][:50]}... (score: {result['score']:.3f})")
                break  # Found results, move to next query
    
    print("\nDebug completed!")

if __name__ == "__main__":
    debug_search()