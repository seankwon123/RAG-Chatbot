#!/usr/bin/env python3
"""
Script to populate Qdrant with all articles from PostgreSQL
This should be run once to set up the vector database
"""

import time
import logging
from data_processing import DataProcessor
from qdrant_manager import QdrantManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to populate Qdrant with all articles"""
    print("Starting Qdrant population process...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Initialize processors
        logger.info("Initializing data processor...")
        data_processor = DataProcessor()
        
        logger.info("Initializing Qdrant manager...")
        qdrant_manager = QdrantManager()
        
        # Create/verify collection
        logger.info("Creating Qdrant collection...")
        qdrant_manager.create_collection(vector_size=384, force_recreate=False)
        
        # Check if collection already has data
        collection_info = qdrant_manager.get_collection_info()
        if collection_info and collection_info.get('points_count', 0) > 0:
            print(f"WARNING: Collection already contains {collection_info['points_count']} articles")
            response = input("Do you want to recreate the collection? (y/N): ")
            if response.lower() == 'y':
                qdrant_manager.create_collection(vector_size=384, force_recreate=True)
            else:
                print("Skipping population. Collection already exists.")
                return
        
        # Process all articles
        print("\nProcessing articles...")
        processed_articles = data_processor.process_articles_for_qdrant()
        
        if not processed_articles:
            print("ERROR: No articles found to process!")
            return
        
        print(f"SUCCESS: Processed {len(processed_articles)} articles")
        
        # Store in Qdrant
        print("\nStoring articles in Qdrant...")
        qdrant_manager.store_articles(processed_articles)
        
        # Verify storage
        final_info = qdrant_manager.get_collection_info()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 50)
        print("Population completed successfully!")
        print(f"Articles stored: {final_info.get('points_count', 'Unknown')}")
        print(f"Total time: {duration:.1f} seconds")
        print(f"Average: {duration/len(processed_articles):.2f} seconds per article")
        print("=" * 50)
        
        # Quick test search
        print("\nTesting search functionality...")
        test_query = "technology innovation"
        test_embedding = data_processor.embedding_model.encode([test_query])[0].tolist()
        results = qdrant_manager.search_similar_articles(test_embedding, limit=3)
        
        if results:
            print(f"Search test successful! Found {len(results)} relevant articles:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']} (score: {result['score']:.3f})")
        else:
            print("WARNING: Search test returned no results")
        
    except Exception as e:
        logger.error(f"Error during population: {e}")
        print(f"ERROR: Population failed: {e}")
        raise

if __name__ == "__main__":
    main()