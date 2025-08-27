from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import logging
import uuid
from config import get_qdrant_client, COLLECTION_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self):
        """Initialize Qdrant manager"""
        self.client = get_qdrant_client()
        self.collection_name = COLLECTION_NAME
        
        if not self.client:
            raise Exception("Could not connect to Qdrant")
        
        logger.info("QdrantManager initialized successfully")
    
    def create_collection(self, vector_size: int = 384, force_recreate: bool = False):
        """Create a collection in Qdrant for storing article embeddings"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                if force_recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(collection_name=self.collection_name)
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return True
            
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def store_articles(self, articles: List[Dict[str, Any]]) -> bool:
        """Store articles with embeddings in Qdrant"""
        try:
            logger.info(f"Storing {len(articles)} articles in Qdrant...")
            
            points = []
            for article in articles:
                # Create point for Qdrant
                point = PointStruct(
                    id=article['id'],  # Use article ID as point ID
                    vector=article['embedding'],
                    payload={
                        'title': article['title'],
                        'url': article['url'],
                        'content': article['content'],
                        'excerpt': article['excerpt'],
                        'author': article['author'],
                        'published_date': article['published_date'],
                        'tags': article['tags'],
                        'word_count': article['word_count']
                    }
                )
                points.append(point)
            
            # Upload points in batches for efficiency
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            logger.info(f"Successfully stored {len(articles)} articles in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error storing articles: {e}")
            raise
    
    def search_similar_articles(self, query_embedding: List[float], limit: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar articles using vector similarity"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for scored_point in search_result:
                result = {
                    'id': scored_point.id,
                    'score': scored_point.score,
                    'title': scored_point.payload['title'],
                    'url': scored_point.payload['url'],
                    'content': scored_point.payload['content'],
                    'excerpt': scored_point.payload['excerpt'],
                    'author': scored_point.payload['author'],
                    'published_date': scored_point.payload['published_date'],
                    'tags': scored_point.payload['tags'],
                    'word_count': scored_point.payload['word_count']
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar articles")
            return results
            
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                'name': info.config.params.vectors.size,
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the collection (use with caution)"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

def test_qdrant_manager():
    """Test Qdrant manager functionality"""
    try:
        manager = QdrantManager()
        
        # Test creating collection
        manager.create_collection(vector_size=384)
        print("Collection created/verified")
        
        # Test collection info
        info = manager.get_collection_info()
        if info:
            print(f"Collection info: {info['points_count']} points, {info['vector_size']} dimensions")
        
        print("QdrantManager test completed successfully!")
        
    except Exception as e:
        print(f"QdrantManager test failed: {e}")

if __name__ == "__main__":
    test_qdrant_manager()