# qdrant_manager.py
"""
Qdrant vector database manager for the Bitovi RAG system.
Handles vector storage, semantic search, and filtering.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, Range, MatchValue,
    FilterSelector, HasIdCondition, MatchAny
)

from models import Article, RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    """Manages Qdrant vector database operations"""
    
    def __init__(self, config: RAGConfig = None):
        """Initialize Qdrant client"""
        self.config = config or RAGConfig
        
        try:
            self.client = QdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT
            )
            self.collection_name = self.config.QDRANT_COLLECTION
            logger.info(f"Connected to Qdrant at {self.config.QDRANT_HOST}:{self.config.QDRANT_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test the Qdrant connection"""
        try:
            collections = self.client.get_collections()
            logger.info(f"✅ Qdrant connected. Found {len(collections.collections)} collections")
            
            # List collections
            for collection in collections.collections:
                logger.info(f"  Collection: {collection.name}")
            
            # Check if our collection exists
            collection_names = [c.name for c in collections.collections]
            if self.collection_name in collection_names:
                info = self.client.get_collection(self.collection_name)
                logger.info(f"  📦 '{self.collection_name}' has {info.points_count} points")
            
            return True
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            return False
    
    def create_collection(self, vector_size: int, recreate: bool = False):
        """
        Create or recreate the Qdrant collection.
        
        Args:
            vector_size: Dimension of the embedding vectors
            recreate: If True, delete existing collection and create new
        """
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if recreate:
                    logger.info(f"🗑️  Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"✅ Collection {self.collection_name} already exists")
                    return
            
            logger.info(f"📦 Creating collection: {self.collection_name} with vector size {vector_size}")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indexes for efficient filtering
            self._create_payload_indexes()
            
            logger.info(f"✅ Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def _create_payload_indexes(self):
        """Create indexes on payload fields for faster filtering"""
        index_fields = [
            ("published_timestamp", "float"),  # For date filtering
            ("author", "keyword"),            # For author filtering
            ("tags", "keyword"),              # For tag filtering
            ("id", "integer"),               # For ID lookups
        ]
        
        for field_name, field_type in index_fields:
            try:
                if field_type == "float":
                    # For numeric fields
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema="float"
                    )
                elif field_type == "integer":
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema="integer"
                    )
                else:
                    # For text fields
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema="keyword"
                    )
                logger.info(f"  Created index for field: {field_name}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"  Could not create index for {field_name}: {e}")
    
    def insert_articles(self, articles: List[Article], embeddings: List[List[float]]):
        """
        Insert articles with their embeddings into Qdrant.
        
        Args:
            articles: List of Article objects
            embeddings: List of embedding vectors (same order as articles)
        """
        if len(articles) != len(embeddings):
            raise ValueError("Number of articles must match number of embeddings")
        
        points = []
        for article, embedding in zip(articles, embeddings):
            # Create point with article metadata
            point = PointStruct(
                id=article.id,  # Use article ID as point ID
                vector=embedding,
                payload=article.to_metadata()
            )
            points.append(point)
        
        # Batch upsert for efficiency
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            total_inserted += len(batch)
            logger.info(f"  Inserted batch {i//batch_size + 1} ({total_inserted}/{len(points)} points)")
        
        logger.info(f"✅ Successfully inserted {len(points)} articles into Qdrant")
    
    def search_semantic(
        self, 
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Filter] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Optional Qdrant filter
            
        Returns:
            List of (article_id, score, metadata) tuples
        """
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding,
            "limit": top_k
        }
        
        if score_threshold:
            search_params["score_threshold"] = score_threshold
        
        if filter_conditions:
            search_params["query_filter"] = filter_conditions
        
        search_result = self.client.search(**search_params)
        
        results = []
        for point in search_result:
            results.append((
                point.id,
                point.score,
                point.payload
            ))
        
        return results
    
    def search_by_date_range(
        self,
        query_embedding: Optional[List[float]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search articles within a date range, optionally with semantic search.
        """
        conditions = []
        
        if start_date:
            conditions.append(
                FieldCondition(
                    key="published_timestamp",
                    range=Range(gte=start_date.timestamp())
                )
            )
        
        if end_date:
            conditions.append(
                FieldCondition(
                    key="published_timestamp",
                    range=Range(lte=end_date.timestamp())
                )
            )
        
        filter_obj = Filter(must=conditions) if conditions else None
        
        if query_embedding:
            # Semantic search with date filtering
            return self.search_semantic(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_conditions=filter_obj
            )
        else:
            # Just retrieve by date without semantic search
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                (point.id, 1.0, point.payload)
                for point in results[0]
            ]
    
    def search_by_tags(
        self,
        tags: List[str],
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search articles that contain any of the specified tags"""
        
        # Since tags are stored as an array in payload, we need to use MatchAny
        filter_obj = Filter(
            should=[
                FieldCondition(
                    key="tags",
                    match=MatchAny(any=tags)
                )
            ]
        )
        
        if query_embedding:
            # Semantic search with tag filtering
            return self.search_semantic(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_conditions=filter_obj
            )
        else:
            # Just retrieve by tags
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                (point.id, 1.0, point.payload)
                for point in results[0]
            ]
    
    def get_latest_article(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Get the most recent article based on published_date"""
        # Qdrant doesn't have native sorting, so we'll get all and sort
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Get all
            with_payload=True,
            with_vectors=False
        )
        
        articles_with_dates = []
        for point in results[0]:
            if point.payload.get('published_timestamp'):
                articles_with_dates.append((
                    point.id,
                    point.payload,
                    point.payload['published_timestamp']
                ))
        
        if not articles_with_dates:
            return None
        
        # Sort by timestamp descending
        articles_with_dates.sort(key=lambda x: x[2], reverse=True)
        latest = articles_with_dates[0]
        
        return (latest[0], latest[1])
    
    def count_matching_articles(
        self,
        query_embedding: List[float],
        score_threshold: float = 0.3
    ) -> int:
        """Count articles that match the query above threshold"""
        # We need to search with a high limit to count
        results = self.search_semantic(
            query_embedding=query_embedding,
            top_k=1000,  # High limit for counting
            score_threshold=score_threshold
        )
        
        return len(results)
    
    def delete_all_articles(self):
        """Delete all articles from the collection (useful for reindexing)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

def test_qdrant_manager():
    """Test the Qdrant manager"""
    print("\n" + "="*50)
    print("Testing Qdrant Manager")
    print("="*50)
    
    qdrant = QdrantManager()
    
    # Test connection
    if not qdrant.test_connection():
        print("Failed to connect to Qdrant!")
        return
    
    # Get collection info if it exists
    info = qdrant.get_collection_info()
    if info:
        print(f"\n📊 Collection Info:")
        print(f"  Points: {info.get('points_count', 0)}")
        print(f"  Status: {info.get('status', 'Unknown')}")
    
    print("\n✅ Qdrant manager test complete!")

if __name__ == "__main__":
    test_qdrant_manager()