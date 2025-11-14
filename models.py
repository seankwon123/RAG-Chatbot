# models.py
"""
Enhanced database models and configuration for the RAG system.
This improves on the basic config by adding structured models and better search capabilities.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SearchMode(Enum):
    """Different search modes for the RAG system"""
    SEMANTIC = "semantic"      # Pure vector similarity search
    KEYWORD = "keyword"        # Traditional keyword matching
    HYBRID = "hybrid"         # Combination of vector and keyword
    TEMPORAL = "temporal"     # Date-based filtering with semantic
    COUNT = "count"          # Counting matching documents

@dataclass
class Article:
    """
    Article model matching your PostgreSQL schema.
    This structure ensures we can properly weight different fields for search.
    """
    id: int
    title: str
    url: str
    content: Optional[str] = None
    excerpt: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    tags: Optional[str] = None  # JSON array stored as text
    word_count: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    content_hash: Optional[str] = None
    
    def get_tags_list(self) -> List[str]:
        """Parse tags JSON string into list"""
        if not self.tags:
            return []
        try:
            # Handle both string and list formats
            if isinstance(self.tags, str):
                tags = json.loads(self.tags) if self.tags.startswith('[') else [self.tags]
            else:
                tags = self.tags
            return tags if isinstance(tags, list) else []
        except:
            return []
    
    def to_searchable_text(self) -> str:
        """
        Combine ALL searchable fields into a single text for vectorization.
        This is CRITICAL for making sure vector search considers all relevant fields,
        not just the title.
        
        Different fields are weighted differently by repetition.
        """
        parts = []
        
        # Title gets highest weight (3x)
        if self.title:
            parts.extend([self.title] * 3)
        
        # Tags get high weight (3x) for topic matching
        tags_list = self.get_tags_list()
        if tags_list:
            tags_text = " ".join(tags_list)
            parts.extend([tags_text] * 3)
        
        # Excerpt gets medium weight (2x)
        if self.excerpt:
            parts.extend([self.excerpt] * 2)
        
        # Author for author-based queries
        if self.author:
            parts.append(f"Written by {self.author}")
        
        # Content gets normal weight (1x), limited to prevent overwhelming
        if self.content:
            # Take first 2000 chars to balance with other fields
            content_preview = self.content[:2000]
            parts.append(content_preview)
        
        # Add published year for temporal context
        if self.published_date:
            year = self.published_date.year
            parts.append(f"Published in {year}")
        
        return " ".join(parts)
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert article to metadata dict for Qdrant storage"""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "excerpt": self.excerpt or "",
            "author": self.author or "",
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "published_timestamp": self.published_date.timestamp() if self.published_date else 0,
            "tags": self.get_tags_list(),
            "tags_string": " ".join(self.get_tags_list()),  # For keyword search
            "word_count": self.word_count or 0,
            "content_preview": self.content[:500] if self.content else "",
            "searchable_text": self.title + " " + (self.excerpt or "") + " " + " ".join(self.get_tags_list())
        }

@dataclass
class SearchResult:
    """Search result with relevance score and explanation"""
    article: Article
    score: float
    match_reason: str = ""
    matched_fields: List[str] = None  # Which fields matched the query
    
    def __post_init__(self):
        if self.matched_fields is None:
            self.matched_fields = []

class RAGConfig:
    """
    Enhanced configuration that builds on your existing config.py
    """
    
    # PostgreSQL settings (from your .env)
    DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    DB_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
    DB_NAME = os.getenv('POSTGRES_DB', 'n8n')
    DB_USER = os.getenv('POSTGRES_USER', 'n8n')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    
    # Qdrant settings (from your .env)
    QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
    QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'articles_collection')
    
    # Ollama settings (from your .env)
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5')
    
    # Embedding settings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-minilm')  # Ollama embedding model
    EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension, will be auto-detected
    
    # Search settings
    DEFAULT_TOP_K = 10
    SIMILARITY_THRESHOLD = 0.3  # Minimum similarity for semantic search
    COUNT_THRESHOLD = 0.25  # Lower threshold when counting articles
    HYBRID_ALPHA = 0.7  # Balance: 0.7 semantic, 0.3 keyword
    
    # Processing settings
    BATCH_SIZE = 20
    MAX_CONTENT_LENGTH = 10000
    
    @classmethod
    def get_postgres_config(cls) -> Dict[str, Any]:
        """Get PostgreSQL connection parameters"""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'database': cls.DB_NAME,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required settings are present"""
        required = [
            cls.DB_HOST, cls.DB_NAME, cls.DB_USER, cls.DB_PASSWORD,
            cls.QDRANT_HOST, cls.QDRANT_PORT,
            cls.OLLAMA_HOST, cls.OLLAMA_MODEL
        ]
        
        missing = [s for s in required if not s]
        if missing:
            print(f"Missing required settings: {missing}")
            return False
        
        return True
    
    @classmethod
    def display(cls):
        """Display current configuration (hiding passwords)"""
        print("\n=== Current Configuration ===")
        print(f"PostgreSQL: {cls.DB_USER}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}")
        print(f"Qdrant: {cls.QDRANT_HOST}:{cls.QDRANT_PORT} (Collection: {cls.QDRANT_COLLECTION})")
        print(f"Ollama: {cls.OLLAMA_HOST} (Model: {cls.OLLAMA_MODEL})")
        print(f"Embeddings: {cls.EMBEDDING_MODEL}")
        print(f"Search: top_k={cls.DEFAULT_TOP_K}, threshold={cls.SIMILARITY_THRESHOLD}")
        print("=" * 30)

# For backward compatibility with your existing config.py
POSTGRES_CONFIG = RAGConfig.get_postgres_config()
QDRANT_HOST = RAGConfig.QDRANT_HOST
QDRANT_PORT = RAGConfig.QDRANT_PORT
COLLECTION_NAME = RAGConfig.QDRANT_COLLECTION

if __name__ == "__main__":
    # Test configuration
    if RAGConfig.validate():
        RAGConfig.display()
        print("\nGOOD: Configuration is valid!")
    else:
        print("\nBAD: Configuration has errors!")