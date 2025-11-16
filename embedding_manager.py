# embedding_manager.py
"""
Embedding manager for generating text embeddings using Ollama.
Handles conversion of articles to vectors for semantic search.
"""

import logging
from typing import List, Optional
import requests
import numpy as np
from abc import ABC, abstractmethod

from models import Article, RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        pass
    
    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        pass

class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding provider"""
    
    def __init__(self, model: str = "all-minilm", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._dimension = None
        
        # Test connection and get dimension
        if not self._test_connection():
            raise ConnectionError(f"Failed to connect to Ollama at {base_url}")

        logger.info(f"Initialized Ollama embedding provider with model: {model}")
        logger.info(f"Embedding dimension: {self._dimension}")
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama and get embedding dimension"""
        try:
            # Try to get a test embedding
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": "test"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    self._dimension = len(data["embedding"])
                    return True
            
            # # If nomic-embed-text doesn't work, try alternatives
            # logger.warning(f"Model {self.model} not available, trying alternatives...")
            
            # # Try alternative embedding models
            # alternative_models = ["mxbai-embed-large", "all-minilm", "llama2"]
            
            # for alt_model in alternative_models:
            #     try:
            #         response = requests.post(
            #             f"{self.base_url}/api/embeddings",
            #             json={
            #                 "model": alt_model,
            #                 "prompt": "test"
            #             },
            #             timeout=30
            #         )
                    
            #         if response.status_code == 200:
            #             data = response.json()
            #             if "embedding" in data:
            #                 self.model = alt_model
            #                 self._dimension = len(data["embedding"])
            #                 logger.info(f"✅ Using alternative model: {alt_model}")
            #                 return True
            #     except:
            #         continue
            logger.error(f"Failed to get embedding from Ollama - check models.py 'EMBEDDING_MODEL': {response.status_code} - {response.text}")
            
            return False
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            # Truncate text if too long (Ollama has limits)
            max_length = 8192
            if len(text) > max_length:
                text = text[:max_length]
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["embedding"]
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Failed to get embedding: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting Ollama embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (one by one for Ollama)"""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"  Generated {i + 1}/{total} embeddings...")
                    
            except Exception as e:
                logger.error(f"Failed to embed text {i}: {e}")
                # Return zero vector on failure
                embeddings.append([0.0] * self._dimension)
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self._dimension or 768  # Default fallback

class EmbeddingManager:
    """Main embedding manager that handles article processing"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig
        
        # Create Ollama provider
        self.provider = OllamaEmbeddingProvider(
            model=self.config.EMBEDDING_MODEL,
            base_url=self.config.OLLAMA_HOST
        )
        
        # Update vector size in config based on actual dimension
        self.config.EMBEDDING_DIMENSION = self.provider.get_dimension()
        
        logger.info(f"Embedding Manager initialized")
        logger.info(f"Using model: {self.provider.model}")
        logger.info(f"Vector dimension: {self.config.EMBEDDING_DIMENSION}")
    
    def embed_article(self, article: Article) -> List[float]:
        """
        Generate embedding for a single article.
        Uses the combined searchable text that includes all fields.
        """
        # Use the combined searchable text (title + tags + excerpt + content)
        text = article.to_searchable_text()
        
        # Generate embedding
        return self.provider.get_embedding(text)
    
    def embed_articles_batch(self, articles: List[Article]) -> List[List[float]]:
        """
        Generate embeddings for multiple articles.
        This is the main method for populating the vector database.
        """
        # Convert articles to searchable text
        texts = []
        for article in articles:
            text = article.to_searchable_text()
            texts.append(text)
        
        logger.info(f"Processing {len(articles)} articles for embedding...")
        
        # Generate embeddings
        embeddings = self.provider.get_embeddings_batch(texts)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Enhances the query slightly for better matching.
        """
        # Add some context to improve search
        enhanced_query = f"Article about: {query}. Topics: {query}"
        
        return self.provider.get_embedding(enhanced_query)
    
    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Cosine similarity
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def test_embedding(self) -> bool:
        """Test the embedding system"""
        try:
            # Test with sample text
            test_texts = [
                "DevOps best practices for continuous integration",
                "React component lifecycle methods",
                "Cloud architecture patterns"
            ]
            
            print("\nTesting embeddings...")
            
            for text in test_texts:
                embedding = self.provider.get_embedding(text)
                print(f"  Testing Done'{text[:30]}...' -> vector of dimension {len(embedding)}")
            
            # Test similarity
            vec1 = self.provider.get_embedding("DevOps continuous integration")
            vec2 = self.provider.get_embedding("DevOps CI/CD pipeline")
            vec3 = self.provider.get_embedding("Machine learning algorithms")
            
            sim1 = self.compute_similarity(vec1, vec2)
            sim2 = self.compute_similarity(vec1, vec3)
            
            print(f"\nSimilarity test:")
            print(f"  DevOps vs DevOps CI/CD: {sim1:.3f} (should be high)")
            print(f"  DevOps vs Machine Learning: {sim2:.3f} (should be lower)")
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
            return False

def test_embedding_manager():
    """Test the embedding manager"""
    print("\n" + "="*50)
    print("Testing Embedding Manager")
    print("="*50)
    
    try:
        # Initialize
        manager = EmbeddingManager()
        
        # Run tests
        if manager.test_embedding():
            print("\n✅ Embedding manager test complete!")
        else:
            print("\n❌ Embedding manager test failed!")
            
    except Exception as e:
        print(f"\n❌ Failed to initialize embedding manager: {e}")
        print("\n !!! Make sure Ollama is running and has an embedding model installed:")
        print("   1. Check if Ollama is running: curl http://localhost:11434")
        print("   2. Pull an embedding model: ollama pull all-minilm")
        print("   3. Or try: ollama pull mxbai-embed-large")

if __name__ == "__main__":
    test_embedding_manager()