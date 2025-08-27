import psycopg2
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
from config import get_postgres_connection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor with embedding model"""
        logger.info("Loading sentence transformer model...")
        # Using a good free model that's fast and accurate
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully!")
    
    def fetch_articles_from_db(self) -> List[Dict[str, Any]]:
        """Fetch all articles from PostgreSQL database"""
        logger.info("Fetching articles from PostgreSQL...")
        
        conn = get_postgres_connection()
        if not conn:
            raise Exception("Could not connect to PostgreSQL")
        
        try:
            cursor = conn.cursor()
            
            # Fetch all articles with necessary fields
            query = """
                SELECT id, title, url, content, excerpt, author, 
                       published_date, tags, word_count,
                       created_at, updated_at
                FROM articles 
                WHERE content IS NOT NULL 
                AND content != ''
                ORDER BY id
            """
            
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            articles = []
            
            for row in cursor.fetchall():
                article = dict(zip(columns, row))
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from database")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def prepare_text_for_embedding(self, article: Dict[str, Any]) -> str:
        """Prepare article text for embedding generation"""
        # Combine title, excerpt, and content for better context
        text_parts = []
        
        if article.get('title'):
            text_parts.append(f"Title: {article['title']}")
        
        if article.get('excerpt'):
            text_parts.append(f"Excerpt: {article['excerpt']}")
        
        if article.get('content'):
            # Use more content for better semantic understanding
            content = article['content']
            if len(content) > 4000:  # Increased from 2000 to 4000 for better context
                content = content[:4000] + "..."
            text_parts.append(f"Content: {content}")
        
        combined_text = "\n\n".join(text_parts)
        
        # Debug: Show what's being embedded for first few articles
        if article['id'] <= 3:
            print(f"DEBUG - Article {article['id']} embedding text length: {len(combined_text)} chars")
            print(f"DEBUG - Includes title: {bool(article.get('title'))}")
            print(f"DEBUG - Includes excerpt: {bool(article.get('excerpt'))}")
            print(f"DEBUG - Content length: {len(article.get('content', ''))}")
        
        return combined_text
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            # Generate embeddings in batches for efficiency
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Convert to list of lists for JSON serialization
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.info("Embeddings generated successfully!")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def process_articles_for_qdrant(self) -> List[Dict[str, Any]]:
        """Process all articles and prepare them for Qdrant storage"""
        logger.info("Processing articles for Qdrant storage...")
        
        # Fetch articles from database
        articles = self.fetch_articles_from_db()
        
        if not articles:
            logger.warning("No articles found in database!")
            return []
        
        # Prepare texts for embedding
        texts = []
        for article in articles:
            text = self.prepare_text_for_embedding(article)
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Combine articles with their embeddings
        processed_articles = []
        for i, article in enumerate(articles):
            processed_article = {
                'id': article['id'],
                'title': article['title'],
                'url': article['url'],
                'content': article['content'],
                'excerpt': article['excerpt'],
                'author': article['author'],
                'published_date': str(article['published_date']) if article['published_date'] else None,
                'tags': article['tags'],
                'word_count': article['word_count'],
                'embedding': embeddings[i],
                'text_for_embedding': texts[i]  # Keep this for debugging
            }
            processed_articles.append(processed_article)
        
        logger.info(f"Processed {len(processed_articles)} articles successfully!")
        return processed_articles

def test_data_processing():
    """Test the data processing functionality"""
    try:
        processor = DataProcessor()
        
        # Test fetching articles
        articles = processor.fetch_articles_from_db()
        print(f"OK: Fetched {len(articles)} articles")
        
        if articles:
            # Test processing first article
            sample_text = processor.prepare_text_for_embedding(articles[0])
            print(f"OK: Sample text length: {len(sample_text)} characters")
            
            # Test embedding generation on small sample
            sample_embeddings = processor.generate_embeddings([sample_text])
            print(f"OK: Generated embedding with {len(sample_embeddings[0])} dimensions")
            
            print("\nOK: Data processing test completed successfully!")
        else:
            print("!!! No articles found in database")
            
    except Exception as e:
        print(f"ERROR: Data processing test failed: {e}")

if __name__ == "__main__":
    test_data_processing()