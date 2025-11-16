import os
import psycopg2
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'n8n'),
    'user': os.getenv('POSTGRES_USER', 'n8n'),
    'password': os.getenv('POSTGRES_PASSWORD', 'n8n')
}

# Qdrant Configuration
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))

# Ollama Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5')

# Collection name for articles
COLLECTION_NAME = "articles_collection"

def get_postgres_connection():
    """Create PostgreSQL connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def get_qdrant_client():
    """Create Qdrant client"""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None

def test_connections():
    """Test all connections"""
    print("Testing connections...")
    
    # Test PostgreSQL
    pg_conn = get_postgres_connection()
    if pg_conn:
        print("✓ PostgreSQL connected")
        pg_conn.close()
    else:
        print("✗ PostgreSQL connection failed")
    
    # Test Qdrant
    qdrant_client = get_qdrant_client()
    if qdrant_client:
        try:
            collections = qdrant_client.get_collections()
            print("✓ Qdrant connected")
        except Exception as e:
            print(f"✗ Qdrant connection failed: {e}")
    else:
        print("✗ Qdrant connection failed")
    
    # Test Ollama (we'll add this next)
    print("Ollama test will be added in next file...")

if __name__ == "__main__":
    test_connections()