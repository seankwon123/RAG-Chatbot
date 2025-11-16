import requests
import json
import logging
from typing import List, Dict, Any, Optional
from data_processing import DataProcessor
from qdrant_manager import QdrantManager
from config import OLLAMA_HOST, OLLAMA_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        """Initialize RAG system"""
        logger.info("Initializing RAG system...")
        
        self.data_processor = DataProcessor()
        self.qdrant_manager = QdrantManager()
        self.ollama_host = OLLAMA_HOST
        self.ollama_model = OLLAMA_MODEL
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        logger.info("RAG system initialized successfully!")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if any(self.ollama_model in name for name in model_names):
                    logger.info(f"Ollama connection successful. Model '{self.ollama_model}' is available.")
                else:
                    logger.warning(f"Model '{self.ollama_model}' not found. Available models: {model_names}")
                    raise Exception(f"Model '{self.ollama_model}' not available")
            else:
                raise Exception(f"Ollama API returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Cannot connect to Ollama at {self.ollama_host}: {e}")
    
    def retrieve_relevant_articles(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant articles based on the query"""
        logger.info(f"Retrieving articles for query: '{query}'")
        
        try:
            # Generate embedding for the query
            query_embedding = self.data_processor.embedding_model.encode([query])[0].tolist()
            
            # Search for similar articles
            results = self.qdrant_manager.search_similar_articles(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=0.2  # Minimum relevance threshold
            )
            
            logger.info(f"Retrieved {len(results)} relevant articles")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            raise
    
    def format_context(self, articles: List[Dict[str, Any]]) -> str:
        """Format retrieved articles into context for the LLM"""
        if not articles:
            return "No relevant articles found."
        
        context_parts = []
        for i, article in enumerate(articles, 1):
            # Create a formatted context entry with publication date
            published_date = article.get('published_date', None)
            if published_date and published_date != 'None' and published_date.strip():
                # Extract just the date part (YYYY-MM-DD) from the full timestamp
                date_str = published_date.split(' ')[0]  # Gets '2024-02-08' from '2024-02-08 00:00:00+00:00'
                date_display = f"Published: {date_str}"
            else:
                date_display = "Published: Date not available"
            
            context_entry = f"""
Article {i}:
Title: {article['title']}
Author: {article.get('author', 'Unknown')}
{date_display}
URL: {article.get('url', 'N/A')}
Content: {article['excerpt'] or article['content'][:500] + '...'}
"""
            context_parts.append(context_entry.strip())
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Ollama with the provided context"""
        try:
            # Create the prompt
            prompt = f"""You are a helpful assistant that answers questions based on the provided article context. 
Use only the information from the articles to answer the question. If the information is not available in the articles, say so clearly.

Context Articles:
{context}

Question: {query}

Answer based on the articles above:"""
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more focused responses
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "Sorry, I encountered an error generating the response."
                
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "Sorry, the response took too long to generate."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating the response."
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function that combines retrieval and generation"""
        logger.info(f"Processing chat query: '{query}'")
        
        try:
            # Step 1: Retrieve relevant articles
            relevant_articles = self.retrieve_relevant_articles(query, limit=3)
            
            if not relevant_articles:
                return {
                    'response': "I couldn't find any relevant articles to answer your question. Please try rephrasing or asking about a different topic.",
                    'sources': [],
                    'success': True
                }
            
            # Step 2: Format context
            context = self.format_context(relevant_articles)
            
            # Step 3: Generate response
            response = self.generate_response(query, context)
            
            # Step 4: Format sources for display
            sources = []
            for article in relevant_articles:
                sources.append({
                    'title': article['title'],
                    'author': article.get('author', 'Unknown'),
                    'url': article.get('url', ''),
                    'score': round(article['score'], 3)
                })
            
            return {
                'response': response,
                'sources': sources,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in chat function: {e}")
            return {
                'response': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'success': False
            }

def test_rag_system():
    """Test the RAG system"""
    try:
        rag = RAGSystem()
        
        # Test queries
        test_queries = [
            # "What is Bitovi's latest blog post about?",
            # "Can you show me all Bitovi articles about DevOps?",
            # "How many articles does Bitovi have about AI?",
            # "What kind of tools does Bitovi recommend for E2E testing?",

            "What are the best practices for software development?",
            "How can businesses implement AI strategies?",
            "Show me all articles about DevOps"  # This should return more articles
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            print("-" * 50)
            
            result = rag.chat(query)
            
            if result['success']:
                print("Response:")
                print(result['response'])
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    # Show clean date format
                    pub_date = source.get('published_date', 'Unknown')
                    if pub_date and pub_date != 'Unknown':
                        clean_date = pub_date.split(' ')[0] if ' ' in pub_date else pub_date
                    else:
                        clean_date = 'Unknown'
                    print(f"  {i}. {source['title']} ({clean_date}) (score: {source['score']})")
            else:
                print(f"Error: {result['response']}")
        
        print("\nRAG system test completed!")
        
    except Exception as e:
        print(f"RAG system test failed: {e}")

if __name__ == "__main__":
    test_rag_system()