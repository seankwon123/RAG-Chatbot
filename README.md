Original Setup Workflow:
- Fetch blog content - HTTP request
- Parse article content - HTML extract
- Clean and structure data - Code Node
- Store articles - Database Node
- Automate regular fetching - Cron/Schedule

Schedule Trigger → HTTP Request (RSS/API) → Parse Articles → 
Clean Content → Check for Duplicates → Store in Database

n8n - Workflow automation (data ingestion)
Postgres - Article storage and metadata
Qdrant - Vector database for embeddings/search
Docker Compose - Container orchestration

11/15: Update - Using python scripts to replace "n8n" workflows.
 - Fetch blog content 
 - Parse article content
 - Clean and structure data
 - Store articles to PostgreSQL service
 - Build Index for Qdrant data vectorization

# RAG Architecture
Using LangChain:
 - PostgreSQL database of articles (blog entries)
 - Qdrant vectorization and nearest neighbor analysis for RAG
 - Send prompt with context from Qdrant context to Ollama model
 - Produce useful, fact-based output for user about information from the blog.

## RAG Architecture Details
Articles in Postgres -> Embedding Generation -> Store in Qdrant -> Query Interface

- Embedding model: OpenAI's ada-002
- Vector DB: Qdrant
- Query API: FastAPI with LangChain for RAG
- LLM: OpenAI Ollama

Future Steps: Docker containers on AWS/GCP if you need public access

## Code layout (excluding article extraction workflow)
project/
├── n8n-getting-started/      # where you set up docker containers 
├── RAG-Chatbot/              # The Chatbot 
    ├── data_processing.py      # Article processing, local embeddings
    ├── qdrant_manager.py       # Qdrant operations
    ├── rag_system.py           # RAG logic with OpenAI chat
    ├── chatbot.py              # Streamlit interface
    ├── populate_qdrant.py      # One-time setup script
    └── requirements.txt



# FULL RUN OF THE WHOLE APPLICATION AND RAG SYSTEM:

## Start up RAG
 - Check if Ollama is running.
  ```
  ollama list
  ```
  - This should list all-minilm, nomic-embed-text, qwen2.5, and llama3.1 
  - if not,
  ```
  ollama serve
  ```
 - Make sure all containers are up (Postgres + Qdrant)
  ```
  cd .../n8n-getting-started
  sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml up -d
  ```
 - Verify
  ```
  sudo docker ps
  ```
 - Port Checks
  ```
  sudo lsof -i :5432   # Postgres
  sudo lsof -i :6333   # Qdrant
  sudo lsof -i :8080   # Adminer if used
  ```
 - Set up Data, index into Qdrant, get rag_service ready
  ```
  cd .../RAG-Chatbot
  source venv/bin/activate
  ```
  - Optional: confirm "articles" exists in Postgres container (might need to install psql)
  ``` 
  # articles exists
  PGPASSWORD=password psql \
      -h localhost \
      -p 5432 \
      -U n8n \
      -d n8n \
      -c "\d articles"

  # articles total count
  PGPASSWORD=password psql \
  -h localhost \
  -p 5432 \
  -U n8n \
  -d n8n \
  -c "SELECT COUNT(*) FROM articles;"

  ```
 - If articles exist and total count is 452, SKIP SCRAPE
 - Scrape and upsert into Postgres
  ```
  python3 scrape_articles.py
  ```
 
 - Check Qdrant collection exists
 ```
 # Check exists
 curl http://localhost:6333/collections
 
 # Expected response
 {
  "collections": [
    {
      "name": "articles_collection"
    }
  ]
 }

 # If exists, check vector points count: Should be - "points_count": 452
 curl http://localhost:6333/collections/articles_collection
 ```

 - If Qdrant vectors exist, SKIP INDEX
 - Build Qdrant index from Postgres
  ```
  python3 build_index.py
  ```
  - This should:
    - fetch all articles from Postgres
    - embed them with all-minilm
    - create or recreate the Qdrant collection articles_collection
    - push all vectors in batches
 - After those two...
   - Postgres has full article data
   - Qdrant has up-to-date embeddings
   - rag_service.py has everything needed at runtime

  - Optional: sanity check of rag_service
  ```
  python3 rag_service.py --question "What kind of tools are recommended for E2E testing?"
  ```

## Start up the app:
 - start venv:
  ```
  source venv/bin/activate
  ```
 - start the app:
  ```
  streamlit run streamlit_app.py
  ```
