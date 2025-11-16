# n8n Workflow:

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

(local deployment setup)

# 'n8n-getting-started' setup

## install docker
## If necessary, install docker, and setup the services
sudo snap install docker

## Install Docker
sudo apt install docker.io

## Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

## Verify Docker service is running
sudo systemctl status docker

## Docker "unable to get image" issue
### Add your user to the docker group
sudo usermod -aG docker $USER

### Apply the new group membership (or you can log out and back in)
newgrp docker

### Verify Docker works without sudo
docker --version


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

