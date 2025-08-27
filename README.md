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

# n8n
## Docker setup
## install docker
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


# STOP AND START all three services
### Stop all containers with sudo
sudo docker stop $(sudo docker ps -q)

### Start fresh
cd ~/Documents/Projects/Bitovi-RAG/n8n-getting-started
sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml -f Adminer/docker-compose.yml up -d


### Verify all 3 are running
sudo docker ps

### Find the network name
sudo docker network ls | grep n8n


- Adminer SQL articles table



# RAG Architecture


# Deployment


# Overall Workflow Outline
## Step 1: Setting up n8n Workflow for Bitovi Blog
- Set up environment and getting it running


## Step 2: RAG Atchitecture 
Articles in Postgres -> Embedding Generation -> Store in Qdrant -> Query Interface

- Embedding model: OpenAI's ada-002 or open-source alt?
- Vector DB: Qdrant
- Query API: FastAPI with LangChain for RAG
- LLM: OpenAI GPT-4 or Anthropic Claude

Hosting: local setup is perfect for development. 

Docker containers on AWS/GCP if you need public access

## Code layout (excluding article extraction workflow)
project/
├── config.py              # API keys, connections
├── data_processing.py      # Article processing, local embeddings
├── qdrant_manager.py       # Qdrant operations
├── rag_system.py          # RAG logic with OpenAI chat
├── chatbot.py             # Streamlit interface
├── populate_qdrant.py     # One-time setup script
└── requirements.txt



# STEPS TO SET UP CONTAINERS
- cd ~/Documents/Projects/Bitovi-RAG/n8n-getting-started
- sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml -f Adminer/docker-compose.yml up -d

- Connect to project with: python3 config.py

- data_processing.py will:
  - Download the sentence-transformers model (first time only)
  - Fetch your 445 articles from PostgreSQL
  - Test embedding generation on a sample article
  - Show you the dimensions and confirm everything works



# Notes:

8/24:

Update:
- n8n workflow: url, url_canonical, published_date, author all NOT working
- Once database is filled through n8n workflow, continue towards using the database for RAG.
- Qdrant, and LLM -> create the RAG chatbot


8/25:
Update: 
- articles sucessfully into database
- moving onto starting vector storage with Qdrant

- Linking containers to project with .env file through port access 

- Qdrant is working,
- RAG is working, 
- Chatbot passes tets, but has issues with:
  - "Can you show me all Bitovi articles about DevOps"
  - "What is Bitovi's latest blog post about?" 

8/26:
Update:
- making version 2 of chatbot, working from ground up to set up much better info retreival and better answers
