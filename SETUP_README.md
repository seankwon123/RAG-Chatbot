
# STOP AND START all three services
### Stop all containers with sudo
sudo docker stop $(sudo docker ps -q)

### Remove all containers (any stuck in 'Created' state)
sudo docker rm $(sudo docker ps -aq)

### Check ports:
 - sudo lsof -i :5678
 - sudo lsof -i :8080
 - sudo lsof -i :5432


### Start fresh
cd ~/Documents/Projects/RAG-Chatbot/n8n-getting-started
sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml -f Adminer/docker-compose.yml up -d



# STEPS TO SET UP CONTAINERS
- cd ~/Documents/Projects/RAG-Chatbot/n8n-getting-started
- sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml -f Adminer/docker-compose.yml up -d

- Connect to project with: python3 config.py

- data_processing.py will:
  - Download the sentence-transformers model (first time only)
  - Fetch your 445 articles from PostgreSQL
  - Test embedding generation on a sample article
  - Show you the dimensions and confirm everything works


# Deployment
 - sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml -f Adminer/docker-compose.yml up -d
   - -f Adminer/... optional, it is just GUI for PostgreSQL




# WORKFLOW RUN WITH RAG CHATBOT

## n8n Workflow setup and execution
Go to n8n-getting-started if needed and set up.
Once set, use this
 - "sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml -f Adminer/docker-compose.yml up -d"
 - -f Adminer... is optional (UI for PostgreSQL)

Run workflow by dropping json of workflow file and setting up credentials
 - Host: pg-n8n
 - Database: n8n
 - User: n8n
 - Password: password
 - Port: 5432


## Run chatbot

 - FIRST: install requirements.txt 

 Commands to run each part of RAG + chatbot:
 - python models.py - requires ollama's qwen2.5 (ollama pull qwen2.5)
 - python qdrant_manager.py
 - python embedding_manager.py - requires all-minilm embedding model
 - python rag_service.py
 - streamlit run chatbot.py





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
  - "Can you show me all articles about DevOps"
  - "What is the latest blog post about?" 

8/26:
Update:
- making version 2 of chatbot, working from ground up to set up much better info retreival and better answers


9/10:
Wiring up LangChain into the rag_service

11/15:
Attempting to change from using n8n workflows to pure python scripts and SQL. 