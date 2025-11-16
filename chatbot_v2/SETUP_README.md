# WORKFLOW RUN WITH RAG CHATBOT

## n8n Workflow setup and execution
Go to n8n-getting-started if needed and set up.
Once set, use this
 - sudo docker-compose -f docker-compose.yml -f Postgres/docker-compose.yml -f Qdrant/docker-compose.yml - Adminer docker-compose.yml up -d

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


