# Using Langchain with Docker
## Introduction


## Installation instructions
In order to install this repo, you need to have docker installed.

1. Build and run the container
```
docker compose up --build
```
You can run it in detached mode if you would like to pull the models in the same container
```
docker compose up --build -d
```

2. Download the models you are interested in. For this demo we chose llama3 for text generation and all-minilm for the embeddings
Docker exec to download model in ollama container 
```
docker exec -it langchain-with-docker-demo-ollama-container-1 ollama pull llama3
docker exec -it langchain-with-docker-demo-ollama-container-1 ollama pull all-minilm

```

3. Change the variables in app.py to your liking

The models you would like to use. Need to be supported by Ollama

- MODEL_NAME_LLM = The text generation model
- MODEL_NAME_RETRIEVER = The embedding model

Specifics about the repository you would like to query.
- REPO_NAME = The repository you would like to clone. If it is private, you would need to add authentication
- APP_DIR = The path to where the libraries that you would like to include in the vector database are
- PROGRAMMING_LANGUAGE = Which Programming language the app is written in. Needs to be supported by the tree_sitter Language Parser library


4. Acess the logs to see the output
If running the container in detached mode, you can access the logs of the container like so:
```
docker compose logs langchain-with-docker-demo-langchain-app-1 -f
```