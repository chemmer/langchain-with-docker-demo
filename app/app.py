from langchain_community.llms import Ollama
from git import Repo
import time
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


class LogHandler(BaseCallbackHandler):
    def on_retriever_end(
        self,
        documents,
        *,
        run_id,
        parent_run_id,
        **kwargs: Any,
    ):
        print("Retrieved {} documents".format(len(documents)))
        print("First document {}".format(documents[0]))

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs: Any,
    ):
        print("Starting LLM")
        print("System prompt: {}".format(prompts[0]))


handler = LogHandler()
config = {"callbacks": [handler]}

print("Starting! Waiting 5 seconds for the ollama container to be ready")
time.sleep(5)
print("Let's load the ollama container.")
MODEL_NAME_LLM = "llama3"
MODEL_NAME_RETRIEVER = "all-minilm"
model_base_url = "http://ollama-container:11434"
llm = Ollama(model=MODEL_NAME_LLM, base_url=model_base_url, verbose=True)

REPO_NAME = "https://github.com/langchain-ai/langchain.git"
repo_path = "/app/repo"
APP_DIR = "/libs/core/langchain_core"
persist_dir = "/app/chroma_db"
PROGRAMMING_SUFFIX = ".py"
PROGRAMMING_LANGUAGE = Language.PYTHON
if not Path(repo_path + APP_DIR).exists():
    print("Repo doesn't exist yet, cloning...")
    repo = Repo.clone_from(REPO_NAME, to_path=repo_path)

# Create Embeddings if not yet exists
if not Path(persist_dir, "chroma.sqlite3").exists():
    print("No embeddings yet, loading...")
    loader = GenericLoader.from_filesystem(
        repo_path + APP_DIR,
        glob="**/*",
        suffixes=[PROGRAMMING_SUFFIX],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=PROGRAMMING_LANGUAGE, parser_threshold=500),
    )
    documents = loader.load()
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=PROGRAMMING_LANGUAGE, chunk_size=2000, chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)
    print("Amount of chunks: " + str(len(texts)))
    db2 = Chroma.from_documents(
        texts,
        OllamaEmbeddings(model=MODEL_NAME_RETRIEVER, base_url=model_base_url),
        persist_directory=persist_dir,
        collection_name="v_db",
    )
    print("getting documents done")
    db2.persist()
    print("persisting done")

db3 = Chroma(
    collection_name="v_db",
    persist_directory=persist_dir,
    embedding_function=OllamaEmbeddings(
        model=MODEL_NAME_RETRIEVER, base_url=model_base_url
    ),
)
print("Amount of documents in storage: " + str(len(db3.get()["documents"])))
print("Retrieving from disk done")

retriever = db3.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)


document_chain = create_stuff_documents_chain(llm, prompt)
print("Setup complete, let's ask some questions.")
qa = create_retrieval_chain(retriever_chain, document_chain)

question = (
    "Only answer using the provided context. How does the FileCallbackHandler work?"
)
result = qa.invoke({"input": question}, config=config)
print(result["answer"])
