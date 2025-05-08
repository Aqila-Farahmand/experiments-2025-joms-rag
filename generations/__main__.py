# set of rag under test + model under test
import csv
import os

import pandas as pd
import logging
from langchain_core.language_models import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from documents import PATH as DATA_PATH
from generations import PATH as GENERATIONS_PATH, MetaInfo
from generations import RagUnderTest
from generations.generate_replies import generate_replies_from_rag
from rag.hybrid_retriever import generate_hybrid_rag
from rag.vector_rerank_retriever import generate_vector_rerank_rag
from rag.vector_store_retriever import generate_vector_store_rag
from utils import name_from_llm

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
embeddings = {
    "nomic": HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
    "mxbai": HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True),
}

llms = {
    "qwen3-0.6b": Ollama(model="qwen3:0.6b", request_timeout=60000),
    "qwen3-1.7b": Ollama(model="qwen3:1.7b", request_timeout=60000),
    "qwen3-4b": Ollama(model="qwen3:4b", request_timeout=60000),
    "qwen3-8b": Ollama(model="qwen3:8b", request_timeout=60000),
    "gemma3-1b": Ollama(model="gemma3:1b", request_timeout=60000),
    "gemma3-4b": Ollama(model="gemma3:4b", request_timeout=60000),
    "gemma3-12b": Ollama(model="gemma3:12b", request_timeout=60000),
    "medllama3-v20": Ollama(model="ahmgam/medllama3-v20:latest", request_timeout=60000),
    "llama3.2-3b": Ollama(model="llama3.2:latest", request_timeout=60000),
    "llama3.2-1b": Ollama(model="llama3.2:1b", request_timeout=60000),
    "deepseek-r1-1.5b": Ollama(model="deepseek-r1:1.5b", request_timeout=60000),
    "deepseek-r1-7b": Ollama(model="deepseek-r1:latest", request_timeout=60000),
}

# launch a subcommand which stop each model
for llm in llms:
    # ollama stop <model_name>, use cmd
    if isinstance(llm, Ollama):
        logging.info(f"Stopping {llm.model}")
        os.system(f"ollama stop {llm.model}")

# adapt ollama to have model_name
data_under_test = pd.read_csv(DATA_PATH / "test.csv")[:1]  # remove :5 for the full dataset
base = DATA_PATH / "data-generated.csv"

RETRIEVES = {
    "vector_store": generate_vector_store_rag,
    "vector_rerank": generate_vector_rerank_rag,
    "hybrid": generate_hybrid_rag
}


def generate_rags_for_llm(llm: str, embedding: str) -> list[RagUnderTest]:
    rags: list[RagUnderTest] = []
    for retriever_name, retriever_fn in RETRIEVES.items():
        logging.info(f"Generating {retriever_name} for {llm} with {embedding}")

        rag, index = retriever_fn(
            csv_path=str(DATA_PATH / "data-generated.csv"),
            chunk_size=256,
            overlap_ratio=0.5,
            embedding_model=embeddings[embedding],
            llm=llms[llm],
            k=3,
            alpha=0.5,
            persist=True,
            collection_name=embedding,
        )

        rags.append(
            RagUnderTest(
                rag=rag,
                metainfo=MetaInfo(
                    retriever=retriever_name,
                    embedder=embedding,
                    llm=llm,
                )
            )
        )
    return rags


def generate_response_and_store(where: str, rag_under_test: RagUnderTest) -> None:
    # Generate responses
    os.makedirs(where, exist_ok=True)
# CSV path
    csv_path = os.path.join(where, f"{rag_under_test.tag()}.csv")
    # Check if the file already exists
    if os.path.exists(csv_path):
        logging.info(f"File {csv_path} already exists. Skipping generation.")
        return

    responses = generate_replies_from_rag(rag_under_test.rag, data_under_test)

    if isinstance(llm, Ollama):
        logging.info(f"Stopping {llm.model}")
        os.system(f"ollama stop {llm.model}")

    retriever, embedder_model, llm_model = rag_under_test.metainfo.retriever, rag_under_test.metainfo.embedder, rag_under_test.metainfo.llm

    # Write to CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["embedder", "retriever", "LLM", "question", "response"])
        for item in responses:
            writer.writerow([
                embedder_model,
                retriever,
                llm_model,
                item["question"],
                item["response"]
            ])


# iterate over the llms and embedding
for embedding_model in embeddings:
    for llm_model in llms:
        logging.info(f"Generating responses for {llm_model} with {embedding_model}")
        rags = generate_rags_for_llm(llm_model, embedding_model)
        for rag in rags:
            logging.info(f"Generating responses for {rag.tag()}")
            # generate response and store
            cache_path = GENERATIONS_PATH / "cache"
            generate_response_and_store(cache_path, rag)
            logging.info(f"Generated responses for {rag.tag()} and stored in {cache_path}")


if __name__ == "__main__":
    # Example usage
    # generate_response_and_store(GENERATIONS_PATH, rag)
    pass
