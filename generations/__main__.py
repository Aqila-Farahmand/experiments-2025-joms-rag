# set of rag under test + model under test
import csv
import os

import pandas as pd
from langchain_core.language_models import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.prompts import RichPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM

from documents import PATH as DATA_PATH
from generations import PATH as GENERATIONS_PATH
from generations import RagUnderTest
from generations.generate_replies import generate_replies_from_rag
from rag.hybrid_retriever import generate_hybrid_rag
from rag.vector_rerank_retriever import generate_vector_rerank_rag
from rag.vector_store_retriever import generate_vector_store_rag
from utils import name_from_llm

# from analysis import CHROMA_COLLECTION_NAME

embedding = {
    # OllamaEmbedding(model_name="mxbai-embed-large", base_url="http://clusters.almaai.unibo.it:11434/"),
    # GoogleGenAIEmbedding()
    "nomic": HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
    "mxbai": HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True),
}

llms = [
    #HuggingFaceLLM(model_name="Qwen/Qwen3-0.6B"),
    GoogleGenAI(),
    Ollama(model="qwen3:0.6b"),
    Ollama(model="qwen3:1.7b"),
    Ollama(model="qwen3:4b"),
    Ollama(model="qwen3:8b"),
    Ollama(model="gemma3:1b"),
    Ollama(model="gemma3:4b"),
    Ollama(model="gemma3:12b"),
    Ollama(model="ahmgam/medllama3-v20:latest"),
    Ollama(model="llama3.2:latest"),
    Ollama(model="llama3.2:1b"),
    Ollama(model="deepseek-r1:1.5b"),
    Ollama(model="deepseek-r1:latest")
    # ollama(model_name="mixtral:latest", base_url="http://clusters.almaai.unibo.it:11434/")
]

# adapt ollama to have model_name
data_under_test = pd.read_csv(DATA_PATH / "test.csv")  # remove :5 for the full dataset
base = DATA_PATH / "data-generated.csv"

RETRIEVES = {
    "vector_store": generate_vector_store_rag,
    "vector_rerank": generate_vector_rerank_rag,
    "hybrid": generate_hybrid_rag
}


def generate_rags_for_llm(llm: LLM, embedding: BaseEmbedding, collection_base_name: str) -> list[RagUnderTest]:
    rags: list[RagUnderTest] = []
    for retriever_name, retriever_fn in RETRIEVES.items():
        print(f"Generating {retriever_name} for {name_from_llm(llm)} with {embedding.model_name}")

        rag, index = retriever_fn(
            csv_path=str(DATA_PATH / "data-generated.csv"),
            chunk_size=256,
            overlap_ratio=0.5,
            embedding_model=embedding,
            llm=llm,
            k=3,
            alpha=0.5,
            persist=True,
            collection_name=collection_base_name,
        )
        filtered_name = embedding.model_name.replace("/", "_")
        rags.append(
            RagUnderTest(
                rag=rag,
                tag=f"{retriever_name}__{name_from_llm(llm)}__{filtered_name}"
            )
        )
    return rags


def generate_response_and_store(where: str, rag_under_test: RagUnderTest) -> None:
    # Generate responses
    responses = generate_replies_from_rag(rag_under_test.rag, data_under_test)

    # Ensure the output directory exists
    os.makedirs(where, exist_ok=True)

    # CSV path
    csv_path = os.path.join(where, f"{rag_under_test.tag}.csv")

    # Extract metadata from tag
    try:
        retriever, llm_model, embedder_model = rag_under_test.tag.split("__")
    except ValueError:
        retriever = llm_model = embedder_model = "unknown"

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
for embedding_model in embedding:
    for llm in llms:
        print(f"Generating responses for {name_from_llm(llm)} with {embedding_model}")
        # generate rags
        # rags = generate_rags_for_llm(llm, embedding_model)

        # Generate RAGs with prompt injected
        rags = generate_rags_for_llm(llm, embedding[embedding_model], embedding_model)

        # iterate over the rags
        for rag in rags:
            print(f"Generating responses for {rag.tag}")
            # generate response and store
            cache_path = GENERATIONS_PATH / "cache"
            generate_response_and_store(cache_path, rag)
            print(f"Generated responses for {rag.tag} and stored in {cache_path}")


if __name__ == "__main__":
    # Example usage
    # generate_response_and_store(GENERATIONS_PATH, rag)
    pass
