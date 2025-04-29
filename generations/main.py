# set of rag under test + model under test
import json
import os
import pickle

import pandas as pd
from langchain_core.language_models import LLM

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama

from documents import PATH as DATA_PATH
from generations import PATH as GENERATIONS_PATH
from generations import RagUnderTest
from generations.generate_replies import generate_replies_from_rag
from rag.vector_store_retriever import generate_vector_store_rag
from rag.vector_rerank_retriever import generate_vector_rerank_rag
from rag.hybrid_retriever import generate_hybrid_rag
# from analysis import CHROMA_COLLECTION_NAME

embedding = [
    OllamaEmbedding(model_name="mxbai-embed-large", base_url="http://clusters.almaai.unibo.it:11434/"),
    GoogleGenAIEmbedding()
]

llms = [
    Ollama(model="qwen2.5:1.5b", base_url="http://clusters.almaai.unibo.it:11434/"),
    GoogleGenAI()
    # others
]

data_under_test = pd.read_csv(DATA_PATH / "test.csv")[:5]  # remove :5 for the full dataset
base = DATA_PATH / "data-generated.csv"

RETRIEVES = {
    "vector_store": generate_vector_store_rag,
    "vector_rerank": generate_vector_rerank_rag,
    "hybrid": generate_hybrid_rag
}


def generate_rags_for_llm(llm: LLM, embedding: BaseEmbedding) -> list[RagUnderTest]:
    rags: list[RagUnderTest] = []

    for retriever_name, retriever_fn in RETRIEVES.items():
        print(f"Generating {retriever_name} for {llm.model} with {embedding.model_name}")

        rag, index = retriever_fn(
            csv_path=str(DATA_PATH / "data-generated.csv"),
            chunk_size=256,
            overlap_ratio=0.5,
            embedding_model=embedding,
            llm=llm,
            k=3,
            alpha=0.5,
            persist=True
        )

        print(f"Number of docs indexed: {len(index.docstore.docs)}")

        rags.append(
            RagUnderTest(
                rag=rag,
                tag=f"{retriever_name}__{llm.model}__{embedding.model_name}"
            )
        )

    return rags


def generate_response_and_store(where: str, rag_under_test: RagUnderTest) -> None:
    # paths where you will store outputs
    json_path = os.path.join(where, f"{rag_under_test.tag}.json")
    pickle_path = os.path.join(where, f"{rag_under_test.tag}.pkl")

    # if already exists, skip
    # if os.path.exists(json_path) and os.path.exists(pickle_path):
        # print(f"Skipping {rag_under_test.tag}, already exists.")
        # return

    # otherwise, generate responses
    responses = generate_replies_from_rag(rag_under_test.rag, data_under_test)

    # check if the folder exists, if not create
    if not os.path.exists(where):
        os.makedirs(where)

    # store "not raw" with json
    with open(json_path, "w") as f:
        json.dump([response.response for response in responses], f, indent=2)

    # store raw with pickle
    with open(pickle_path, "wb") as f:
        pickle.dump(responses, f)


# iterate over the llms and embedding
for embedding_model in embedding:
    for llm in llms:
        print(f"Generating responses for {llm.model} with {embedding_model.model_name}")
        # generate rags
        rags = generate_rags_for_llm(llm, embedding_model)
        # iterate over the rags
        for rag in rags:
            print(f"Generating responses for {rag.tag}")
            # generate response and store
            generate_response_and_store(GENERATIONS_PATH, rag)
            print(f"Generated responses for {rag.tag} and stored in {GENERATIONS_PATH}")
