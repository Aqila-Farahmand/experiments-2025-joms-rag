# set of rag under test + model under test
import csv
import logging
import os
import pickle
from abc import ABC
from typing import Any

import fire
from pydantic import BaseModel, Field

import pandas as pd
from llama_index.core import Response
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama

from documents import PATH as DATA_PATH
from generations import PATH as GENERATIONS_PATH, MetaInfo
from generations import RagUnderTest
from generations.generate_replies import generate_replies_from_rag
from rag.hybrid_retriever import generate_hybrid_rag
from rag.vector_rerank_retriever import generate_vector_rerank_rag
from rag.vector_store_retriever import generate_vector_store_rag


CACHE_PATH = GENERATIONS_PATH / "cache"
EMBEDDINGS = {
    "nomic": HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
    #"mxbai": HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True),
}
LLMs = {
    #"qwen3-0.6b": Ollama(model="qwen3:0.6b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"qwen3-1.7b": Ollama(model="qwen3:1.7b", base_url="http://clusters.almaai.unibo.it:11434/",  request_timeout=60000),
    #"qwen3-4b": Ollama(model="qwen3:4b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"qwen3-8b": Ollama(model="qwen3:8b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"gemma3-1b": Ollama(model="gemma3:1b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"gemma3-4b": Ollama(model="gemma3:4b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"gemma3-12b": Ollama(model="gemma3:12b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"medllama3-v20": Ollama(model="ahmgam/medllama3-v20:latest", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    "llama3.2-3b": Ollama(model="llama3.2:3b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    "llama3.2-1b": Ollama(model="llama3.2:1b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"deepseek-r1-1.5b": Ollama(model="deepseek-r1:1.5b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    #"deepseek-r1-7b": Ollama(model="deepseek-r1:latest", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000),
    "gemini-2.0": GoogleGenAI(model_name="models/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
}
RETRIEVES = {
    #"vector_store": generate_vector_store_rag,
    "vector_rerank": generate_vector_rerank_rag,
    "hybrid": generate_hybrid_rag
}


# launch a subcommand which stop each model
for llm in LLMs:
    # ollama stop <model_name>, use cmd
    if isinstance(llm, Ollama):
        logging.info(f"Stopping {llm.model}")
        os.system(f"ollama stop {llm.model}")

# adapt ollama to have model_name
data_under_test = pd.read_csv(DATA_PATH / "test_generated_it.csv")  # [:10]
base = DATA_PATH / "data_raw.csv"


def full_prompt():
    # load data in DATA_PATH as pd
    df = pd.read_csv(DATA_PATH / "train.csv")
    # convert to: Q: {question} A: {answer}
    df["text"] = df.apply(lambda x: f"Q: {x['Sentence']} A: {x['Response']}", axis=1)
    # create a prompt with all of these example
    prompt = "\n".join(df["text"].tolist())
    # create a prompt template
    prompt_template = RichPromptTemplate(
        """
            Sei un medico esperto nell'ipertensione e nella salute cardiovascolare. 
            Aiuta a rispondere a questa domanda (in modo empatico).
            Cerca di rispondere in modo simile a questi esempi:"""
            + prompt + "\n La domanda a cui devi rispondere (in modo conciso) è: {{ question }} "

    )
    return prompt_template


PROMPTS = {
    "role_playing":
        RichPromptTemplate("""
        Sei un medico esperto nell'ipertensione e nella salute cardiovascolare. 
        Aiuta a rispondere a questa domanda (in modo empatico e conciso): {{ question }}
        """),
    "full": full_prompt()
}


def generate_rags_for_llm(llm: str, embedding: str) -> list[RagUnderTest]:
    rags: list[RagUnderTest] = []
    for retriever_name, retriever_fn in RETRIEVES.items():
        logging.info(f"Generating {retriever_name} for {llm} with {embedding}")

        rag, index = retriever_fn(
            csv_path=str(DATA_PATH / "data_raw.csv"),
            chunk_size=256,
            overlap_ratio=0.5,
            embedding_model=EMBEDDINGS[embedding],
            llm=LLMs[llm],
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


def generate_rag_response_and_store(where: str, rag_under_test: RagUnderTest) -> None:
    # Generate responses
    os.makedirs(where, exist_ok=True)
    # CSV path
    csv_path = os.path.join(where, f"{rag_under_test.tag()}.csv")
    # Check if the file already exists
    if os.path.exists(csv_path):
        logging.info(f"File {csv_path} already exists. Skipping generation.")
        return

    responses = generate_replies_from_rag(rag_under_test.rag, data_under_test)

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
                item["response"].response
            ])

    # Save responses to a pickle file
    pickle_path = os.path.join(where, f"{rag_under_test.tag()}.pkl")
    to_store = {
        "responses": responses,
        "retriever": retriever,
        "embedder": embedder_model,
        "llm": llm_model
    }
    with open(pickle_path, mode="wb") as pkl_file:
        pickle.dump(to_store, pkl_file)


def generate_llm_response_and_store(where: str, llm: str, prompt: RichPromptTemplate, prompt_kind: str) -> None:
    # Generate responses
    os.makedirs(where, exist_ok=True)
    # CSV path
    csv_path = os.path.join(where, f"prompt__{prompt_kind}__{llm}.csv")
    # Check if the file already exists
    if os.path.exists(csv_path):
        logging.info(f"File {csv_path} already exists. Skipping generation.")
        return

    responses = []
    for i, question in enumerate(data_under_test["Sentence"]):
        formatted_prompt = prompt.format(
            question=question
        )
        response_text = LLMs[llm].complete(formatted_prompt).text
        response_text = response_text.split("</think>")[-1]
        responses.append({ "question": question, "response": Response(response_text)})

    # Write to CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["LLM", "question", "response"])
        for item in responses:
            writer.writerow([
                llm,
                item["question"],
                item["response"]
            ])
    # store as pickle
    pickle_path = os.path.join(where, f"prompt__{prompt_kind}__{llm}.pkl")
    to_store = {
        "responses": responses,
        "llm": llm,
        "prompt": prompt_kind
    }
    with open(pickle_path, mode="wb") as pkl_file:
        pickle.dump(to_store, pkl_file)


def main():
    for llm_model in LLMs:
        for embedding_model in EMBEDDINGS:
            logging.info(f"Generating responses for {llm_model} with {embedding_model}")
            rags = generate_rags_for_llm(llm_model, embedding_model)
            for rag in rags:
                logging.info(f"Generating responses for {rag.tag()}")
                # generate response and store
                generate_rag_response_and_store(CACHE_PATH, rag)
                logging.info(f"Generated responses for {rag.tag()} and stored in {CACHE_PATH}")
        for prompt in PROMPTS:
            logging.info(f"Generating responses for {llm_model} and prompt {prompt}")
            # generate response and store
            generate_llm_response_and_store(CACHE_PATH, llm_model, PROMPTS[prompt], prompt)
        if isinstance(LLMs[llm_model], Ollama):
            logging.info(f"Stopping {llm_model}")
            os.system(f"ollama stop {LLMs[llm_model].model}")


if __name__ == "__main__":
    fire.Fire(main)
