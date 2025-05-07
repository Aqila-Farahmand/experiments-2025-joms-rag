import asyncio
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict

import chromadb
import fire
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from analysis import PATH as ANALYSIS_PATH
from chroma import PATH as BASE_DB_PATH
from documents import PATH as DOCUMENTS_PATH

# Evaluation parameters
CHUNK_SIZES = [128, 256, 512, 1024]
OVERLAP_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Set up Google Gemini API Key and configure genai
#API_KEY: str = os.getenv("GOOGLE_API_KEY")
#client = genai.Client(api_key=API_KEY)


# Choose the LLM for chunk size and overlap evaluation
# llm = Ollama(model="llama3.1", request_timeout=120.0)ù
llm = GoogleGenAI(model_name="models/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

async def evaluate_metrics(query_engine, questions: List[str]) -> Tuple[float, float, float]:
    """
    Compute average latency, faithfulness and relevancy over a list of queries.
    """
    total_time = total_faith = total_rel = 0.0
    total_pass_time = 0.0
    n = len(questions)
    print(f"Evaluating {n} queries...")
    replies = []
    for q in questions[:n]:
        print(".", end="", flush=True)
        start = time.time()
        replies.append(query_engine.query(q))
        total_time += time.time() - start
    batch_scores = await asyncio.gather(
        *[
            asyncio.gather(
                faithfulness_evaluator.aevaluate_response(response=resp),
                relevancy_evaluator.aevaluate_response(query=q, response=resp)
            )
            for q, resp in zip(questions, replies)
        ]
    )
    batch_faith_scores = [ scores[0] for scores in batch_scores ]
    batch_rel_scores = [ scores[1] for scores in batch_scores ]

    total_faith = sum(float(faith.score) for faith in batch_faith_scores)
    total_rel = sum(float(rel.score) for rel in batch_rel_scores)

    print(f"Total time: {total_pass_time:.2f}s")
    print(f"Average time: {total_time / n:.2f}s")
    print(f"Average faithfulness: {total_faith / n:.2f}")
    print(f"Average relevancy: {total_rel / n:.2f}")
    return total_time / n, total_faith / n, total_rel / n


async def main(
    csv_path: str = str(DOCUMENTS_PATH / "data-generated.csv"),
    question_col: str = "Sentence",
    answer_col: str = "Response",
    db_base_path: str = str(BASE_DB_PATH),
) -> None:
    """
    For each Chroma DB (chunk_size + overlap), build an in-memory index
    from the existing vectors and run evaluation.
    """
    # 1. Read CSV and prepare questions & documents
    df = pd.read_csv(csv_path)
    questions = df[question_col].astype(str).tolist()
    documents = [
        Document(text=txt, doc_id=str(i))
        for i, txt in enumerate(df[answer_col].astype(str).tolist())
    ]

    # 2. Configure embedding
    embeddings = {
        #"gemini": GoogleGenAIEmbedding(model_name="models/text-embedding-004"),
        "nomic": HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
        "mxbai": HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True),
    }
    for name, embedding in embeddings.items():
        result_file = ANALYSIS_PATH / f"chunks_evaluation_{name}.csv"
        if result_file.exists():
            print(f"File {result_file} already exists. Skipping...")
            continue

        results: List[Dict[str, float]] = []

        # 3. Loop over chunk sizes and overlap ratios
        for cs in CHUNK_SIZES:
            for ov in OVERLAP_RATIOS:
                suffix = f"{name}_chunk_size_{cs}_overlapping_{int(ov * 100)}"
                db_folder = Path(db_base_path) / suffix

                # 4. Initialize persistent ChromaDB client & collection
                client = chromadb.PersistentClient(path=str(db_folder))
                collection = client.get_or_create_collection(name=suffix)

                # 5. Build ChromaVectorStore from existing collection & embeddings
                vector_store = ChromaVectorStore(
                    chroma_collection=collection,
                    embedding=embedding
                )

                # 6. Create in-memory index directly from the vector store
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    embed_model=embedding,
                )  # :contentReference[oaicite:0]{index=0}

                # 7. Prepare query engine and evaluate
                query_engine = index.as_query_engine(llm=llm)
                print(f"Evaluating chunk_size={cs}, overlap={ov}")
                avg_time, avg_faith, avg_rel = await evaluate_metrics(query_engine, questions)

                results.append({
                    "chunk_size":       cs,
                    "overlap":          ov,
                    "avg_time":         avg_time,
                    "avg_faithfulness": avg_faith,
                    "avg_relevancy":    avg_rel,
                })

        # 8. Save and print metrics
        df_metrics = pd.DataFrame(results)
        print(df_metrics)
        df_metrics.to_csv(result_file, index=False)


if __name__ == "__main__":
    fire.Fire(main)
