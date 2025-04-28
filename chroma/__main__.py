import os
import uuid
from pathlib import Path
from typing import Set, List

import fire
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from google import genai
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from chroma import PATH as CHROMA_PATH, generate_chroma_db
from documents import PATH as DOCUMENTS_PATH, read_csv, from_pandas_to_list

# — DEFAULT CONFIGURATION —
DEFAULT_CHUNK_SIZES: Set[int] = {128, 256, 512, 1024}
DEFAULT_OVERLAP_RATIOS: Set[float] = {0.1, 0.2, 0.3, 0.4, 0.5}
API_KEY: str = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
#embedding_lm = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
embedding_lm = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

def main(
    document: Path = DOCUMENTS_PATH / "data-generated.csv",
    chunk_sizes: Set[int] = DEFAULT_CHUNK_SIZES,
    overlap_ratios: Set[float] = DEFAULT_OVERLAP_RATIOS,
    base_path: Path = CHROMA_PATH
) -> None:
    # Ensure base directory exists
    base_path.mkdir(parents=True, exist_ok=True)

    # Load CSV and convert to list of lists of strings
    df = read_csv(document)
    docs: List[str] = from_pandas_to_list(df)

    for chunk_size in sorted(chunk_sizes):
        for overlap in sorted(overlap_ratios):
            generate_chroma_db(
                docs,
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_lm=embedding_lm,
                base_path=base_path,
                force_recreate=False,
                #db_name_base="nomadic"
            )

    print("✅ All embeddings have been created and saved.")


if __name__ == "__main__":
    fire.Fire(main)
