import fire
import os
import uuid
from pathlib import Path
from typing import Set, List
import google.generativeai as genai
from chromadb import PersistentClient
from chroma import PATH as CHROMA_PATH
from documents import PATH as DOCUMENTS_PATH, read_csv, from_pandas_to_list

# — DEFAULT CONFIGURATION —
DEFAULT_CHUNK_SIZES: Set[int] = {128, 256, 512, 1024}
DEFAULT_OVERLAP_RATIOS: Set[float] = {0.1, 0.2, 0.3, 0.4, 0.5}
API_KEY: str = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    """
    Split `text` into chunks of length `chunk_size` with `overlap_ratio`.
    """
    step = int(chunk_size * (1 - overlap_ratio))
    if step <= 0 or len(text) <= chunk_size:
        return [text]

    chunks: List[str] = [
        text[i : i + chunk_size]
        for i in range(0, len(text) - chunk_size + 1, step)
    ]

    # Append the last chunk if there's leftover text
    if (len(text) - chunk_size) % step != 0:
        chunks.append(text[-chunk_size:])

    return chunks


def embed_text(text: str, model_name: str) -> List[float]:
    """
    Call Gemini embedding API via genai.embed_content.
    """
    response = genai.embed_content(
        model=model_name,
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]


def main(
    model_name: str = "models/embedding-001",
    document: Path = DOCUMENTS_PATH / "data.csv",
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
            # Meaningful name for the database and collection
            suffix = f"chunk_size_{chunk_size}_overlapping_{int(overlap * 100)}"
            db_name = f"gemini_{suffix}"
            db_folder = base_path / db_name
            db_folder.mkdir(parents=True, exist_ok=True)

            # Initialize persistent client and collection
            client = PersistentClient(path=str(db_folder))
            collection = client.get_or_create_collection(name=db_name)

            print(f"Indexing with chunk_size={chunk_size}, overlap={overlap}")
            for doc_id, text in enumerate(docs):
                print(f"Processing document {doc_id + 1}/{len(docs)}")
                chunks = chunk_text(text, chunk_size, overlap)
                for pos, chunk in enumerate(chunks):
                    try:
                        embedding = embed_text(chunk, model_name)
                        collection.add(
                            documents=[chunk],
                            embeddings=[embedding],
                            ids=[str(uuid.uuid4())],
                            metadatas=[{
                                "doc_id": doc_id,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "position": pos
                            }]
                        )
                    except Exception as e:
                        print(f"[Error] doc={doc_id}, cs={chunk_size}, ov={overlap}: {e}")

    print("✅ All embeddings have been created and saved.")


if __name__ == "__main__":
    fire.Fire(main)
