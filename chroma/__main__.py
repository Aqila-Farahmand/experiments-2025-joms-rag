import fire
import os
import uuid
from pathlib import Path
from typing import Set, List
from google import genai
from google.genai import types
from chromadb import PersistentClient
from chroma import PATH as CHROMA_PATH
from documents import PATH as DOCUMENTS_PATH, read_csv, from_pandas_to_list
import time

# — DEFAULT CONFIGURATION —
DEFAULT_CHUNK_SIZES: Set[int] = {128, 256, 512, 1024}
DEFAULT_OVERLAP_RATIOS: Set[float] = {0.1, 0.2, 0.3, 0.4, 0.5}
API_KEY: str = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)


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


def embed_text(text: list[str], model_name: str) -> List[List[float]]:
    """
    Call Gemini embedding API via genai.embed_content.
    """
    response = client.models.embed_content(
        model=model_name,
        contents=text,
        config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING")
    )
    embeddings = [embedding.values for embedding in response.embeddings]
    return embeddings


def main(
    model_name: str = "models/text-embedding-004",
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
            # Meaningful name for the database and collection
            suffix = f"chunk_size_{chunk_size}_overlapping_{int(overlap * 100)}"
            db_name = f"gemini_{suffix}"
            db_folder = base_path / db_name
            db_folder.mkdir(parents=True, exist_ok=True)

            # Initialize persistent client and collection
            client = PersistentClient(path=str(db_folder))
            collection = client.get_or_create_collection(name=db_name)

            print(f"Indexing with chunk_size={chunk_size}, overlap={overlap}")

            # Pre-process all documents to get chunks
            all_doc_chunks = []
            for doc_id, text in enumerate(docs):
                print(f"Pre-processing document {doc_id + 1}/{len(docs)}")
                chunks = chunk_text(text, chunk_size, overlap)
                # Store tuple of (doc_id, chunk_position, chunk_text)
                for pos, chunk in enumerate(chunks):
                    all_doc_chunks.append((doc_id, pos, chunk))

            # Process all chunks in batches across all documents
            batch_size = 100
            total_chunks = len(all_doc_chunks)
            print(f"Total chunks across all documents: {total_chunks}")

            for i in range(0, total_chunks, batch_size):
                start_index = i
                end_index = min(i + batch_size, total_chunks)
                current_batch = all_doc_chunks[start_index:end_index]

                print(
                    f"Processing batch {i // batch_size + 1} of {(total_chunks - 1) // batch_size + 1}, chunks {start_index}-{end_index - 1}")

                # Extract chunk texts for embedding
                chunk_texts = [item[2] for item in current_batch]

                try:
                    # Get embeddings for the entire batch at once
                    embedding_batch = embed_text(chunk_texts, model_name)

                    # Prepare batch data for collection
                    documents = []
                    embeddings = []
                    ids = []
                    metadatas = []

                    for j, (doc_id, pos, chunk) in enumerate(current_batch):
                        documents.append(chunk)
                        embeddings.append(embedding_batch[j])
                        ids.append(str(uuid.uuid4()))
                        metadatas.append({
                            "doc_id": doc_id,
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "position": pos
                        })

                    # Add the entire batch to the collection at once
                    collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas
                    )
                except Exception as e:
                    print(f"[Error] batch {i // batch_size + 1}, cs={chunk_size}, ov={overlap}: {e}")

    print("✅ All embeddings have been created and saved.")


if __name__ == "__main__":
    fire.Fire(main)
