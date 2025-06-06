import uuid
from collections.abc import Collection
from pathlib import Path
from typing import List
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
import logging
PATH = Path(__file__).parent

# why not the SentenceSplitter => from llama_index.core.node_parser import SentenceSplitter
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


def generate_chroma_db(
        docs: List[str],
        chunk_size: int,
        overlap: float,
        embedding_lm,
        base_path: Path = PATH,
        db_name_base: str = "gemini",
        force_recreate: bool = False
) -> Collection:
    """
    Generate a Chroma database for the given documents with specified chunk size and overlap ratio.
    If the database already exists, it will be skipped unless force_recreate is True.

    Args:
        docs: List of document strings to process
        chunk_size: Size of each text chunk
        overlap: Overlap ratio between chunks (0.0 to 1.0)
        embedding_lm: The embedding model object used to generate embeddings
        base_path: Base directory for storing the database
        db_name_base: Base name for the database (default: "gemini")
        force_recreate: If True, recreate the database even if it exists (default: False)
    """
    # Ensure base directory exists
    base_path.mkdir(parents=True, exist_ok=True)

    # Meaningful name for the database and collection
    suffix = f"chunk_size_{chunk_size}_overlapping_{int(overlap * 100)}"
    db_name = f"{db_name_base}_{suffix}"
    db_folder = base_path / db_name
    logging.info(f"Creating database '{db_name}' at {db_folder}")
    # Check if database already exists
    if db_folder.exists() and not force_recreate:
        logging.info(f"Database '{db_name}' already exists. Skipping creation.")
        client = PersistentClient(path=str(db_folder))
        collection = client.get_collection(name=db_name)
        logging.info(f"Collection '{db_name}' contains {collection.count()} entries.")
        return collection

    # Create directory for new database
    db_folder.mkdir(parents=True, exist_ok=True)

    # Initialize persistent client and collection
    client = PersistentClient(path=str(db_folder))
    collection = client.get_or_create_collection(name=db_name)

    logging.info(f"Indexing with chunk_size={chunk_size}, overlap={overlap}")

    # Pre-process all documents to get chunks
    all_doc_chunks = []
    for doc_id, text in enumerate(docs):
        logging.info(f"Pre-processing document {doc_id + 1}/{len(docs)}")
        chunks = chunk_text(text, chunk_size, overlap)
        # Store tuple of (doc_id, chunk_position, chunk_text)
        for pos, chunk in enumerate(chunks):
            all_doc_chunks.append((doc_id, pos, chunk))

    # Process all chunks in batches across all documents
    batch_size = 100
    total_chunks = len(all_doc_chunks)
    logging.info(f"Total chunks across all documents: {total_chunks}")

    for i in range(0, total_chunks, batch_size):
        start_index = i
        end_index = min(i + batch_size, total_chunks)
        current_batch = all_doc_chunks[start_index:end_index]

        logging.info(
            f"Processing batch {i // batch_size + 1} of {(total_chunks - 1) // batch_size + 1}, chunks {start_index}-{end_index - 1}")

        # Extract chunk texts for embedding
        chunk_texts = [item[2] for item in current_batch]

        try:
            # Get embeddings for the entire batch at once
            embedding_batch = embedding_lm.get_text_embedding_batch(chunk_texts)

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
            logging.info(f"[Error] batch {i // batch_size + 1}, cs={chunk_size}, ov={overlap}: {e}")

    logging.info(f"✅ DB '{db_name}' created successfully with {total_chunks} chunks.")

    return collection
