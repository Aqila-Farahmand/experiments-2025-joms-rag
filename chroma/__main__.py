import fire
import os
import uuid
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from chroma import PATH as CHROMA_PATH
from documents import PATH as DOCUMENTS_PATH, read_csv, from_pandas_to_list

DEFAULT_CHUNK_SIZES = {128, 256, 512, 1024}
DEFAULT_OVERLAP_RATIOS = {0.1, 0.2, 0.3, 0.4, 0.5}
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)


def chunk_text(text: str, chunk_size: int, overlap_percent: float) -> list[str]:
    step: int = int(chunk_size * (1 - overlap_percent))
    chunks: list[str] = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]
    if len(text) > 0 and (len(text) - chunk_size) % step != 0:
        chunks.append(text[-chunk_size:])
    return chunks


def embed_text(text: str, model) -> list[float]:
    response: dict = model.embed_content(
        contents=text,
        task_type="retrieval_document"
    )
    return response["embedding"]


def main(model_name:str = "embedding-001",
         document:str = DOCUMENTS_PATH / "data.csv",
         chunk_sizes:set[int] = DEFAULT_CHUNK_SIZES,
         overlap_ratios:set[float] = DEFAULT_OVERLAP_RATIOS):

    # Load the document
    docs_df = read_csv(document)
    docs_strings: list[str] = from_pandas_to_list(docs_df)
    model = genai.get_model(model_name)

    chroma_client: chromadb.Client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(CHROMA_PATH)
    ))
    collection = chroma_client.get_or_create_collection("gemini_embeddings")

    for chunk_size in chunk_sizes:
        for overlap in overlap_ratios:
            print(f"▶ Embeddings: chunk_size={chunk_size}, overlap={overlap}")
            for doc_id, doc in enumerate(docs_strings):
                chunks: list[str] = chunk_text(doc, chunk_size, overlap)
                for i, chunk in enumerate(chunks):
                    try:
                        embedding: list[float] = embed_text(chunk, model)
                        collection.add(
                            documents=[chunk],
                            embeddings=[embedding],
                            ids=[str(uuid.uuid4())],
                            metadatas=[{
                                "doc_id": doc_id,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "position": i
                            }]
                        )
                    except Exception as e:
                        print(f"[Error] doc={doc_id}, chunk_size={chunk_size}, overlap={overlap}: {e}")

    chroma_client.persist()
    print("✔ Embedding complete.")



if __name__ == "__main__":
    fire.Fire(main)