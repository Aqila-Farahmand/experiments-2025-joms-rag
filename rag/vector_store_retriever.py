import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from documents import from_pandas_to_list
from chroma import PATH as CHROMA_PATH

# Name the chroma embeddings
_gemini_embedding = "gemini_chunk_size_256_overlapping_50"
_nomadic_embedding = "nomadic_chunk_size_256_overlapping_50"


def generate_vector_store_rag(
        csv_path: str,
        chunk_size: int,
        overlap_ratio: float,
        embedding_model: BaseEmbedding,
        llm: LLM,
        k: int,
        alpha: float,
) -> BaseQueryEngine:
    """
    Generate RAG (Retrieval-Augmented Generation) model using ChromaDB.
    """
    # Initialize existing Chromadb and choose the embedding name
    db = chromadb.PersistentClient(path=str(CHROMA_PATH))
    chroma_collection = db.get_or_create_collection(_gemini_embedding)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create vector store index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding_model,
    )

    # Create retriever with parameters
    retriever = index.as_retriever(
        similarity_top_k=k,  # Number of documents to retrieve
    )

    # Create query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )

    return query_engine
