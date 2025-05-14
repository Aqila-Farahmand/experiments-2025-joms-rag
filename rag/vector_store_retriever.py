# rag/vector_store_retriever.py
import chromadb
import pandas as pd
from llama_index.core import VectorStoreIndex, ChatPromptTemplate
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import logging
from chroma import PATH as CHROMA_PATH, generate_chroma_db
from documents import from_pandas_to_list
from rag import refine_template_str, text_qa_template_str, refine_template_system, text_qa_message_system, \
    update_prompts


def generate_vector_store_rag(
        csv_path: str,
        chunk_size: int,
        overlap_ratio: float,
        embedding_model: BaseEmbedding,
        llm: LLM,
        k: int = 3,
        alpha: float = 0.5,
        *,
        persist: bool,
        collection_name: str = None,
        prompt_template: RichPromptTemplate = None
) -> tuple[RetrieverQueryEngine, VectorStoreIndex]:
    """
    Generate RAG (Retrieval-Augmented Generation) pipeline using ChromaDB.

    Args:
        csv_path: Path to the CSV file with data.
        chunk_size: Not used here directly, assumed pre-processed if needed.
        overlap_ratio: Not used here directly, assumed pre-processed if needed.
        embedding_model: Embedding model to use.
        llm: LLM for answering.
        k: Number of retrieved documents.
        alpha: Placeholder for potential future use (e.g., hybrid retrievers).
        persist: Whether to use persistent Chroma DB.
        collection_name: Optional name for Chroma collection. If None, a unique one is generated.
    """
    # Load CSV and process dataframe
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)
    # Convert raw strings into Document objects
    docs = [
        Document(text=doc)
        for doc in documents
        if isinstance(doc, str) and doc.strip()
    ]

    # Create Chroma client
    if persist:
        db = chromadb.PersistentClient(path=str(CHROMA_PATH))
    else:
        db = chromadb.Client()

    # Set or generate collection name
    if not collection_name:
        raise ValueError("Collection name must be provided for vector store retriever.")

    #real_collection = f"{collection_name}_chunk_size_{chunk_size}_overlapping_{int(overlap_ratio * 100)}"
    collection = generate_chroma_db(
        docs=docs,
        chunk_size=chunk_size,
        overlap=overlap_ratio,
        embedding_lm=embedding_model,
        db_name_base=collection_name,
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embedding_model,
        vector_store=vector_store,
    )

    logging.info(f"[INFO] Indexed {len(index.docstore.docs)} documents into collection '{collection_name}'.")

    retriever = index.as_retriever(similarity_top_k=k)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )

    update_prompts(query_engine)
    return query_engine, index
