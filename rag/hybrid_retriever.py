# rag/hybrid_retriever.py
import chromadb
import pandas as pd
import logging
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from chroma import PATH as CHROMA_PATH, generate_chroma_db
from documents import from_pandas_to_list
from rag import update_prompts


def generate_hybrid_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: BaseEmbedding,
    llm: LLM,
    k: int,
    alpha: float = 0.5,
    collection_name: str = None,
) -> RetrieverQueryEngine:
    """
    Generate a Hybrid RAG using both dense (vector) and sparse (BM25) retrieval.
    """

    # Load CSV and process into text nodes
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)

    docs = [
        Document(text=doc)
        for doc in documents
        if isinstance(doc, str) and doc.strip()
    ]

    collection = generate_chroma_db(
        docs=docs,
        chunk_size=chunk_size,
        overlap=overlap_ratio,
        embedding_lm=embedding_model,
        db_name_base=collection_name,
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Use a SimpleDocumentStore for BM25
    doc_store = SimpleDocumentStore()
    doc_store.add_documents(docs)

    # Build index from documents and embedding model
    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embedding_model,
        vector_store=vector_store
    )

    logging.info(f"Hybrid RAG indexed {len(index.docstore.docs)} docs into collection '{collection_name}'")

    # Create retrievers
    vector_retriever = index.as_retriever(similarity_top_k=k)
    bm25_retriever = BM25Retriever.from_defaults(docstore=doc_store, similarity_top_k=k)

    # Combine into hybrid retriever
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        retriever_weights=[alpha, 1 - alpha],
        num_queries=3,
        use_async=True,
        llm=llm,
    )

    # Set up query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm
    )
    update_prompts(query_engine)
    return query_engine
