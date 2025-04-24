from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Node
from chroma.__main__ import generate_chroma_db
import pandas as pd
import chromadb
import os
from documents import from_pandas_to_list


def generate_hybrid_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: GoogleGenAIEmbedding,
    llm: GoogleGenAI,
    k: int,
    alpha: float = 0.5  # Blending weight: 0.0 = only BM25, 1.0 = only vector
):
    """
    Generate a Hybrid RAG (Retrieval-Augmented Generation) model using
    both BM25 (keyword) and vector similarity search with LlamaIndex.
    """

    # Load CSV data
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)

    # Convert raw documents (strings) to Node objects
    nodes = [Node(text=doc) for doc in documents]

    # Generate vector DB (Chroma)
    stored_data = generate_chroma_db(
        documents,
        chunk_size=chunk_size,
        overlap=overlap_ratio,
        embedding_lm=embedding_model,
        force_recreate=False,
        db_name_base="hybrid_rag"
    )

    # Initialize Chroma PersistentClient to create the Chroma vector store
    db = chromadb.PersistentClient(path="./chroma")  # Ensure this path exists
    chroma_collection = db.get_or_create_collection("dense_vectors")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create SimpleDocumentStore to use with BM25Retriever
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)  # Adding Node objects to docstore

    # Create the vector store index (for dense retrieval)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding_model,
    )

    # Create retrievers for BM25 and vector-based retrieval
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore, similarity_top_k=k  # Use the docstore here
    )
    vector_retriever = index.as_retriever(similarity_top_k=k)

    # Combine BM25 and Vector retrievers into a hybrid retriever
    hybrid_retriever = QueryFusionRetriever(
        [
            vector_retriever,  # Dense vector-based retriever
            bm25_retriever,    # BM25 keyword-based retriever
        ],
        num_queries=1,
        use_async=True,  # Enable async queries for faster performance if needed
    )

    # Final query engine setup
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
    )

    return query_engine
