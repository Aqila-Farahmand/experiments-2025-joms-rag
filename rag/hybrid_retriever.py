import chromadb
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Node
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from chroma import generate_chroma_db
from documents import from_pandas_to_list


# to do: fix the bug in the bm25 retriever


def generate_hybrid_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: BaseEmbedding,
    llm: LLM,
    k: int,
    alpha: float = 0.5  # Blending weight: 0.0 = only BM25, 1.0 = only vector
) -> BaseQueryEngine:
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
        db_name_base=f"hybrid_rag_{embedding_model.model_name}"
    )

    # Initialize Chroma PersistentClient to create the Chroma vector store
    db = chromadb.PersistentClient(path="./chroma")
    chroma_collection = db.get_or_create_collection("dense_vectors")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create SimpleDocumentStore to use with BM25Retriever
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # Create the vector store index (for dense retrieval)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding_model,
    )

    # Create retrievers for BM25 and vector-based retrieval
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore, similarity_top_k=k
    )
    vector_retriever = index.as_retriever(similarity_top_k=k)

    # Combine BM25 and Vector retrievers into a hybrid retriever
    hybrid_retriever = QueryFusionRetriever(
        [
            vector_retriever,  # Dense vector-based retriever
            bm25_retriever,    # BM25 keyword-based retriever
        ],
        num_queries=1,
        use_async=True,
    )

    # Final query engine setup
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
    )

    return query_engine
