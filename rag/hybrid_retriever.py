import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import HybridRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from documents import PATH as DOCUMENTS_PATH
from chroma.__main__ import generate_chroma_db
from documents import from_pandas_to_list

def generate_hybrid_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: GoogleGenAIEmbedding,
    llm: GoogleGenAI,
    k: int
):
    # Load CSV data
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)

    # Generate Chroma DB
    stored_data = generate_chroma_db(
        documents,
        chunk_size=chunk_size,
        overlap=overlap_ratio,
        embedding_lm=embedding_model,
        force_recreate=False,
        db_name_base="hybrid"
    )

    # Load vector store
    vector_store = ChromaVectorStore(chroma_collection=stored_data)

    # Create vector store index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding_model,
    )

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        index=index,
        similarity_top_k=k,
        alpha=0.5  # Balance between keyword and vector similarity
    )

    # Create query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
    )

    return query_engine
