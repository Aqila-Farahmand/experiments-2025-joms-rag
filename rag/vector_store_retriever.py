import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore

from chroma import generate_chroma_db
from documents import from_pandas_to_list


def generate_vector_store_rag(
        csv_path: str, #  = str(DOCUMENTS_PATH / "data-generated.csv")
        chunk_size: int,
        overlap_ratio: float,
        embedding_model: BaseEmbedding,
        llm: LLM,
        k: int
) -> BaseQueryEngine:
    """
    Generate a simple RAG (Retrieval-Augmented Generation) model using ChromaDB.
    """
    # Load CSV data
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)

    # load the collection
    stored_data = generate_chroma_db(
        documents,
        chunk_size=chunk_size,
        overlap=overlap_ratio,
        embedding_lm=embedding_model,
        force_recreate=False,
        db_name_base=f"simple_{embedding_model.model_name}"
    )
    # load a VectorStore:
    vector_store = ChromaVectorStore(chroma_collection=stored_data)

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
