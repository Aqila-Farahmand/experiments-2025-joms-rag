import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore

from chroma import generate_chroma_db
from documents import from_pandas_to_list


def generate_multi_step_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: BaseEmbedding,
    llm: LLM,
    k: int
) -> BaseQueryEngine:
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
        db_name_base=f"multi_step_{embedding_model.model_name}"
    )

    # Load vector store
    vector_store = ChromaVectorStore(chroma_collection=stored_data)

    # Create vector store index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding_model,
    )

    # Create retriever
    retriever = index.as_retriever(similarity_top_k=k)

    # Create base query engine
    base_query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )

    # Create multi-step query engine
    multi_step_engine = MultiStepQueryEngine(
        query_engine=base_query_engine,
        llm=llm
    )

    return multi_step_engine

