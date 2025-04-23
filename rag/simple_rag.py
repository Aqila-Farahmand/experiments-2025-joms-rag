import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from documents import PATH as DOCUMENTS_PATH
from chroma.__main__ import generate_chroma_db
from documents import from_pandas_to_list


def generate_simple_rag(
        csv_path: str, #  = str(DOCUMENTS_PATH / "data-generated.csv")
        chunk_size: int,
        overlap_ratio: float,
        embedding_model: BaseEmbedding,
        llm: BaseLLM,
        k: int
):
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
        db_name_base="simple"
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

embedding = GoogleGenAIEmbedding()
llm = GoogleGenAI(model_name="models/gemini-2.0-flash")
query = generate_simple_rag(
    csv_path=str(DOCUMENTS_PATH / "data-generated.csv"),
    chunk_size=512,
    overlap_ratio=0.5,
    embedding_model=embedding,
    llm=llm,
    k=3
)

print(query.query("How to handle hypertension?"))
