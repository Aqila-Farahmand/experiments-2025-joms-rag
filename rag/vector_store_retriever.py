# rag/vector_store_retriever.py
import uuid
import pandas as pd
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import Document, ObjectType
from documents import from_pandas_to_list
from chroma import PATH as CHROMA_PATH
from llama_index.core.prompts import RichPromptTemplate


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
        collection_name = f"temp_{embedding_model.model_name}_{uuid.uuid4().hex[:6]}"

    collection = db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embedding_model,
        vector_store=vector_store,
    )

    print(f"[INFO] Indexed {len(index.docstore.docs)} documents into collection '{collection_name}'.")

    retriever = index.as_retriever(similarity_top_k=k)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )

    if prompt_template:
        query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})

    return query_engine, index
