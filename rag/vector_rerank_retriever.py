# rag/vector_rerank_retriever.py
import uuid
import pandas as pd
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
from documents import from_pandas_to_list
from chroma import PATH as CHROMA_PATH


def generate_vector_rerank_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: BaseEmbedding,
    llm: LLM,
    k: int,
    alpha: float,
    *,
    persist: bool = False,
    collection_name: str = None
) -> tuple[RetrieverQueryEngine, VectorStoreIndex]:
    """
    Generate a vector store retriever with LLM reranking.
    """
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
        collection_name = f"rerank_{embedding_model.model_name}_{uuid.uuid4().hex[:6]}"

    collection = db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embedding_model,
        vector_store=vector_store
    )

    print(f"[INFO] Indexed {len(index.docstore.docs)} docs into collection '{collection_name}' with reranking.")

    retriever = index.as_retriever(similarity_top_k=k)
    reranker = LLMRerank(llm=llm, top_n=k)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[reranker]
    )

    return query_engine, index
