# rag/hybrid_retriever.py
import chromadb
import pandas as pd
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

from chroma import PATH as CHROMA_PATH
from documents import from_pandas_to_list
from rag import refine_template_str, text_qa_template_str


def generate_hybrid_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: BaseEmbedding,
    llm: LLM,
    k: int,
    alpha: float = 0.5,
    *,
    persist: bool = False,
    collection_name: str = None,
    prompt_template: RichPromptTemplate = None
) -> tuple[RetrieverQueryEngine, VectorStoreIndex]:
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

    # Initialize Chroma client
    db = chromadb.PersistentClient(path=str(CHROMA_PATH)) if persist else chromadb.Client()

    # Remove / from the embedding model name
    # Set or generate collection name
    if not collection_name:
        raise ValueError("Collection name must be provided for vector store retriever.")

    real_collection = f"{collection_name}_chunk_size_{chunk_size}_overlapping_{int(overlap_ratio * 100)}"
    collection = db.get_or_create_collection(name=real_collection)

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

    print(f"[INFO] Hybrid RAG indexed {len(index.docstore.docs)} docs into collection '{collection_name}'")

    # Create retrievers
    vector_retriever = index.as_retriever(similarity_top_k=k)
    bm25_retriever = BM25Retriever.from_defaults(docstore=doc_store, similarity_top_k=k)

    # Combine into hybrid retriever
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        retriever_weights=[alpha, 1 - alpha],
        num_queries=1,
        use_async=True,
        llm=llm,
    )

    # Set up query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm
    )
    refine_template = query_engine.get_prompts()["response_synthesizer:refine_template"]
    refine_template.default_template.template = refine_template_str
    text_qa_template = query_engine.get_prompts()["response_synthesizer:text_qa_template"]
    text_qa_template.default_template.template = text_qa_template_str
    return query_engine, index
