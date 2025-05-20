# rag/bm25_retriever.py
import pandas as pd
import logging
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from documents import from_pandas_to_list
from rag import update_prompts
import Stemmer
from llama_index.core.node_parser import SentenceSplitter


def generate_bm25_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: None,
    llm: LLM,
    k: int,
    alpha: float = None,
    collection_name: str = None,
) -> RetrieverQueryEngine:
    """
    Generate a RAG pipeline using only BM25 sparse retrieval.
    """

    # Load CSV and convert to list of document texts
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)

    # initialize node parser
    splitter = SentenceSplitter(chunk_size=chunk_size)

    docs = [
        Document(text=doc)
        for doc in documents
        if isinstance(doc, str) and doc.strip()
    ]

    nodes = splitter.get_nodes_from_documents(docs)

    # Initialize and populate a simple document store
    # doc_store = SimpleDocumentStore()
    # doc_store.add_documents(nodes)

    # Create a BM25 retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=k,
        stemmer=Stemmer.Stemmer("italian"),
        language="italian",

    )

    # Create query engine using only BM25 retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=bm25_retriever,
        llm=llm
    )

    update_prompts(query_engine)
    logging.info(f"BM25 RAG indexed {len(docs)} documents.")
    return query_engine
