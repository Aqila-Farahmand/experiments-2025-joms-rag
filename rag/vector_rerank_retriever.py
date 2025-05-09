# rag/vector_rerank_retriever.py
import chromadb
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import logging
from chroma import PATH as CHROMA_PATH, generate_chroma_db
from documents import from_pandas_to_list
from rag import refine_template_str, text_qa_template_str, text_qa_message_system, refine_template_system, \
    update_prompts


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
    collection_name: str = None,
    prompt_template: RichPromptTemplate = None
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

    collection = generate_chroma_db(
        docs=docs,
        chunk_size=chunk_size,
        overlap=overlap_ratio,
        embedding_lm=embedding_model,
        db_name_base=collection_name,
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embedding_model,
        vector_store=vector_store
    )

    logging.info(f"[INFO] Indexed {len(index.docstore.docs)} docs into collection '{collection_name}' with reranking.")

    retriever = index.as_retriever(similarity_top_k=k)
    reranker = LLMRerank(llm=llm, top_n=k)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[reranker]
    )
    refine_template = query_engine.get_prompts()["response_synthesizer:refine_template"]
    refine_template.default_template.template = refine_template_str

    refine_template.conditionals[0][1].message_templates = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=refine_template_system
        )
    ]
    text_qa_template = query_engine.get_prompts()["response_synthesizer:text_qa_template"]
    text_qa_template.default_template.template = text_qa_template_str
    text_qa_template.conditionals[0][1].message_templates = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=text_qa_message_system
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=text_qa_template_str
        ),
    ]
    query_engine.update_prompts(
        {
            "response_synthesizer:refine_template": refine_template,
            "response_synthesizer:text_qa_template": text_qa_template,
        }
    )
    update_prompts(query_engine)
    return query_engine, index
