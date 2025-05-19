import chromadb
import pandas as pd
import logging
from llama_index.llms.ollama import Ollama
from chromadb.config import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from chroma import PATH as CHROMA_PATH, generate_chroma_db
from documents import from_pandas_to_list
from rag import (
    refine_template_str,
    text_qa_template_str,
    text_qa_message_system,
    refine_template_system,
    update_prompts,
)


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
    Generate a vector store retriever with LLM reranking and customizable prompts.
    """
    # Load and convert documents
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)
    docs = [
        Document(text=doc)
        for doc in documents
        if isinstance(doc, str) and doc.strip()
    ]

    # Create or load Chroma collection
    if not CHROMA_PATH.exists():
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        collection = generate_chroma_db(
            docs=docs,
            chunk_size=chunk_size,
            overlap=overlap_ratio,
            embedding_lm=embedding_model,
            db_name_base=collection_name,
        )
    else:
        client = chromadb.Client(Settings(persist_directory=str(CHROMA_PATH)))
        collection = client.get_or_create_collection(name=collection_name)

    # Initialize index and retriever
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embedding_model,
        vector_store=vector_store
    )
    retriever = index.as_retriever(similarity_top_k=k)

    # Configure reranker and query engine
    #llm_for_reranking = Ollama(model="qwen2.5:32b", base_url="http://clusters.almaai.unibo.it:11434/", request_timeout=60000)
    reranker = LLMRerank(llm=llm, top_n=k)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        reranker=reranker,
        llm=llm
    )

    # Update prompts
    prompts = query_engine.get_prompts()

    prompts["response_synthesizer:refine_template"].default_template.template = refine_template_str
    prompts["response_synthesizer:refine_template"].conditionals[0][1].message_templates = [
        ChatMessage(role=MessageRole.SYSTEM, content=refine_template_system)
    ]

    prompts["response_synthesizer:text_qa_template"].default_template.template = text_qa_template_str
    prompts["response_synthesizer:text_qa_template"].conditionals[0][1].message_templates = [
        ChatMessage(role=MessageRole.SYSTEM, content=text_qa_message_system),
        ChatMessage(role=MessageRole.USER, content=text_qa_template_str),
    ]

    query_engine.update_prompts(prompts)
    update_prompts(query_engine)

    logging.info(
        f"[INFO] Indexed {len(index.docstore.docs)} docs into collection '{collection_name}' with reranking."
    )

    return query_engine, index
