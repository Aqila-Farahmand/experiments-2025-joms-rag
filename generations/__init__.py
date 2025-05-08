from pathlib import Path

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM

PATH = Path(__file__).parent


class MetaInfo:
    def __init__(self, retriever: str, llm: str, embedder: str):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder

class RagUnderTest:
    def __init__(self, rag: BaseQueryEngine, metainfo: MetaInfo):
        self.rag = rag
        self.metainfo = metainfo

    def tag(self):
        return f"{self.metainfo.retriever}__{self.metainfo.llm}__{self.metainfo.embedder}"

