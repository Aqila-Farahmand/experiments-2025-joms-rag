from pathlib import Path

from llama_index.core.base.base_query_engine import BaseQueryEngine

PATH = Path(__file__).parent

class RagUnderTest:
    def __init__(self, rag: BaseQueryEngine, tag: str):
        self.rag = rag
        self.tag = tag