import asyncio
from pathlib import Path
import google.generativeai as genai
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import List, Any

PATH = Path(__file__).parent


class GeminiEmbedding(BaseEmbedding):
    def __init__(self, /, model_name: str = "models/embedding-001", **data: Any):
        super().__init__(**data)
        self._model_name = model_name

    @property
    def model(self) -> str:
        return self._model_name

    def _get_text_embedding(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self._model_name,
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]

    def _get_query_embedding(self, query: str) -> List[float]:
        response = genai.embed_content(
            model=self._model_name,
            content=query,
            task_type="retrieval_query"
        )
        return response["embedding"]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)