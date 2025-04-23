import os
import google.generativeai as genai
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.types import CompletionResponse, CompletionResponseGen


class GeminiLLM(LLM):
    def __init__(self, model: str = "models/gemini-1.5-pro", api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GOOGLE_API_KEY env variable.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self.model.generate_content(prompt)
        return CompletionResponse(text=response.text)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        raise NotImplementedError("Streaming not supported for Gemini.")
