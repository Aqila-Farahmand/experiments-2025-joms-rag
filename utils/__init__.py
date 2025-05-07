from pathlib import Path

from langchain_community.llms.ollama import Ollama
from llama_index.core.llms import LLM

PATH = Path(__file__).parent

def name_from_llm(llm: LLM) -> str:
    if isinstance(llm, Ollama):
        return llm.model
    else :
        return llm.model