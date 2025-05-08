from llama_index.core import Response
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from pandas import DataFrame
import logging
MAX_OUTPUT_BOUND = 50
def generate_replies_from_rag(chain: BaseQueryEngine, data_under_test: DataFrame) -> list[dict]:
    result = []
    for i, question in enumerate(data_under_test["Sentence"]):
        response = chain.query(question)
        # split by </think>
        response_text = response.response.split("</think>")[-1]
        response.response = response_text
        logging.info(f"[{i}] Question: {question}\nResponse: {response_text[:MAX_OUTPUT_BOUND]}\n")
        result.append({
            "question": question,
            "response": response
        })
    return result

if __name__ == "__main__":
    # Example usage
    pass