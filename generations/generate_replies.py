from llama_index.core import Response
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from pandas import DataFrame


def generate_replies_from_rag(chain: BaseQueryEngine, data_under_test: DataFrame) -> list[Response]:
    result = []
    for i, question in enumerate(data_under_test["Sentence"]):
        # Generate a response using the chain
        result.append(chain.query(question))
        # Print the response
    return result

def generate_from_llm_with_prompt(
    llm: LLM,
    data_under_test: DataFrame,
    prompt_template: RichPromptTemplate
) -> list[Response]:
    result = []
    for i, question in enumerate(data_under_test["Sentence"]):
        # Generate a response using the chain
        prompt_template.format(question=question)
        result.append(Response(llm.complete(question).text))
        # Print the response
    return result