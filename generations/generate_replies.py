from llama_index.core import Response
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate
from pandas import DataFrame


def generate_replies_from_rag(chain: BaseQueryEngine, data_under_test: DataFrame) -> list[dict]:
    result = []
    for i, question in enumerate(data_under_test["Sentence"]):
        response = chain.query(question)
        print(f"[{i}] Question: {question}\nResponse: {response.response}\n")
        result.append({
            "question": question,
            "response": response.response
        })
    return result


def generate_from_llm_with_prompt(
    llm: LLM,
    data_under_test: DataFrame,
    prompt_template: RichPromptTemplate
) -> list[Response]:
    result = []
    for i, question in enumerate(data_under_test["Sentence"]):
        # Manually provide both variables expected by the prompt template
        formatted_prompt = prompt_template.format(
            context_str="(Nessun contesto disponibile)",
            query_str=question
        )
        response_text = llm.complete(formatted_prompt).text
        result.append(Response(response_text))
        print(f"[{i}] Question: {question}\nFormatted Prompt:\n{formatted_prompt}\nResponse:\n{response_text}\n")
    return result


if __name__ == "__main__":
    # Example usage
    pass