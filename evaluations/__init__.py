import os
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from llama_index.core.evaluation import RelevancyEvaluator, SemanticSimilarityEvaluator, CorrectnessEvaluator, \
    FaithfulnessEvaluator
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from requests import Response
import logging
PATH = Path(__file__).parent

judge_deep_eval = GeminiModel(
    model_name="gemini-2.0-flash",
    api_key=os.environ.get("GOOGLE_API_KEY"),
)
embedding = GoogleGenAIEmbedding()

judge_llama_index = GoogleGenAI(model="models/gemini-2.0-flash-lite")

medical_faithfulness = GEval(
    name="Medical Correctness",
    evaluation_steps=[
        "Extract medical claims or diagnoses from the 0 'actual output'.",
        "Verify each medical claim against the 'expected output', such as clinical guidelines or medical literature.",
        "Identify any contradictions or unsupported medical claims that could lead to misdiagnosis.",
        "Heavily penalize hallucinations, especially those that could result in incorrect medical advice.",
        "Provide reasons for the faithfulness score, emphasizing the importance of clinical accuracy and patient safety."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=judge_deep_eval
)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llama_index)
correctness_evaluator = CorrectnessEvaluator(llm=judge_llama_index)
semantic_similarity_evaluator = SemanticSimilarityEvaluator(embed_model=embedding)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llama_index)


def eval_responses(responses: list[dict], data_under_test) -> dict:
    result = {
        'correctness': [],
        'semantic_similarity': [],
        'g_eval': []
    }
    for i, question in enumerate(data_under_test["Response"]):
        response = responses[i]
        reference = data_under_test["Response"].iloc[i]
        print(".", end="", flush=True)
        correctness_score = float(
            correctness_evaluator.evaluate_response(query=question, response=response['response'], reference=reference).score
        )
        semantic_similarity_score = float(
            semantic_similarity_evaluator.evaluate_response(query=question, response=response['response'],
                                                            reference=reference).score
        )
        # DeepEval score
        test_case = LLMTestCase(
            input=question,
            actual_output=response['response'].response,
            expected_output=reference
        )
        g_eval = evaluate(
            test_cases=[test_case],
            metrics=[medical_faithfulness],
            show_indicator=False,
            display=None,
            print_results=False
        )

        result['correctness'].append(correctness_score)
        result['semantic_similarity'].append(semantic_similarity_score)
        result['g_eval'].append(g_eval.test_results[0].metrics_data[0].score)
    return result


def eval_rag(responses: list[dict], data_under_test):
    result = eval_responses(responses=responses, data_under_test=data_under_test)
    result['faithfulness'] = []
    result['relevancy'] = []
    for i, question in enumerate(data_under_test["Response"]):
        response = responses[i]
        print(".", end="", flush=True)

        # Evaluate metrics (LLaMa Index)
        faithfulness_score = float(
            faithfulness_evaluator.evaluate_response(response=response['response']).score
        )

        relevancy_score = float(
            relevancy_evaluator.evaluate_response(query=question, response=response['response']).score
        )
        result['faithfulness'].append(faithfulness_score)
        result['relevancy'].append(relevancy_score)
    return result
