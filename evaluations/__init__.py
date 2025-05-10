import os
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from llama_index.core.evaluation import RelevancyEvaluator, SemanticSimilarityEvaluator, CorrectnessEvaluator, \
    FaithfulnessEvaluator
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from requests import Response
import logging
import asyncio
from deepeval import evaluate
from tqdm.asyncio import tqdm_asyncio

PATH = Path(__file__).parent

judge_deep_eval = GeminiModel(
    model_name="gemini-2.0-flash",
    api_key=os.environ.get("GOOGLE_API_KEY"),
)
embedding = OllamaEmbedding("nomic-embed-text")

judge_llama_index = GoogleGenAI(model="models/gemini-2.5-flash-preview-04-17")

medical_faithfulness = GEval(
    name="Medical Correctness",
    evaluation_steps=[
        "Extract medical claims or diagnoses from the 0 'actual output'.",
        "Verify each medical claim against the 'expected output', such as clinical guidelines or medical literature.",
        "Identify any contradictions or unsupported medical claims that could lead to misdiagnosis.",
        "Heavily penalize hallucinations, especially those that could result in incorrect medical advice.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=judge_deep_eval
)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llama_index)
correctness_evaluator = CorrectnessEvaluator(llm=judge_llama_index)
semantic_similarity_evaluator = SemanticSimilarityEvaluator(embed_model=embedding)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llama_index)


async def eval_responses(responses: list[dict], data_under_test) -> dict:
    result = {
        'correctness': [],
        'semantic_similarity': [],
        'g_eval': []
    }

    # Create all test cases first for batch evaluation
    test_cases = []
    eval_tasks = []

    for i, question in enumerate(data_under_test["Response"]):
        response = responses[i]
        reference = data_under_test["Response"].iloc[i]

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=response['response'].response,
            expected_output=reference
        )
        test_cases.append(test_case)

        # Create async tasks for LlamaIndex evaluators
        eval_tasks.append(correctness_evaluator.aevaluate_response(
            query=question, response=response['response'], reference=reference
        ))
        eval_tasks.append(semantic_similarity_evaluator.aevaluate_response(
            query=question, response=response['response'], reference=reference
        ))

    # Run DeepEval in parallel with LlamaIndex evaluations
    g_eval_results = evaluate(
        test_cases=test_cases,
        metrics=[medical_faithfulness],
        show_indicator=False,
        display=None,
        print_results=False
    )

    # Execute all LlamaIndex evaluation tasks
    all_scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

    # Process results
    for i in range(0, len(all_scores), 2):
        result['correctness'].append(float(all_scores[i].score))
        result['semantic_similarity'].append(float(all_scores[i + 1].score))

    # Process DeepEval results
    for test_result in g_eval_results.test_results:
        result['g_eval'].append(test_result.metrics_data[0].score)

    return result


async def eval_rag(responses: list[dict], data_under_test):
    result = await eval_responses(responses=responses, data_under_test=data_under_test)
    result['faithfulness'] = []
    result['relevancy'] = []

    # Create async tasks for faithfulness and relevancy evaluations
    eval_tasks = []

    for i, question in enumerate(data_under_test["Response"]):
        response = responses[i]

        # Create async tasks
        eval_tasks.append(
            faithfulness_evaluator.aevaluate_response(response=response['response'])
        )
        eval_tasks.append(
            relevancy_evaluator.aevaluate_response(query=question, response=response['response'])
        )

    # Execute all tasks with progress bar
    all_scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating RAG metrics")

    # Process results - faithfulness at even indices, relevancy at odd indices
    for i in range(0, len(all_scores), 2):
        result['faithfulness'].append(float(all_scores[i].score))
        result['relevancy'].append(float(all_scores[i + 1].score))

    return result
