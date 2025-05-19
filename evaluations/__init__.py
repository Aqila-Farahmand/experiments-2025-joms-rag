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

judge_llama_index = GoogleGenAI(model="gemini-2.5-flash-preview-04-17")

medical_faithfulness = GEval(
    name="Medical Correctness",
    evaluation_steps=[
        "Estrai le affermazioni mediche o le diagnosi dal 'actual output'.",
        "Verifica ogni affermazione medica contro il 'expected output', come le linee guida cliniche o la letteratura medica.",
        "Identifica eventuali contraddizioni o affermazioni mediche non supportate che potrebbero portare a una diagnosi errata.",
        "Penalizza pesantemente le allucinazioni, in particolare quelle che potrebbero generare consigli medici errati.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=judge_deep_eval
)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llama_index)
correctness_evaluator = CorrectnessEvaluator(llm=judge_llama_index, score_threshold=3.0)
semantic_similarity_evaluator = SemanticSimilarityEvaluator(embed_model=embedding)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llama_index)


async def eval_responses(responses: list[dict], data_under_test) -> dict:
    result = {
        #'correctness': [],
        #'semantic_similarity': [],
        'g_eval': []
    }

    # Create all test cases first for batch evaluation
    test_cases = []
    eval_tasks = []

    for i, question in enumerate(data_under_test["Sentence"]):
        response = responses[i]
        reference = data_under_test["Response"].iloc[i]

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=response['response'].response,
            expected_output=reference
        )
        test_cases.append(test_case)
        print(":::::::::::::::: Question :::::::::::::::::::")
        print(f"Question: {question}")
        print(":::::::::::::::: Response :::::::::::::::::::")
        print(response['response'])
        print(":::::::::::::::: Reference :::::::::::::::::::")
        print(reference)
        print(":::::::::::::::::::::::::::::::::::")
        # Create async tasks for LlamaIndex evaluators
        #eval_tasks.append(correctness_evaluator.aevaluate_response(
        #    query=question, response=response['response'], reference=reference
        #))
        #eval_tasks.append(semantic_similarity_evaluator.aevaluate_response(
        #    query=question, response=response['response'], reference=reference
        #))

    # Run DeepEval in parallel with LlamaIndex evaluations
    g_eval_results = evaluate(
        test_cases=test_cases,
        metrics=[medical_faithfulness],
        show_indicator=False,
        display=None,
        print_results=False
    )

    # Execute all LlamaIndex evaluation tasks
    #all_scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

    # Process results
    #for i in range(0, len(all_scores), 2):

    #   result['correctness'].append(float(all_scores[i].passing) if all_scores[i].passing is not None else False)
    #   result['semantic_similarity'].append(float(all_scores[i + 1].score) if all_scores[i + 1].score is not None else 0.0)

    # Process DeepEval results
    for test_result in g_eval_results.test_results:
        result['g_eval'].append(float(test_result.metrics_data[0].success))

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

    async def safe_task_runner(task_coroutine, timeout=30):
        try:
            # Add timeout using asyncio.wait_for
            return await asyncio.wait_for(task_coroutine, timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Task timed out after {timeout} seconds")
            return None
        except Exception as e:
            print(f"Task failed with error: {e}")
            return None

    # Wrap each task with the safe runner and timeout
    safe_eval_tasks = [safe_task_runner(task) for task in eval_tasks]

    # Gather all tasks, allowing some to fail or timeout
    all_scores = await tqdm_asyncio.gather(*safe_eval_tasks, desc="Evaluating RAG metrics")

    # Process results - faithfulness at even indices, relevancy at odd indices
    for i in range(0, len(all_scores), 2):
        # Use 0 for failed or timed out tasks (None values)
        faith_score = 0.0 if all_scores[i] is None else float(all_scores[i].score)
        rel_score = 0.0 if all_scores[i + 1] is None else float(all_scores[i + 1].score)

        result['faithfulness'].append(faith_score)
        result['relevancy'].append(rel_score)
    return result
