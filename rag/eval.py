import os
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import GEval
# deepeval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from llama_index.core import Response
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.evaluation import FaithfulnessEvaluator
# llama index eval
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
# import paths and modules from root
from documents import PATH as DATA_PATH
from results.cache import PATH as CACHE_PATH
from rag.simple_rag import generate_simple_rag
from rag.hybrid_retriever import generate_hybrid_rag

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

test = pd.read_csv(DATA_PATH / "test.csv")

rag = generate_hybrid_rag(
    csv_path=str(DATA_PATH / "data-generated.csv"),
    chunk_size=256,
    overlap_ratio=0.5,
    embedding_model=embedding,
    llm=judge_llama_index,
    k=3,
    alpha=0.5
)
# remove for the full dataset
dataset_under_test = test[:5]
# consider to add caching
replies_rag = [rag.query(question) for question in dataset_under_test["Sentence"]]

# Evaluate all metrics
n = len(dataset_under_test)

result = {
    'correctness': [],
    'semantic_similarity': [],
    'g_eval': []
}


def eval_responses(responses: list[Response], data_under_test, _result: dict) -> dict:
    for i, question in enumerate(data_under_test["Response"]):
        response = responses[i]
        reference = data_under_test["Response"].iloc[i]
        print(".", end="", flush=True)

        correctness_score = float(
            correctness_evaluator.evaluate_response(query=question, response=response, reference=reference).score
        )
        semantic_similarity_score = float(
            semantic_similarity_evaluator.evaluate_response(query=question, response=response,
                                                            reference=reference).score
        )
        # DeepEval score
        test_case = LLMTestCase(
            input=question,
            actual_output=response.response,
            expected_output=reference
        )
        g_eval = evaluate(
            test_cases=[test_case],
            metrics=[medical_faithfulness],
            show_indicator=False,
            display=None,
            print_results=False
        )

        _result['correctness'].append(correctness_score)
        _result['semantic_similarity'].append(semantic_similarity_score)
        _result['g_eval'].append(g_eval.test_results[0].metrics_data[0].score)
    return _result


def eval_rag(responses: list[Response], data_under_test, _result):
    results = eval_responses(responses=responses, data_under_test=data_under_test, _result=_result)
    results['faithfulness'] = []
    results['relevancy'] = []
    for i, question in enumerate(data_under_test["Response"]):
        response = responses[i]
        reference = data_under_test["Response"].iloc[i]
        print(".", end="", flush=True)

        # Evaluate metrics (LLaMa Index)
        faithfulness_score = float(
            faithfulness_evaluator.evaluate_response(response=response).score
        )

        relevancy_score = float(
            relevancy_evaluator.evaluate_response(query=question, response=response).score
        )
        results['faithfulness'].append(faithfulness_score)
        results['relevancy'].append(relevancy_score)
    return results


result = eval_rag(responses=replies_rag, data_under_test=dataset_under_test, _result=result)
total_faithfulness = sum(result['faithfulness'])
total_correctness = sum(result['correctness'])
total_semantic_similarity = sum(result['semantic_similarity'])
total_relevancy = sum(result['relevancy'])
total_g_eval = sum(result['g_eval'])


print("")
print(f"Average faithfulness: {total_faithfulness / n:.2f}")
print(f"Average correctness: {total_correctness / n:.2f}")
print(f"Average semantic similarity: {total_semantic_similarity / n:.2f}")
print(f"Average relevancy: {total_relevancy / n:.2f}")
print(f"Average G-Eval: {total_g_eval / n:.2f}")

# Create DataFrame with results
results = pd.DataFrame({
    'Metric': ['Faithfulness', 'Correctness', 'Semantic Similarity', 'Relevancy', "G-Eval"],
    'Score': [
        total_faithfulness / n,
        total_correctness / n,
        total_semantic_similarity / n,
        total_relevancy / n,
        total_g_eval / n
    ]
})

print(results)
# Save results to CSV
results.to_csv(CACHE_PATH / "hybrid_retriever_results.csv", index=False)
