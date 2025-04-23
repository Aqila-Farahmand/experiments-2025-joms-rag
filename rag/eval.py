import pandas as pd
import os

from deepeval import evaluate
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from simple_rag import generate_simple_rag
from documents import PATH as DATA_PATH
## llama index eval
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.evaluation import FaithfulnessEvaluator
## deepeval
from deepeval.models import GeminiModel
from deepeval.metrics import GEval

model = GeminiModel(
    model_name="gemini-2.0-flash",
    api_key=os.environ.get("GOOGLE_API_KEY"),
)

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
    model=model
)

test = pd.read_csv(DATA_PATH / "test.csv")

judge = GoogleGenAI(model="models/gemini-2.0-flash")
llm = GoogleGenAI(model="models/gemini-2.0-flash-lite")
embedding = GoogleGenAIEmbedding()

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
correctness_evaluator = CorrectnessEvaluator(llm=llm)
semantic_similarity_evaluator = SemanticSimilarityEvaluator(embed_model=embedding)
relevancy_evaluator = RelevancyEvaluator(llm=llm)
rag = generate_simple_rag(
    csv_path=str(DATA_PATH / "data-generated.csv"),
    chunk_size=512,
    overlap_ratio=0.5,
    embedding_model=embedding,
    llm=llm,
    k=3
)

dataset_under_test = test[:5] ## remove for the full dataset
replies_rag = [rag.query(question) for question in dataset_under_test["Sentence"]] # consider to add caching

# Evaluate all metrics
total_faithfulness = 0
total_correctness = 0
total_semantic_similarity = 0
total_relevancy = 0
total_g_eval = 0
n = len(dataset_under_test)

print(f"Evaluating {n} queries...")
for i, question in enumerate(dataset_under_test["Sentence"]):
    response = replies_rag[i]
    reference = dataset_under_test["Response"].iloc[i]
    print(".", end="", flush=True)

    # Evaluate metrics (LLaMa Index)
    faithfulness_score = float(faithfulness_evaluator.evaluate_response(response=response).score)
    correctness_score = float(correctness_evaluator.evaluate_response(
        query=question, response=response, reference=reference).score)
    semantic_similarity_score = float(semantic_similarity_evaluator.evaluate_response(
        query=question, response=response, reference=reference).score)
    relevancy_score = float(relevancy_evaluator.evaluate_response(
        query=question, response=response).score)
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
    # Accumulate scores
    total_faithfulness += faithfulness_score
    total_correctness += correctness_score
    total_semantic_similarity += semantic_similarity_score
    total_relevancy += relevancy_score
    total_g_eval += g_eval.test_results[0].metrics_data[0].score

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

