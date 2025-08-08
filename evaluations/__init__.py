import os
import time
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from google.genai.types import GenerateContentConfig, ThinkingConfig
from llama_index.core import PromptTemplate
from llama_index.core.evaluation import RelevancyEvaluator, SemanticSimilarityEvaluator, CorrectnessEvaluator, \
    FaithfulnessEvaluator
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from requests import Response
import logging
import asyncio
from deepeval import evaluate
from torch.distributed import group
from tqdm.asyncio import tqdm_asyncio

PATH = Path(__file__).parent

judge_deep_eval = GeminiModel(
    model_name="gemini-2.0-flash",
    api_key=os.environ.get("GOOGLE_API_KEY"),
)
embedding = OllamaEmbedding("nomic-embed-text")

judge_llama_index = GoogleGenAI(model="gemini-2.5-flash-lite")
judge_llama_index_eval = GoogleGenAI(model="gemini-2.5-pro", generation_config=GenerateContentConfig(thinking_config=ThinkingConfig(thinking_budget=128)))

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

prompt = PromptTemplate("""
Sei un medico incaricato di valutare la qualità delle risposte in un dominio di ipertensione.
 Il tuo compito è confrontare la risposta fornita con quella corretta e assegnare un punteggio da 1 (molto diversa da quella corretta) a 5 (simile a quella corretta).

## IMPORTANTE: Rispondi ESCLUSIVAMENTE con un numero da 1 a 5. Non aggiungere spiegazioni, commenti o altro testo.

## Criteri di Valutazione (considera sempre quanto si discosta dalla risposta corretta):

**1 - Risposta fuori contesto**
La risposta è completamente estranea alla domanda posta, 
oppure è completamente illeggibile e incomprensibile rispetto alla lingua italiana.
(es. se chiedo "Come ti chiami?", la risposta è "18 anni")

**2 - Risposta in contesto, ma con informazioni errate e/o pericolose**
La risposta è pertinente alla domanda, ma contiene informazioni sbagliate che potrebbero causare danni
(es. se chiedo "Come si misura la pressione?", la risposta è "Per misurare la pressione devi inserire un ago nel braccio")

**3 - Risposta sostanzialmente corretta, ma con imprecisioni minori**
La risposta è perlopiù corretta e utile, ma presenta piccoli errori, 
dettagli trascurabili o informazioni non del tutto pertinenti, 
oppure ha dei piccoli problemi con la grammatica italiana
(es. se chiedo "Come si misura la pressione?", la risposta descrive il modo corretto ma indica un valore di ipertensione accettabile palesemente sbagliato come "190/20 mmHg")

**4 - Risposta corretta, ma con informazioni superflue o con un linguaggio poco comprensibile**
La risposta è accurata e completa, ma include dettagli ridondanti 
o non strettamente necessari che un esperto avrebbe omesso per brevità.
(es. se chiedo "Come si misura la pressione?", la risposta descrive il metodo corretto ma aggiunge lunghi paragrafi su consigli alimentari e sulla gestione generale dell'ipertensione)

**5 - Risposta corretta, concisa ed efficace**
La risposta è precisa, va dritta al punto e fornisce tutte le informazioni necessarie in modo chiaro e sintetico, 
senza aggiunte inutili. È la risposta che darebbe un esperto

## Processo di Valutazione:

**Step di valutazione:**
- leggi la domanda (Domanda)
- leggi la risposta dell'altro medico (Risposta del Medico da Valutare)
- dai uno score seguendo rigorosamente i criteri sopra elencati, 
---

**Domanda:**
{question}

**Risposta Corretta:**
{ground_truth}

**Risposta del Medico da Valutare:**
{reply}

NOTA!!! cerca di evitare QUANTO più possibile di dare degli score intermedi!!! (es. 3, 2 ecc.), ma piuttosto < 2 o > 4, in modo da avere una valutazione più chiara e netta.
**VALUTAZIONE (solo numero da 1 a 5):**
""")
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
    prompts = []
    for i, question in enumerate(data_under_test["Sentence"]):
        response = responses[i]
        reference = data_under_test["Response"].iloc[i]

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=response['response'].response,
            expected_output=reference
        )
        prompts.append(prompt.format(question=question, ground_truth=reference, reply=response['response'].response[:2500]))
        print(len(response['response'].response[:2500]))
        print(":::::::::::::::: Question :::::::::::::::::::")
        print(f"Question: {question}")
        print(":::::::::::::::: Response :::::::::::::::::::")
        print(response['response'])
        print(":::::::::::::::: Reference :::::::::::::::::::")
        print(reference)
        print(":::::::::::::::::::::::::::::::::::")
        ## Create async tasks for LlamaIndex evaluators
        #eval_tasks.append(correctness_evaluator.aevaluate_response(
        #    query=question, response=response['response'], reference=reference
        #))
        #eval_tasks.append(semantic_similarity_evaluator.aevaluate_response(
        #    query=question, response=response['response'], reference=reference
        #))
    # run async and wait
    results = [judge_llama_index_eval.acomplete(prompt) for prompt in prompts]
    # Gather all results
    g_eval_results = await tqdm_asyncio.gather(*results, desc="Evaluating with DeepEval")
    time.sleep(30)
    # Execute all LlamaIndex evaluation tasks
    #all_scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

    # Process results
    #for i in range(0, len(all_scores), 2):

    #   result['correctness'].append(float(all_scores[i].passing) if all_scores[i].passing is not None else False)
    #   result['semantic_similarity'].append(float(all_scores[i + 1].score) if all_scores[i + 1].score is not None else 0.0)

    # Process DeepEval results
    for test_result in g_eval_results:
        print(test_result)
        result['g_eval'].append(float(str(test_result)))

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

async def embed_response(response: list[dict]) -> list[float]:
    """Embed the responses using the Ollama embedding model."""
    tasks = []
    for resp in response:
        tasks.append(embedding.aget_text_embedding(resp['response'].response))

    embeddings = await tqdm_asyncio.gather(*tasks, desc="Embedding responses")
    return embeddings

async def embed_ground_truth(data_under_test: dict) -> list[float]:
    """Embed the ground truth responses."""
    tasks = []
    for resp in data_under_test["Response"]:
        tasks.append(embedding.aget_text_embedding(resp))

    embeddings = await tqdm_asyncio.gather(*tasks, desc="Embedding ground truth")
    return embeddings
