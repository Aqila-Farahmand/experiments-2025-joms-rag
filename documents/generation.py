import os
import time

from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.testset import TestsetGenerator
from documents import PATH as DATA_PATH
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

import asyncio

documents = SimpleDirectoryReader(DATA_PATH, exclude=['test.csv', 'data-generated.csv', "test_generated.csv", "*.py"]).load_data()

llm = GoogleGenAI(model_name="models/gemini-2.5-pro-preview-04-17", temperature=1.2, api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

personas = [
    Persona(
        name="Paziente",
        role_description="Sei un paziente che è vuole monitorare la propria ipertensione con questo chatbot. Fai sempre domande ben scritte",
    ),
]

transforms = [HeadlineSplitter(), NERExtractor(llm=LlamaIndexLLMWrapper(llm))]


distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=LlamaIndexLLMWrapper(llm)), 1.0),
]

generator = TestsetGenerator(
    llm=LlamaIndexLLMWrapper(llm),
    embedding_model=LlamaIndexEmbeddingsWrapper(embeddings),
    persona_list=personas
)

for query, _ in distribution:
    prompts = asyncio.run(query.adapt_prompts("italian", llm=LlamaIndexLLMWrapper(llm)))
    query.set_prompts(**prompts)

not_generated = True

while not_generated:
    try:
        print("Generating dataset...")
        dataset = generator.generate_with_llamaindex_docs(
            documents,
            testset_size=20,
            transforms_llm=llm,
            transforms_embedding_model=embeddings,
            query_distribution=distribution,
            #raise_exceptions=False
        )
        not_generated = False
        print(dataset.to_pandas())
        df = dataset.to_pandas()
        # store the dataset in a CSV file
        print(df)
        df[["user_input", "reference"]].to_csv(DATA_PATH / "test_generated.csv")

        # rag_dataset = dataset_generator.generate_questions_from_nodes()
        # questions = [e.query for e in rag_dataset.examples]
        # print(questions)
    except:
        print("Retrying...")
        time.sleep(10)

