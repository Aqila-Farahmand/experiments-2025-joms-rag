import os
import time

import pandas as pd
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

documents = SimpleDirectoryReader(DATA_PATH, exclude=['test.csv', 'data-generated.csv', "test_generated.csv", "test_generated.it.csv", "train.csv", "data_raw.csv", "*.py"]).load_data()

# from data_raw.csv load as pandas df
df = pd.read_csv(DATA_PATH / "data_raw.csv")
# break it in several q&a of 100 elements, ad store them as (data_0, data_100, ...).

batch = 100
for i in range(0, len(df), batch):
    # get the batch
    batch_df = df.iloc[i:i + batch]
    # store
    batch_df.to_csv(DATA_PATH / f"data_{i}_{i + batch}.csv", index=False)

llm = GoogleGenAI(model_name="models/gemini-2.0-flash", temperature=0.9, api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

personas = [
    Persona(
        name="Paziente",
        role_description="Sei un paziente che è vuole monitorare la propria ipertensione con questo chatbot. Fai sempre domande ben scritte",
    ),
]

distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=LlamaIndexLLMWrapper(llm)), 1.0),
]

generator = TestsetGenerator(
    llm=LlamaIndexLLMWrapper(llm),
    embedding_model=LlamaIndexEmbeddingsWrapper(embeddings),
    #persona_list=personas
)

#for query, _ in distribution:
#    prompts = asyncio.run(query.adapt_prompts("italian", llm=LlamaIndexLLMWrapper(llm)))
#    query.set_prompts(**prompts)

not_generated = True

while not_generated:
    try:
        print("Generating dataset...")
        dataset = generator.generate_with_llamaindex_docs(
            documents,
            testset_size=20,
            transforms_llm=llm,
            transforms_embedding_model=embeddings,
            #query_distribution=distribution,
            #raise_exceptions=False
        )
        not_generated = False
        df = dataset.to_pandas()
        # store the dataset in a CSV file
        df[["user_input", "reference"]].to_csv(DATA_PATH / "test_generated.csv")

        # rag_dataset = dataset_generator.generate_questions_from_nodes()
        # questions = [e.query for e in rag_dataset.examples]
        # print(questions)
    except Exception as e:
        print(e)
        print("Retrying...")
        time.sleep(10)

