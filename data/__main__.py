import os
import time
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.file import PandasCSVReader, CSVReader
from llama_index.core.evaluation.dataset_generation import DatasetGenerator
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from documents import PATH as DOCUMENTS_PATH
import nltk
import pandas as pd
from results.cache import PATH as CACHE_PATH
from google.genai import types


# Download NLTK stopwords
nltk.download('stopwords')

# Set up Google Gemini API Key and configure genai
api_key = os.environ['GOOGLE_API_KEY']
genai.configure(api_key=api_key)

# load data as document object
parser = CSVReader()
file_extractor = {".csv": parser}
documents = SimpleDirectoryReader(
    DOCUMENTS_PATH, file_extractor=file_extractor
).load_data()

# Define the Gemini LLM model
gemini_llm = GoogleGenAI(model_name="models/gemini-1.5-pro-latest")

# Select a subset of documents for evaluation
eval_documents = documents[:20]

# Use the Gemini model for the DatasetGenerator
data_generator = DatasetGenerator.from_documents(eval_documents, llm=gemini_llm)
eval_questions = data_generator.generate_questions_from_nodes(num=20)
print(f"Generated {len(eval_questions)} evaluation questions.")

# Set up evaluators for Faithfulness and Relevancy using the Gemini model
faithfulness_gemini = FaithfulnessEvaluator(llm=gemini_llm)
relevancy_gemini = RelevancyEvaluator(llm=gemini_llm)

print("Successfully set up evaluators using Gemini.")


def generate_gemini_embedding(self, input: documents):
    title = "Custom query"
    response = client.models.embed_content(
        model="models/embedding-001",  # TODO: change this
        contents=input,
        config=types.EmbedContentConfig(
            task_type="retrieval_document",
            title=title
        )
    )
    return response.embeddings[0].values


# Define the function to calculate average response time, faithfulness, and relevancy metrics
def evaluate_response_time_and_accuracy(chunk_size, eval_documents, gemini_llm):
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # Generate embeddings for the documents (using the model)
    embed_model = get_embeddings(eval_documents)

    if not embed_model:
        print("Failed to fetch embeddings. Exiting evaluation.")
        return None, None, None

    # Create the VectorStoreIndex for querying
    vector_index = VectorStoreIndex.from_documents(
        eval_documents, llm=gemini_llm, embed_model=embed_model, chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10
    )
    query_engine = vector_index.as_query_engine()
    num_questions = len(eval_questions)

    # Evaluate each question
    for question in eval_questions:
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time

        # Evaluate faithfulness and relevancy
        faithfulness_result = faithfulness_gemini.evaluate_response(response=response_vector).passing
        relevancy_result = relevancy_gemini.evaluate_response(query=question, response=response_vector).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    # Compute average metrics
    avg_response_time = total_response_time / num_questions
    avg_faithfulness = total_faithfulness / num_questions
    avg_relevancy = total_relevancy / num_questions

    return avg_response_time, avg_faithfulness, avg_relevancy


# Function to fetch embeddings using the Gemini model
def get_embeddings(documents):
    embeddings = []
    for document in documents:
        try:
            # response = genai.generate_embeddings(model="models/embedding-001", text=document.text)
            embeddings.append(response.embeddings)
        except Exception as e:
            print(f"Error generating embeddings for document: {e}")
            return []
    return embeddings


# Main function to run evaluations for different chunk sizes
def run_evaluations(eval_documents, gemini_llm):
    results = []

    # Iterate over different chunk sizes to evaluate the metrics
    for chunk_size in [128, 256, 512, 1024, 2048]:
        avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
            chunk_size, eval_documents, gemini_llm
        )

        if avg_time is None:
            continue

        print(f"Chunk size {chunk_size} - Average Response time: {avg_time:.2f}s, "
              f"Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")

        # Append the results for this chunk size
        results.append({
            "Chunk Size": chunk_size,
            "Average Response Time (s)": avg_time,
            "Average Faithfulness": avg_faithfulness,
            "Average Relevancy": avg_relevancy
        })

    return results


# Example usage in the if __name__ == "__main__": block
if __name__ == "__main__":
    # Run the evaluation and collect results
    results = run_evaluations(eval_documents, gemini_llm)

    # If results are available, convert them into a DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        print(df)
        df.to_csv(CACHE_PATH / 'chunk_size_results.csv', index=False)
