import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from documents import PATH as DOCUMENTS_PATH
from chroma.__main__ import generate_chroma_db
from documents import from_pandas_to_list


def generate_multi_step_rag(
    csv_path: str,
    chunk_size: int,
    overlap_ratio: float,
    embedding_model: GoogleGenAIEmbedding,
    llm: GoogleGenAI,
    k: int
):
    # Load CSV data
    df = pd.read_csv(csv_path)
    documents = from_pandas_to_list(df)

    # Generate Chroma DB
    stored_data = generate_chroma_db(
        documents,
        chunk_size=chunk_size,
        overlap=overlap_ratio,
        embedding_lm=embedding_model,
        force_recreate=False,
        db_name_base="multi_step"
    )

    # Load vector store
    vector_store = ChromaVectorStore(chroma_collection=stored_data)

    # Create vector store index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding_model,
    )

    # Create retriever
    retriever = index.as_retriever(similarity_top_k=k)

    # Create base query engine
    base_query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )

    # Create multi-step query engine
    multi_step_engine = MultiStepQueryEngine(
        query_engine=base_query_engine,
        llm=llm
    )

    return multi_step_engine

# # Example usage
# if __name__ == "__main__":
#     csv_path = str(DOCUMENTS_PATH / "data-generated.csv")
#     chunk_size = 512
#     overlap_ratio = 0.5
#     embedding_model = GoogleGenAIEmbedding()
#     llm = GoogleGenAI(model="models/gemini-2.0-flash-lite")
#     k = 3

#     rag = generate_multi_step_rag(
#         csv_path=csv_path,
#         chunk_size=chunk_size,
#         overlap_ratio=overlap_ratio,
#         embedding_model=embedding_model,
#         llm=llm,
#         k=k
#     )
#     # Example query
#     query = "What is the capital of France?"
#     response = rag.query(query)
#     print(response)
#     # Example query with multi-step reasoning
#     multi_step_query = "Explain the significance of the capital of France in European history."
#     multi_step_response = rag.query(multi_step_query)
#     print(multi_step_response)
