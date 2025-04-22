import os
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import uuid
import time
from google.api_core.exceptions import ResourceExhausted
from data import PATH as DATA_PATH
# Configure the Gemini API with your API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the Gemini embedding model
embedding_model = genai.GenerativeModel(model_name="gemini-embedding-exp-03-07")

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings())


# Function to chunk text
def chunk_text(text, chunk_size, overlap):
    tokens = text.split()
    step = int(chunk_size * (1 - overlap))
    chunks = []
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + chunk_size]
        if chunk:
            chunks.append(' '.join(chunk))
    return chunks


def get_embedding(text):
    response = genai.embed_content(
        model="gemini-embedding-exp-03-07",
        content=text
    )
    return response['embedding']


def get_embedding_with_retry(text, retries=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            return get_embedding(text)
        except ResourceExhausted as e:
            if attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e


file_path = DATA_PATH / "data.csv"
# Read the CSV file
df = pd.read_csv(file_path)

# Define chunk sizes and overlaps
chunk_sizes = [128, 256, 512, 1024]
overlaps = [0.1, 0.2, 0.3, 0.4, 0.5]

# Process each combination of chunk size and overlap
for chunk_size in chunk_sizes:
    for overlap in overlaps:
        collection_name = f"chunks_{chunk_size}_overlap_{int(overlap*100)}"
        # Create or get the collection
        collection = chroma_client.get_or_create_collection(name=collection_name)
        for idx, row in df.iterrows():
            for column in ['Sentence', 'Response']:
                text = row[column]
                chunks = chunk_text(text, chunk_size, overlap)
                for i, chunk in enumerate(chunks):
                    embedding = get_embedding_with_retry(chunk)
                    doc_id = f"{uuid.uuid4()}"
                    metadata = {
                        "source": column,
                        "original_index": idx,
                        "chunk_index": i,
                        "chunk_size": chunk_size,
                        "overlap": overlap
                    }
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
