import os

from llama_index.llms.google_genai import GoogleGenAI

from documents import PATH as DOCUMENTS_PATH

# load test_generated.csv

import pandas as pd

pd.read_csv(DOCUMENTS_PATH / "test_generated.csv")


llm = GoogleGenAI(model_name="models/gemini-2.5-flash-preview-04-17", api_key=os.getenv("GOOGLE_API_KEY"))
# rename column with: Sentence, Response
df = pd.read_csv(DOCUMENTS_PATH / "test_generated.csv")
df = df.rename(columns={"user_input": "Sentence", "reference": "Response"})
# drop Unnamed: 0
df = df.drop(columns=["Unnamed: 0"])
print(df.columns)
# for each row (Sentence, Response) translate it in italian using llm
for i, row in df.iterrows():
    # get the row
    user_input = row["Sentence"]
    reference = row["Response"]
    # translate it
    user_input_it = llm.complete(f"Translate this to italian: {user_input}. JUST REPLY WITH THE TRANSLATED TEXT!!")
    reference_it = llm.complete(f"Translate this to italian: {reference}. JUST REPLY WITH THE TRANSLATED TEXT!!")
    # store it
    print(f"Translated {i} / {len(df)}")
    print(user_input_it)
    print(reference_it)

    df.at[i, "Sentence"] = user_input_it
    df.at[i, "Response"] = reference_it

# store as test_generated_it.csv
df.to_csv(DOCUMENTS_PATH / "test_generated_it.csv", index=False)