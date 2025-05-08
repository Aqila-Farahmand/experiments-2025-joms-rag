import pandas as pd
from generations import PATH as GENERATIONS_PATH
from documents import PATH as DOCUMENTS_PATH
import os
import pickle
from evaluations import eval_rag, eval_responses
from generations.cache import PATH as GENERATIONS_CACHE_PATH
from evaluations.cache import PATH as EVAL_CACHE_PATH

data_under_test = pd.read_csv(DOCUMENTS_PATH / "test.csv")[:10] # remove :5 for the full dataset


def load_pickle_in_folder(folder):
    """
    Load all the pickle files in a folder.
    """

    # get all the files in the folder
    files = os.listdir(folder)
    # filter for only the pickle files
    pickle_files = [f for f in files if f.endswith(".pkl")]
    # load all the pickle files
    data = {}
    for file in pickle_files:
        with open(os.path.join(folder, file), "rb") as f:
            data[file] = pickle.load(f)
    return data


to_eval = load_pickle_in_folder(GENERATIONS_CACHE_PATH)
for key, value in to_eval.items():
    print(f"Evaluating {key}")
    # remove the .pkl extension
    print(value["responses"][0])
    if("prompt" in key):
        result = eval_responses(value["responses"], data_under_test)
    else:
        result = eval_rag(value["responses"], data_under_test)
    # the result is as follows: {'metric_name': [score]}
    # convert to pands
    df = pd.DataFrame.from_dict(result)
    # store as csv
    key = key.replace(".pkl", "")
    df.to_csv(os.path.join(EVAL_CACHE_PATH, f"{key}.csv"), index=False)
    # In summary, the following metrics are included:
    """
    mean, std
    correctness
    semantic_similarity
    ---
    """
    summary = df.describe().T[["mean", "std"]]
    summary = summary.rename(columns={"mean": "average", "std": "std"})
    # add the name of the metric as a column
    summary["metric"] = summary.index
    # reorder the columns
    summary = summary[["metric", "average", "std"]]

    # store as csv
    summary.to_csv(os.path.join(EVAL_CACHE_PATH, f"{key}_summary.csv"), index=False)
