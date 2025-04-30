from evaluations import PATH as EVAL_PATH

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# load all the csv files in the folder (that does not have summary in the file name)


def load_csv_in_folder(folder):
    """
    Load all the csv files in a folder.
    """
    # get all the files in the folder
    files = os.listdir(folder)
    # filter for only the csv files
    csv_files = [f for f in files if f.endswith(".csv") and "summary" not in f]
    # load all the csv files
    data = {}
    for file in csv_files:
        data[file] = pd.read_csv(os.path.join(folder, file))
    return data


# now merge all the dataframe in one.
# extract from the name the rag method, the model and the embedding


def extract_info_from_name(name: str):
    """
    Extract the rag method, the model and the embedding from the name.
    """
    # split by -
    parts = name.split("__")
    # get the first part
    rag_method = parts[0]
    # get the second part
    model = parts[1]
    # get the third part
    embedding = parts[2]
    embedding = embedding.replace(".csv", "")
    embedding = embedding.replace(".pkl", "")
    return rag_method, model, embedding


# now merge all the dataframes in one


def merge_dataframes(dataframes: dict):
    """
    Merge all the dataframes in one.
    """
    # create an empty dataframe
    merged = pd.DataFrame()
    # iterate over the dataframes
    for name, df in dataframes.items():
        # extract the info from the name
        rag_method, model, embedding = extract_info_from_name(name)
        # add the info to the dataframe
        df["rag_method"] = rag_method
        df["model"] = model
        df["embedding"] = embedding
        # merge the dataframe
        merged = pd.concat([merged, df], ignore_index=True)
    return merged


def plot_data(data: pd.DataFrame, metric: str, consider_model: bool = True):
    """
    Plot the data.
    """
    # set the style
    # create a figure
    plt.figure(figsize=(10, 6))
    # create a boxplot# create a boxplot with groups by embedding and model
    if consider_model:
        sns.catplot(x="model", y=metric, col="rag_method", hue="embedding",
                    kind="box", data=data)
    else:
        sns.boxplot(x="embedding", y=metric, hue="rag_method", data=data)
    # save the plot
    plt.savefig(os.path.join(EVAL_PATH, f"{metric}.png"))


pandas = load_csv_in_folder(EVAL_PATH)
merged = merge_dataframes(pandas)
plot_data(merged, "correctness")
plot_data(merged, "semantic_similarity")
plot_data(merged, "g_eval")
plot_data(merged, "faithfulness", False)
plot_data(merged, "relevancy")

# now plot the data
