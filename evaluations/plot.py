import seaborn as sns
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity

from evaluations.cache import PATH as CACHE_PATH
from evaluations.plots import PATH as PLOTS_PATH


PRETTY_NAMES = {
    "gemma3-1b": "Gemma 3 (1B)",
    "granite3.1-moe:1b": "Granite 3.1 MoE (1B)",
    "falcon3-1b": "Falcon 3 (1B)",
    "qwen3-0.6b": "Qwen 3 (0.6B)",
    "deepseek-r1-1.5b": "DeepSeek R1 (1.5B)",
    "llama3.2-1b": "Llama 3.2 (1B)",
    "smollm2-1.7b": "SmolLM2 (1.7B)",
    "qwen2.5-0.5b": "Qwen 2.5 (0.5B)",
    "role_playing": "Role Playing Prompt",
    "hybrid": "Hybrid Prompt",
    "vector_store": "Vector Store Prompt",
    "vector_rerank": "Vector Rerank Prompt",
    "full": "Full Prompt"
}


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

def merge_dataframes(folder: str, embedder:str = "nomic") -> pd.DataFrame:
    """Load all non-summary CSVs, extract kind/method/model/embedding, and concat."""
    rows = []
    for path in Path(folder).glob("*.csv"):
        if "summary" in path.name:
            continue
        parts = path.stem.split("__")
        if parts[0] == "prompt":
            kind, method, model, embedding = "prompt", parts[1], parts[2], "None"
        else:
            kind, method, model, embedding = "rag", parts[0], parts[1], parts[2]
        print(f"Loading {path} with kind={kind}, method={method}, model={model}, embedding={embedding}")
        df = pd.read_csv(path)
        df = df.assign(
            kind=kind,
            method=method,
            model=model,
            embedding=embedding
        )
        # Convert method to categorical with specified order
        method_order = [PRETTY_NAMES[x] for x in ["role_playing", "full", "vector_store", "vector_rerank", "hybrid"]]
        df["method"] = df["method"].map(PRETTY_NAMES).fillna(df["method"])
        df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
        # sort the method in this way: "role_playing, full, vector_store, vector_rerank, hybrid"
        # filter if the kind is rag and the embedding is nomic
        if kind == "rag" and embedding != embedder:
            continue
        rows.append(df)

    return pd.concat(rows, ignore_index=True)

# Enhanced visualization functions with improved styling and separate scales
def plot_distributions(df: pd.DataFrame, only_g_eval: bool = False) -> None:
    """
    Plot the distribution of scores across different methods using boxplots.
    Each subplot uses its own appropriate scale based on the metric.
    """
    # Set aesthetic parameters
    plt.rcParams.update({'font.size': 10, 'figure.figsize': (14, 10)})
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    
    # Create FacetGrid with separate y-scales
    # Convert 'score' to numeric, forcing errors to NaN
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    g = sns.FacetGrid(df, col="model", row="metric", height=3.5, aspect=1.2, 
                        margin_titles=True, sharey=False)

    print(df["score"])
    # where nan, fill with 0
    #
    # Map boxplot with custom palette
    g.map_dataframe(sns.barplot, x="method", y="score",
                    palette="Set2", width=0.7)
    
    # Enhance appearance and readability
    g.set_axis_labels("Method", "Score")
    g.set_titles(col_template="{col_name} Model", row_template="{row_name}")
    
    # Rotate x-axis labels and add max value for each row
    for i, row_axes in enumerate(g.axes):
        metric_name = df['metric'].unique()[i]
        # Calculate the max value for the entire row (across all models with this metric)
        row_data = df[df['metric'] == metric_name]
        row_max = row_data['score'].max() if not row_data.empty else 0
        
        for ax in row_axes:
            ax.text(0.5, 0.98, f"Row Max: {row_max:.2f}", transform=ax.transAxes, 
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            ax.set_ylim(0, row_max * 1.1)  # Set y-limits to 10% above max value for better visibility
            # Rotate x-axis labels by 45 degrees
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Save high-quality figure
    plt.savefig(PLOTS_PATH / "distribution_scores.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_g_eval_distributions(df: pd.DataFrame, models_per_row: int = 4, embedder: str = "nomic") -> None:
    """
    Plot only the distribution of g_eval scores across different models and methods.
    Organize the subplots in a grid with `models_per_row` columns.
    """
    g_eval_df = df[df["metric"] == "g_eval"].copy()
    g_eval_df = g_eval_df[g_eval_df["model"].isin(PRETTY_NAMES.keys())].copy()
    g_eval_df["score"] = pd.to_numeric(g_eval_df["score"], errors="coerce")
    g_eval_df["model"] = g_eval_df["model"].map(PRETTY_NAMES).fillna(g_eval_df["model"])
    sorted_model_names = sorted(g_eval_df["model"].unique())
    g_eval_df["model"] = pd.Categorical(g_eval_df["model"], categories=sorted_model_names, ordered=True)

    plt.rcParams.update({'font.size': 12, 'figure.figsize': (models_per_row * 6, 20)})
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})

    g = sns.FacetGrid(
        g_eval_df,
        col="model",
        col_wrap=models_per_row,
        height=3.5,
        aspect=1.2,
        sharey=False
    )

    labels = ["role_playing", "full", "vector_store", "vector_rerank", "hybrid"]
    labels = [PRETTY_NAMES[label] for label in labels]

    def barplot_fixed_order(data, **kwargs):
        ax = plt.gca()
        sns.barplot(
            data=data,
            x="method",
            y="score",
            order=labels,
            palette="Set2",
            ax=ax,
            width=0.7
        )
    g.map_dataframe(barplot_fixed_order, x="method", y="score", palette="Set2", width=0.7)
    g.set_axis_labels("", "G-Eval Score")
    g.set_titles(col_template="{col_name}")

    for ax, model in zip(g.axes.flat, g.col_names):
        row_data = g_eval_df[g_eval_df["model"] == model]
        row_max = row_data["score"].max() if not row_data.empty else 0
        # ax.text(0.5, 0.98, f"Max: {row_max:.2f}", transform=ax.transAxes,
        #         ha='center', va='top', fontsize=9, fontweight='bold',
        #         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        ax.set_ylim(0, row_max * 1.1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Set the title for the entire figure
    g.fig.suptitle(f"G-Eval distributions by model using {embedder.capitalize()} embeddings", fontsize=24, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"g_eval_distributions_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_g_eval_correlations(df: pd.DataFrame) -> None:
    # collect g_eval scores for each kind-method-model combo
    evaluations = {}
    # Sort by kind (alphabetical), then by method (categorical order), then by model
    for model in sorted(df['model'].unique()):
        for method in df['method'].cat.categories:
            for kind in df['kind'].unique():
                data = df[
                    (df['kind'] == kind) &
                    (df['method'] == method) &
                    (df['model'] == model) &
                    (df['metric'] == "g_eval")
                ]
                if data.empty:
                    continue
                key = f"{kind[0]}-{method[0]}-{model}"
                evaluations[key] = data["score"].values

    # Build DataFrame and compute cosine similarity matrix
    eval_df = pd.DataFrame(evaluations)
    cos_sim_matrix = cosine_similarity(eval_df.T)
    cos_sim_matrix = np.nan_to_num(cos_sim_matrix)
    print("Cosine similarity matrix:")

    # Plot heatmap of cosine similarities without numbers
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cos_sim_matrix,
        annot=False,
        cmap="coolwarm",
        xticklabels=list(evaluations.keys()),
        yticklabels=list(evaluations.keys())
    )
    plt.title("Cosine Similarity Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "correlation_matrix_cosine.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_improvement_over_prompt_full(df: pd.DataFrame) -> None:
    """
    Plot g_eval improvement over the prompt-full baseline for each model
    in a grid of bar plots, excluding the 'full' method.
    """
    # select only g_eval rows
    gev = df[df["metric"] == "g_eval"].copy()
    models = sorted(gev["model"].unique())
    method_order = ["role_playing", "vector_store"]  # no 'full'
    
    # compute improvements
    records = []
    for model in models:
        sub = gev[gev["model"] == model]
        baseline = sub.loc[
            (sub["kind"] == "prompt") & (sub["method"] == "full"), "score"
        ].mean()
        if np.isnan(baseline):
            continue
        for _, row in sub.iterrows():
            if row["method"] == "full":
                continue
            records.append({
                "model": model,
                "kind": row["kind"],
                "method": row["method"],
                "improvement": row["score"] - baseline
            })
    imp_df = pd.DataFrame(records)

    # plot grid of improvements with red for positive and green for negative
    g = sns.catplot(
        data=imp_df,
        x="method",
        y="improvement",
        col="model",
        kind="bar",
        col_wrap=3,
        order=method_order,
        palette="Set2",   # will override per‐bar below
        height=3,
        aspect=1,
        ci=None
    )

    # Rotate x-axis labels to show method names clearly
    for ax in g.axes.flatten():
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    g.set_axis_labels("Method", "g_eval Improvement")
    g.fig.suptitle("g_eval Improvement over Prompt-Full by Model", y=1.02)

    # Recolor each bar: green if positive, red if negative
    for ax in g.axes.flatten():
        for bar in ax.patches:
            bar.set_color("green" if bar.get_height() >= 0 else "red")

    # Custom legend for sign
    handles = [
        Patch(color="green", label="Positive"),
        Patch(color="red", label="Negative")
    ]
    g.fig.legend(handles=handles, title="Improvement Sign", loc="upper right")

    plt.tight_layout()
    out_path = PLOTS_PATH / "g_eval_improvement_grid.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for embedder in ("nomic", "mxbai"):
        # Load and prepare the data
        df = merge_dataframes(CACHE_PATH, embedder=embedder)
        # Transform to long format for plotting
        df = df.melt(id_vars=["kind", "method", "model", "embedding"],
                        var_name="metric", value_name="score")

        # Generate plots
        # plot_distributions(df)
        plot_g_eval_distributions(df, models_per_row=4, embedder=embedder)
        # plot_g_eval_correlations(df)
        # plot_improvement_over_prompt_full(df)