import os
from analysis import PATH as ANALYSIS_PATH
from figures import PATH as FIGURES_PATH
import pandas as pd
import matplotlib.pyplot as plt
import fire

CHUNK_ANALYSIS_FILES = [
    ANALYSIS_PATH / file
    for file in os.listdir(ANALYSIS_PATH)
    if file.endswith(".csv")
]

def generate_heatmap(
    data: pd.DataFrame,
    metric: str = "avg_faithfulness",
    embedder: str = None
) -> None:
    df = data.copy()
    emb_label = "ALL"
    if embedder is not None:
        df = df[df["llm"] == embedder]
        emb_label = embedder.upper()

    pivot = df.pivot(
        index="overlap",
        columns="chunk_size",
        values=metric
    ).sort_index(ascending=True)
    # Change data types: chunk_size and overlap to str
    pivot.columns = pivot.columns.astype(str)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("viridis")
    c = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.values,
        cmap=cmap,
        shading="auto",
        edgecolors="white",
        linewidth=0.5
    )

    metric_label = metric.replace("avg_", "Average ").capitalize()
    # ax.set_title(f"{metric_label} — {emb_label}", fontsize=16)
    ax.set_xlabel("Chunk size", fontsize=16)
    ax.set_ylabel("Overlap", fontsize=16)

    cb = fig.colorbar(c, ax=ax, pad=0.02)
    cb.set_label(metric.replace("avg_", "Average ").capitalize(), fontsize=16)

    ax.set_yticks(pivot.index)
    ax.set_yticklabels([f"{int(o*100)}%" for o in pivot.index])

    ax.set_xticks(pivot.columns)
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])

    plt.tight_layout()
    fname = f"heatmap_{metric}"
    if embedder:
        fname += f"_{embedder}"
    plt.savefig(FIGURES_PATH / f"{fname}.png", dpi=200)
    plt.savefig(FIGURES_PATH / f"{fname}.eps")
    plt.close()


def main():
    llm_names = [f.name.split("_")[2][:-4] for f in CHUNK_ANALYSIS_FILES]
    dfs = []
    for name, file in zip(llm_names, CHUNK_ANALYSIS_FILES):
        df = pd.read_csv(file)
        df["llm"] = name
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    metrics = ["avg_faithfulness", "avg_relevancy", "avg_time"]
    for metric in metrics:
        for emb in data["llm"].unique():
            generate_heatmap(data, metric=metric, embedder=emb)

if __name__ == "__main__":
    fire.Fire(main)
