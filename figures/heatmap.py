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

def compute_global_vmin_vmax(data: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = data[metric].dropna()
    return values.min(), values.max()

def generate_heatmap(
    data: pd.DataFrame,
    metric: str = "avg_faithfulness",
    embedder: str = None,
    vmin: float = None,
    vmax: float = None,
) -> None:
    df = data.copy()
    if embedder is not None:
        df = df[df["llm"] == embedder]
    else:
        return
    pivot = df.pivot(
        index="overlap",
        columns="chunk_size",
        values=metric
    ).sort_index(ascending=True)
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
        linewidth=0.5,
        vmin=vmin,
        vmax=vmax
    )

    metric_label = metric.replace("avg_", "Average ").capitalize()
    ax.set_xlabel("Chunk size", fontsize=16)
    ax.set_ylabel("Overlap", fontsize=16)

    cb = fig.colorbar(c, ax=ax, pad=0.02)
    cb.set_label(metric_label, fontsize=16)

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


def print_best_joint_configs(data: pd.DataFrame) -> None:
    data["score"] = data["avg_faithfulness"] * data["avg_relevancy"]
    max_score = data["score"].max()

    best_rows = data[data["score"] == max_score]

    print(f"\n>>> Best configurations with max score ({max_score:.5f}):")

    if best_rows.empty:
        print("  [!] No configuration found with the highest score.")
    else:
        for _, row in best_rows.iterrows():
            print(
                f"  - LLM: {row['llm']:<10} | "
                f"Chunk size: {int(row['chunk_size'])} | "
                f"Overlap: {row['overlap']:.2f} | "
                f"Faithfulness: {row['avg_faithfulness']:.5f} | "
                f"Relevancy: {row['avg_relevancy']:.5f} | "
                f"Score: {row['score']:.5f}"
            )

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
        if metric.startswith("avg_") and metric != "avg_time":
            vmin, vmax = compute_global_vmin_vmax(data, metric)
        else:
            vmin = vmax = None
        for emb in data["llm"].unique():
            generate_heatmap(
                data,
                metric=metric,
                embedder=emb,
                vmin=vmin,
                vmax=vmax
            )

    print_best_joint_configs(data)

if __name__ == "__main__":
    fire.Fire(main)
