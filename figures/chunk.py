import os

from analysis import PATH as ANALYSIS_PATH
from figures import PATH as FIGURES_PATH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fire
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import logging

CHUNK_ANALYSIS_FILES = [ANALYSIS_PATH / file for file in os.listdir(ANALYSIS_PATH) if file.endswith(".csv")]


def generate_2d_graph(data: pd.DataFrame, name: str = "all") -> None:
    # create a vectorial (.eps) and a rusted graph (.png) from the chunk analysis file
    # Plot dots in a 2D space where the axis are the average faithfulness and relevancy
    # Each dot has a color (from green to red) based on the average latency
    # Each dot has a label (chunk size and overlap ratio)

    fig, ax = plt.subplots(figsize=(16, 9))
    cmap = plt.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=data["avg_time"].min(), vmax=data["avg_time"].max())
    sm = ScalarMappable(cmap=cmap, norm=norm)

    coords = data.groupby(["avg_faithfulness", "avg_relevancy"])

    group_number = 0
    group_cmap = plt.get_cmap('tab10').colors

    for (x, y), group in coords:
        n = len(group)

        if n == 1:
            row = group.iloc[0]
            color = cmap(norm(row["avg_time"]))
            ax.scatter(x, y, color=color, s=100)
            if name == "all":
                ax.annotate(f"({row['llm'][0]}, {int(row['chunk_size'])}, {int(row['overlap'] * 100)})",
                            (x, y), fontsize=10, ha='center', va='bottom')
            else:
                ax.annotate(f"({row['chunk_size']}, {int(row['overlap'] * 100)})", (x, y), fontsize=10,
                            ha='center', va='bottom')
        else:
            group_color = group_cmap[group_number % len(group_cmap)]
            group_number += 1
            angles = np.linspace(11 / 12 * np.pi, 19 / 12 * np.pi, n, endpoint=False)
            if name == 'all':
                radius = 0.006 + 0.003 * group_number
            else:
                radius = 0.006 * group_number

            for angle, (_, row) in zip(angles, group.iterrows()):
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                x_new, y_new = x + dx, y + dy

                color = cmap(norm(row["avg_time"]))
                ax.scatter(x_new, y_new, color=color, s=100, zorder=2)
                ax.plot([x, x_new], [y, y_new], color=group_color, linestyle='--', linewidth=0.8, zorder=1)
                if name == "all":
                    ax.annotate(f"({row['llm'][0]}, {int(row['chunk_size'])}, {int(row['overlap'] * 100)})", (x_new, y_new), fontsize=10,
                                ha='center', va='bottom')
                else:
                    ax.annotate(f"({int(row['chunk_size'])}, {int(row['overlap'] * 100)})", (x_new, y_new-0.001), fontsize=10,
                            ha='center', va='bottom')

            ax.scatter(x, y, color=group_color, s=50, zorder=1, marker='X')

    ax.set_xlabel("Average Faithfulness", fontsize=16)
    ax.set_ylabel("Average Relevancy", fontsize=16)
    ax.set_title(f"Embedding Evaluation {name}", fontsize=20)
    ax.set_xlim(data["avg_faithfulness"].min() - 0.01, min(data["avg_faithfulness"].max() + 0.1, 1.001))
    ax.set_ylim(data["avg_relevancy"].min() - 0.01, min(data["avg_relevancy"].max() + 0.1, 1.005))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Average Latency (s)", fontsize=16)
    # Save the figure
    plt.savefig(FIGURES_PATH / f"chunk_evaluation_{name}_2d.png")
    plt.savefig(FIGURES_PATH / f"chunk_evaluation_{name}_2d.eps")
    plt.close()


def generate_bar_graph(data: pd.DataFrame, annotate: bool = False) -> None:
    # create a vectorial (.eps) and a rusted graph (.png) from the chunk analysis file
    # Plot dots in a bar graph where for each embedding 3 bars are plotted (average faithfulness, average relevancy, average latency)
    # Each dot has a color (from green to red) based on the average latency
    # Each dot has a label (chunk size and overlap ratio)
    fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.5), 6))
    bar_width = 0.25
    bar_positions = np.arange(len(data))

    # Normalize latency for color mapping
    bars1 = ax.bar(bar_positions - bar_width, data["avg_faithfulness"], width=bar_width, label="Average Faithfulness",color='blue')
    bars2 = ax.bar(bar_positions, data["avg_relevancy"], width=bar_width, label="Average Relevancy", color='orange')
    bars3 = ax.bar(bar_positions + bar_width, data["avg_time"], width=bar_width, label="Average Latency", color='green')

    xtick_labels = [f"({row['llm'][0]}, {int(row['chunk_size'])}, {int(row['overlap'] * 100)})" for _, row in data.iterrows()]
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    if annotate:
        annotate_bars(bars1)
        annotate_bars(bars2)
        annotate_bars(bars3)

    ax.set_xlabel("Chunk Size and Overlap Ratio")
    ax.set_ylabel("Value")
    ax.set_title("Chunk Evaluation")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    # Save the figure
    plt.savefig(FIGURES_PATH / "chunk_evaluation_bar.png")
    plt.savefig(FIGURES_PATH / "chunk_evaluation_bar.eps")
    plt.close()


def main():
    # Read all data from the files
    # Name is the third word of the file name
    llm_names = [file.name.split("_")[2][:-4] for file in CHUNK_ANALYSIS_FILES]
    data_dfs = {name : pd.read_csv(file) for name, file in zip(llm_names, CHUNK_ANALYSIS_FILES)}
    # add a column "llm" to each dataframe with the name of the llm
    for name, df in data_dfs.items():
        generate_2d_graph(df, name=name)
        df["llm"] = name
    # merge all dataframes into one
    data = pd.concat(data_dfs.values(), ignore_index=True)
    generate_2d_graph(data)
    generate_bar_graph(data)

    # get best(s) embedding configuration considering the highest average faithfulness and relevancy (and lowest average latency in case of a tie)
    best_faithfulness = data["avg_faithfulness"].max()
    best_relevancy = data["avg_relevancy"].max()
    best_configs = data[
        (data["avg_faithfulness"] == best_faithfulness) &
        (data["avg_relevancy"] == best_relevancy)
    ]
    best_configs = best_configs.sort_values(by="avg_time", ascending=True)
    logging.info(f"Best configurations (highest average faithfulness and relevancy):")
    for _, row in best_configs.iterrows():
        logging.info(f"model={row['llm']}")
        logging.info(f"chunk_size={int(row['chunk_size'])}")
        logging.info(f"overlap_ratio={int(row['overlap'] * 100)}")
        logging.info(f"average_faithfulness={row['avg_faithfulness']}")
        logging.info(f"average_relevancy={row['avg_relevancy']}")
        logging.info(f"average_latency={row['avg_time']}")
        logging.info("----------------------")



if __name__ == "__main__":
    fire.Fire(main)