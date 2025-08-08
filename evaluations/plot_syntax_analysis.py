import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity

from evaluations.plots import PATH as PLOTS_PATH

# Define the syntax data path
SYNTAX_PATH = Path(__file__).parent / "syntax"

EMBEDDERS = [
    "role_playing",
    "full",
    "vector_store",
    "vector_rerank",
    "bm25",
    "hybrid",
]

PRETTY_NAMES = {
    "gemma3-1b": "Gemma 3 (1B)",
    "granite3.1-moe:1b": "Granite 3.1 MoE (1B)",
    "falcon3-1b": "Falcon 3 (1B)",
    "qwen3-0.6b": "Qwen 3 (0.6B)",
    "deepseek-r1-1.5b": "DeepSeek R1 (1.5B)",
    "llama3.2-1b": "Llama 3.2 (1B)",
    "smollm2-1.7b": "SmolLM2 (1.7B)",
    "qwen2.5-0.5b": "Qwen 2.5 (0.5B)",
    "role_playing": "role-playing prompt",
    "hybrid": "hybrid",
    "vector_store": "vector_search",
    "vector_rerank": "vector_rerank",
    "full": "full-context prompt",
    "bm25": "bm25",
}

SYNTAX_METRICS = {
    "ttr": "Type-Token Ratio",
    "gulpease": "Gulpease Index",
    "flesch_vacca": "Flesch-Vacca Index",
    "lexical_density": "Lexical Density"
}


def merge_syntax_dataframes(folder: str, embedder: str = "nomic") -> pd.DataFrame:
    """Load all syntax CSV files, extract kind/method/model/embedding, and concat."""
    rows = []
    for path in Path(folder).glob("*.csv"):
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
        method_order = [PRETTY_NAMES[x] for x in EMBEDDERS]
        df["method"] = df["method"].map(PRETTY_NAMES).fillna(df["method"])
        df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
        
        # Filter if the kind is rag and the embedding is not the desired one
        if kind == "rag" and embedding != embedder:
            continue
        
        # Explicitly filter out llama3.2-3b model
        if model == "llama3.2-3b":
            continue
            
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def plot_syntax_distributions(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Plot the distribution of syntax scores across different methods using boxplots.
    Each subplot uses its own appropriate scale based on the metric.
    """
    # Set aesthetic parameters
    plt.rcParams.update({'font.size': 10, 'figure.figsize': (16, 12)})
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})

    # Melt the dataframe to have all metrics in one column
    df_melted = df.melt(
        id_vars=["kind", "method", "model", "embedding"],
        value_vars=["ttr", "gulpease", "flesch_vacca", "lexical_density"],
        var_name="metric", 
        value_name="score"
    )
    
    # Convert 'score' to numeric, forcing errors to NaN
    df_melted["score"] = pd.to_numeric(df_melted["score"], errors="coerce")
    
    # Create FacetGrid with shared y-scales within each row
    g = sns.FacetGrid(
        df_melted, 
        col="model", 
        row="metric", 
        height=3.5, 
        aspect=1.2,
        margin_titles=True, 
        sharey='row'
    )

    # Map boxplot with custom palette
    g.map_dataframe(sns.boxplot, x="method", y="score", palette="Set2")

    # Enhance appearance and readability
    g.set_axis_labels("Method", "Score")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    # Rotate x-axis labels and add max value for each row
    for i, row_axes in enumerate(g.axes):
        metric_name = df_melted['metric'].unique()[i]
        metric_pretty_name = SYNTAX_METRICS.get(metric_name, metric_name)
        
        # Calculate the max value for the entire row (across all models with this metric)
        row_data = df_melted[df_melted['metric'] == metric_name]
        row_max = row_data['score'].max() if not row_data.empty else 0
        
        for j, ax in enumerate(row_axes):
            if i == 0:  # Only set model titles on the first row
                model_name = df_melted['model'].unique()[j] if j < len(df_melted['model'].unique()) else ""
                model_pretty_name = PRETTY_NAMES.get(model_name, model_name)
                ax.set_title(model_pretty_name)
            
            if j == 0:  # Only set metric labels on the first column
                ax.set_ylabel(metric_pretty_name)
            
            # Rotate x-axis labels by 45 degrees
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Save high-quality figure
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"syntax_distributions_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_syntax_metric_comparison(df: pd.DataFrame, metric: str, embedder: str = "nomic", models_per_row: int = 4) -> None:
    """
    Plot only the distribution of a specific syntax metric across different models and methods.
    """
    metric_df = df.copy()
    metric_df = metric_df[metric_df["model"].isin(PRETTY_NAMES.keys())].copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df["model"] = metric_df["model"].map(PRETTY_NAMES).fillna(metric_df["model"])
    sorted_model_names = sorted(metric_df["model"].unique())
    metric_df["model"] = pd.Categorical(metric_df["model"], categories=sorted_model_names, ordered=True)

    plt.rcParams.update({'font.size': 13, 'figure.figsize': (models_per_row * 6, 20)})
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})

    g = sns.FacetGrid(
        metric_df,
        col="model",
        col_wrap=models_per_row,
        height=3.5,
        aspect=1.2,
        sharey=True
    )

    labels = [PRETTY_NAMES[label] for label in EMBEDDERS]

    def barplot_fixed_order(data, **kwargs):
        ax = plt.gca()
        sns.barplot(
            data=data,
            x="method",
            y=metric,
            order=labels,
            palette="Set2",
            ax=ax,
            width=0.7,
            estimator=np.mean,
            ci=95
        )

    g.map_dataframe(barplot_fixed_order, x="method", y=metric, palette="Set2", width=0.7)
    g.set_axis_labels("", SYNTAX_METRICS.get(metric, metric))
    g.set_titles(col_template="{col_name}")

    for ax, model in zip(g.axes.flat, g.col_names):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"syntax_{metric}_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_syntax_correlations(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Plot correlation matrix between different syntax metrics across methods and models.
    """
    # Calculate mean scores for each combination
    mean_scores = df.groupby(['kind', 'method', 'model'])[['ttr', 'gulpease', 'flesch_vacca', 'lexical_density']].mean()
    
    # Compute correlation matrix
    corr_matrix = mean_scores.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt='.3f',
        square=True,
        xticklabels=[SYNTAX_METRICS[col] for col in corr_matrix.columns],
        yticklabels=[SYNTAX_METRICS[col] for col in corr_matrix.columns]
    )
    plt.title(f"Syntax Metrics Correlation Matrix ({embedder.capitalize()} embeddings)")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"syntax_correlation_matrix_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_syntax_method_comparison(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Plot comparison of syntax metrics across different methods, averaged across all models.
    """
    # Calculate mean scores for each method across all models
    method_means = df.groupby('method')[['ttr', 'gulpease', 'flesch_vacca', 'lexical_density']].mean()
    
    # Melt for plotting
    method_means_melted = method_means.reset_index().melt(
        id_vars=['method'], 
        var_name='metric', 
        value_name='score'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=method_means_melted,
        x='method',
        y='score',
        hue='metric',
        palette="Set2"
    )
    
    plt.title(f"Average Syntax Metrics by Method ({embedder.capitalize()} embeddings)")
    plt.xlabel("Method")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metric", labels=[SYNTAX_METRICS[m] for m in ['ttr', 'gulpease', 'flesch_vacca', 'lexical_density']])
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"syntax_method_comparison_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_syntax_improvement_over_baseline(df: pd.DataFrame, embedder: str = "nomic", baseline_method: str = "full") -> None:
    """
    Plot syntax metric improvements over the baseline method for each model.
    """
    metrics = ['ttr', 'gulpease', 'flesch_vacca', 'lexical_density']
    models = sorted(df["model"].unique())
    
    # Calculate improvements
    records = []
    for model in models:
        sub = df[df["model"] == model]
        
        for metric in metrics:
            baseline = sub.loc[
                (sub["kind"] == "prompt") & (sub["method"] == baseline_method), metric
            ].mean()
            
            if np.isnan(baseline):
                continue
                
            for _, row in sub.iterrows():
                if row["method"] == baseline_method:
                    continue
                    
                improvement = row[metric] - baseline
                records.append({
                    "model": row["model"],
                    "kind": row["kind"],
                    "method": row["method"],
                    "metric": metric,
                    "improvement": improvement
                })
    
    imp_df = pd.DataFrame(records)
    
    if imp_df.empty:
        print(f"No improvement data available for embedder {embedder}")
        return
    
    # Plot grid of improvements
    g = sns.FacetGrid(
        imp_df,
        col="model",
        row="metric",
        height=3,
        aspect=1.2,
        sharey='row'
    )
    
    g.map_dataframe(sns.barplot, x="method", y="improvement", palette="Set2")
    
    # Rotate x-axis labels
    for ax in g.axes.flatten():
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    g.set_axis_labels("Method", "Improvement")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    # Color bars based on improvement direction
    for ax in g.axes.flatten():
        for bar in ax.patches:
            bar.set_color("green" if bar.get_height() >= 0 else "red")
    
    # Custom legend
    handles = [
        Patch(color="green", label="Positive"),
        Patch(color="red", label="Negative")
    ]
    g.fig.legend(handles=handles, title="Improvement Sign", loc="upper right")
    
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"syntax_improvement_over_{baseline_method}_{embedder}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for embedder in ("nomic", "mxbai"):
        print(f"\n=== Processing {embedder} embeddings ===")
        
        # Load and prepare the data
        try:
            df = merge_syntax_dataframes(SYNTAX_PATH, embedder=embedder)
            
            if df.empty:
                print(f"No data found for {embedder} embeddings")
                continue
            
            print(f"Loaded {len(df)} rows for {embedder}")
            print(f"Available models: {sorted(df['model'].unique())}")
            print(f"Available methods: {sorted(df['method'].unique())}")
            
            # Generate all plots
            print("Generating syntax distributions plot...")
            plot_syntax_distributions(df, embedder=embedder)
            
            print("Generating individual metric plots...")
            for metric in ['ttr', 'gulpease', 'flesch_vacca', 'lexical_density']:
                plot_syntax_metric_comparison(df, metric, embedder=embedder, models_per_row=4)
            
            print("Generating correlation matrix...")
            plot_syntax_correlations(df, embedder=embedder)
            
            print("Generating method comparison plot...")
            plot_syntax_method_comparison(df, embedder=embedder)
            
            print("Generating improvement over baseline plot...")
            plot_syntax_improvement_over_baseline(df, embedder=embedder, baseline_method="full")
            
            print(f"All plots generated for {embedder} embeddings!")
            
        except Exception as e:
            print(f"Error processing {embedder}: {e}")
            import traceback
            traceback.print_exc()
