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

# Define the readability data path
READABILITY_PATH = Path(__file__).parent / "readability"

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

# Readability level interpretation based on the prompt in syntax_analysis_metrics.py
READABILITY_CATEGORIES = {
    "Very Easy (90-100)": (90, 100),
    "Easy (70-89)": (70, 89),
    "Moderate (50-69)": (50, 69),
    "Difficult (30-49)": (30, 49),
    "Very Difficult (0-29)": (0, 29)
}


def merge_readability_dataframes(folder: str, embedder: str = "nomic") -> pd.DataFrame:
    """Load all readability CSV files, extract kind/method/model/embedding, and concat."""
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


def plot_readability_distributions(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Plot the distribution of readability scores across different methods and models using boxplots.
    """
    # Set aesthetic parameters
    plt.rcParams.update({'font.size': 10, 'figure.figsize': (16, 8)})
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})

    # Convert 'readability_level' to numeric, forcing errors to NaN
    df["readability_level"] = pd.to_numeric(df["readability_level"], errors="coerce")
    
    # Create FacetGrid with shared y-scale
    g = sns.FacetGrid(
        df, 
        col="model", 
        height=4, 
        aspect=1.2,
        margin_titles=True, 
        sharey=True,
        col_wrap=4
    )

    # Map boxplot with custom palette
    g.map_dataframe(sns.boxplot, x="method", y="readability_level", palette="Set2")

    # Enhance appearance and readability
    g.set_axis_labels("Method", "Readability Level (0-100)")
    g.set_titles(col_template="{col_name}")

    # Apply model pretty names to titles and rotate x-axis labels
    for ax in g.axes.flat:
        # Get the current title and map it to pretty name
        current_title = ax.get_title()
        model_name = current_title.replace("model = ", "")
        pretty_name = PRETTY_NAMES.get(model_name, model_name)
        ax.set_title(pretty_name)
        
        # Rotate x-axis labels by 45 degrees
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Set y-limits to cover the full readability scale
        ax.set_ylim(0, 100)

    # Save high-quality figure
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"readability_distributions_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_readability_by_method(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Plot average readability scores by method across all models.
    """
    # Calculate mean scores for each method across all models
    method_means = df.groupby('method', observed=False)['readability_level'].agg(['mean', 'std']).reset_index()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(method_means['method'], method_means['mean'], 
                   yerr=method_means['std'], capsize=5)
    
    # Color bars based on readability level
    for i, (bar, mean_val) in enumerate(zip(bars, method_means['mean'])):
        if mean_val >= 70:
            bar.set_color('green')
        elif mean_val >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.title(f"Average Readability Level by Method ({embedder.capitalize()} embeddings)")
    plt.xlabel("Method")
    plt.ylabel("Average Readability Level (0-100)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Add horizontal lines for readability categories
    plt.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Easy (70+)')
    plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate (50-69)')
    plt.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Difficult (<50)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"readability_by_method_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_readability_by_model(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Plot average readability scores by model across all methods.
    """
    # Calculate mean scores for each model across all methods
    df_filtered = df[df["model"].isin(PRETTY_NAMES.keys())].copy()
    df_filtered["model"] = df_filtered["model"].map(PRETTY_NAMES)
    model_means = df_filtered.groupby('model', observed=False)['readability_level'].agg(['mean', 'std']).reset_index()
    model_means = model_means.sort_values('mean', ascending=True)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(model_means['model'], model_means['mean'], 
                    xerr=model_means['std'], capsize=5)
    
    # Color bars based on readability level
    for i, (bar, mean_val) in enumerate(zip(bars, model_means['mean'])):
        if mean_val >= 70:
            bar.set_color('green')
        elif mean_val >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.title(f"Average Readability Level by Model ({embedder.capitalize()} embeddings)")
    plt.ylabel("Model")
    plt.xlabel("Average Readability Level (0-100)")
    plt.xlim(0, 100)
    
    # Add vertical lines for readability categories
    plt.axvline(x=70, color='green', linestyle='--', alpha=0.7, label='Easy (70+)')
    plt.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Moderate (50-69)')
    plt.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Difficult (<50)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"readability_by_model_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_readability_heatmap(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Create a heatmap showing mean readability scores for each model-method combination.
    """
    # Calculate mean scores for each model-method combination
    df_filtered = df[df["model"].isin(PRETTY_NAMES.keys())].copy()
    df_filtered["model"] = df_filtered["model"].map(PRETTY_NAMES)
    
    pivot_table = df_filtered.pivot_table(
        values='readability_level', 
        index='model', 
        columns='method', 
        aggfunc='mean'
    )
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        cmap="RdYlGn",
        center=50,
        fmt='.1f',
        cbar_kws={'label': 'Readability Level (0-100)'},
        vmin=0,
        vmax=100
    )
    plt.title(f"Readability Level Heatmap by Model and Method ({embedder.capitalize()} embeddings)")
    plt.ylabel("Model")
    plt.xlabel("Method")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"readability_heatmap_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_readability_improvement_over_baseline(df: pd.DataFrame, embedder: str = "nomic", baseline_method: str = "full") -> None:
    """
    Plot readability improvements over the baseline method for each model.
    """
    models = sorted(df[df["model"].isin(PRETTY_NAMES.keys())]["model"].unique())
    
    # Calculate improvements
    records = []
    for model in models:
        sub = df[df["model"] == model]
        
        baseline = sub.loc[
            (sub["kind"] == "prompt") & (sub["method"] == baseline_method), "readability_level"
        ].mean()
        
        if np.isnan(baseline):
            continue
            
        for method in sub["method"].unique():
            if method == baseline_method:
                continue
                
            method_mean = sub[sub["method"] == method]["readability_level"].mean()
            improvement = method_mean - baseline
            records.append({
                "model": model,
                "method": method,
                "improvement": improvement
            })
    
    imp_df = pd.DataFrame(records)
    
    if imp_df.empty:
        print(f"No improvement data available for embedder {embedder}")
        return
    
    # Apply pretty names to models
    imp_df["model"] = imp_df["model"].map(PRETTY_NAMES)
    
    # Plot grid of improvements
    plt.figure(figsize=(14, 8))
    
    # Create pivot table for heatmap
    pivot_imp = imp_df.pivot(index='model', columns='method', values='improvement')
    
    sns.heatmap(
        pivot_imp,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt='.1f',
        cbar_kws={'label': 'Readability Improvement'},
        linewidths=0.5
    )
    
    plt.title(f"Readability Improvement over '{baseline_method}' Method ({embedder.capitalize()} embeddings)")
    plt.ylabel("Model")
    plt.xlabel("Method")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"readability_improvement_over_{baseline_method}_{embedder}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_readability_categories(df: pd.DataFrame, embedder: str = "nomic") -> None:
    """
    Plot the distribution of texts across readability categories.
    """
    # Define readability categories
    def categorize_readability(score):
        if score >= 90:
            return "Very Easy (90-100)"
        elif score >= 70:
            return "Easy (70-89)"
        elif score >= 50:
            return "Moderate (50-69)"
        elif score >= 30:
            return "Difficult (30-49)"
        else:
            return "Very Difficult (0-29)"
    
    df["readability_category"] = df["readability_level"].apply(categorize_readability)
    
    # Calculate percentages for each method
    category_counts = df.groupby(['method', 'readability_category'], observed=False).size().unstack(fill_value=0)
    category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100
    
    # Create stacked bar plot
    plt.figure(figsize=(12, 8))
    category_percentages.plot(kind='bar', stacked=True, 
                             color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    
    plt.title(f"Distribution of Readability Categories by Method ({embedder.capitalize()} embeddings)")
    plt.xlabel("Method")
    plt.ylabel("Percentage of Texts")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Readability Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"readability_categories_{embedder}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    for embedder in ("nomic", "mxbai"):
        print(f"\n=== Processing {embedder} embeddings ===")
        
        # Load and prepare the data
        try:
            df = merge_readability_dataframes(READABILITY_PATH, embedder=embedder)
            
            if df.empty:
                print(f"No data found for {embedder} embeddings")
                continue
            
            print(f"Loaded {len(df)} rows for {embedder}")
            print(f"Available models: {sorted(df['model'].unique())}")
            print(f"Available methods: {sorted(df['method'].unique())}")
            
            # Generate all plots
            print("Generating readability distributions plot...")
            plot_readability_distributions(df, embedder=embedder)
            
            print("Generating readability by method plot...")
            plot_readability_by_method(df, embedder=embedder)
            
            print("Generating readability by model plot...")
            plot_readability_by_model(df, embedder=embedder)
            
            print("Generating readability heatmap...")
            plot_readability_heatmap(df, embedder=embedder)
            
            print("Generating readability categories distribution...")
            plot_readability_categories(df, embedder=embedder)
            
            print("Generating improvement over baseline plot...")
            plot_readability_improvement_over_baseline(df, embedder=embedder, baseline_method="full")
            
            print(f"All readability plots generated for {embedder} embeddings!")
            
        except Exception as e:
            print(f"Error processing {embedder}: {e}")
            import traceback
            traceback.print_exc()
