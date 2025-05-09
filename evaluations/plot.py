from evaluations.plots import PATH as PLOTS_PATH
from evaluations.cache import PATH as CACHE_PATH
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from evaluations.plots import PATH as PLOTS_PATH
from evaluations.cache import PATH as CACHE_PATH
import pandas as pd
import os
from matplotlib.patches import Patch
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

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

def merge_dataframes(folder: str) -> pd.DataFrame:
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
        method_order = ["role_playing", "full", "vector_store", "vector_rerank", "hybrid"]
        df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
        # sort the method in this way: "role_playing, full, vector_store, vector_rerank, hybrid"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

    # Enhanced visualization functions with improved styling and separate scales
def plot_distributions(df: pd.DataFrame) -> None:
    """
    Plot the distribution of scores across different methods using boxplots.
    Each subplot uses its own appropriate scale based on the metric.
    """
    # Set aesthetic parameters
    plt.rcParams.update({'font.size': 10, 'figure.figsize': (14, 10)})
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    
    # Create FacetGrid with separate y-scales
    g = sns.FacetGrid(df, col="model", row="metric", height=3.5, aspect=1.2, 
                        margin_titles=True, sharey=False)
    
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
    plt.savefig(PLOTS_PATH / "distribution_scores.png", dpi=300, bbox_inches='tight')
    plt.close()


# Load and prepare the data
df = merge_dataframes(CACHE_PATH)
# Transform to long format for plotting
df = df.melt(id_vars=["kind", "method", "model", "embedding"], 
                var_name="metric", value_name="score")

# Generate plots
plot_distributions(df)