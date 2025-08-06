from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import pandas as pd
from evaluations.cache import PATH as CACHE_PATH
from evaluations.plots import PATH as PLOT_PATH
def load_all_embeddings(folder: str) -> pd.DataFrame:
    """Load all embeddings CSVs and concatenate them into a single DataFrame."""
    rows = []
    for path in Path(folder).glob("*.csv"):
        if "embeddings" not in path.name:
            continue
        if "prompt" not in path.name and "mx" not in path.name:
            continue

        print(f"Loading {path}")
        df = pd.read_csv(path)
        # Extract embedding type from filename
        embedding_type = path.stem.split("__")[-1]
        df['embedding'] = embedding_type
        rows.append((path, df))

    return rows


print("Loading all embeddings")
embeddings_info = load_all_embeddings(CACHE_PATH)

data_by_model_kind = {}

for (path, df) in embeddings_info:
    print(f"Loaded {len(df)} rows from {df['embedding'].iloc[0]}")

    # convert all columns to a single row of floats and drop the last column
    row = df.values[:, :-1]
    scaler = StandardScaler()
    row_scaled = scaler.fit_transform(row)
    pca = PCA(n_components=2)
    row_pca = pca.fit_transform(row_scaled)
    print("PCA components shape:", row_pca.shape)

    # extract model and kind from filename
    fname = path.stem.replace("embeddings__", "")
    kind = fname.split("__")[0]
    if kind == "prompt":
        kind = fname.split("__")[1]
        llm = fname.split("__")[2].replace("_embeddings", "")
    else:
        llm = fname.split("__")[1]
        kind = fname.split("__")[0]

    print(llm)
    key = (llm, kind)
    if key not in data_by_model_kind:
        data_by_model_kind[key] = []
    data_by_model_kind[key].append(row_pca)

# determine unique models and kinds for the grid
models = sorted(list({k for k, _ in data_by_model_kind.keys()}))
kinds = sorted(list({k for _, k in data_by_model_kind.keys()}))

fig, axs = plt.subplots(len(models), len(kinds), figsize=(5 * len(kinds), 5 * len(models)))
if len(models) == 1 and len(kinds) == 1:
    axs = [[axs]]
elif len(models) == 1:
    axs = [axs]
elif len(kinds) == 1:
    axs = [[ax] for ax in axs]

for i, model in enumerate(models):
    for j, kind in enumerate(kinds):
        ax = axs[i][j]
        key = (model, kind)
        if key in data_by_model_kind:
            # combine all PCA arrays for that model and kind
            combined = np.concatenate(data_by_model_kind[key], axis=0)
            x = combined[:, 0]
            y = combined[:, 1]
            kde = gaussian_kde([x, y])
            xi, yi = np.mgrid[-40:40:300j, -40:40:300j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            ax.imshow(zi, extent=[-40, 40, -40, 40], origin='lower', cmap='Blues', aspect='auto')
            ax.set_title(f"{model} - {kind}")
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            # set x and y limits to the fixed window
            ax.set_xlim(-40, 40)
            ax.set_ylim(-40, 40)
        else:
            ax.set_visible(False)

plt.tight_layout()
plt.savefig(PLOT_PATH / "kde_grid_plot.png")

