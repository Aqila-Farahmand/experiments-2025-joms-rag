import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from analysis.chi_square import PATH as ANALYSIS_PATH
from evaluations.cache import PATH as CACHE_PATH
from evaluations.plot import merge_dataframes, PRETTY_NAMES


import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from matplotlib.colors import ListedColormap
from evaluations.plot import PRETTY_NAMES
from pathlib import Path


def chi_square_test_by_model(df: pd.DataFrame, alpha: float = 0.05, output_folder=ANALYSIS_PATH, embedder: str = "nomic") -> dict:
    """
    For each model, perform chi-square tests on all pairs of methods using g_eval scores (0 or 1).
    Outputs significance tables as LaTeX and annotated heatmaps of p-values.
    """
    df = df[df["metric"] == "g_eval"].copy()
    df = df[df["model"].isin(PRETTY_NAMES.keys())].copy()
    df["model"] = df["model"].map(PRETTY_NAMES)

    method_ids = ["role_playing", "full", "vector_store", "vector_rerank", "hybrid"]
    methods = [PRETTY_NAMES[m] for m in method_ids]
    results = {}

    output_folder.mkdir(parents=True, exist_ok=True)

    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model].copy()
        model_df = model_df[model_df["method"].isin(PRETTY_NAMES.values())].copy()

        pvals = pd.DataFrame(index=methods, columns=methods, dtype=float)

        for m1, m2 in itertools.combinations(methods, 2):
            scores1 = model_df[model_df["method"] == m1]["score"]
            scores2 = model_df[model_df["method"] == m2]["score"]

            if scores1.empty or scores2.empty:
                p = float("nan")
            else:
                c1 = scores1.value_counts().reindex([0, 1], fill_value=0)
                c2 = scores2.value_counts().reindex([0, 1], fill_value=0)
                contingency = [c1.tolist(), c2.tolist()]

                try:
                    _, p, _, _ = chi2_contingency(contingency)
                except Exception:
                    p = float("1.0")

            pvals.loc[m1, m2] = p
            pvals.loc[m2, m1] = p

        for m in methods:
            pvals.loc[m, m] = float("nan")

        if pvals.isnull().all().all():
            print(f"Skipping {model} — all comparisons are NaN.")
            continue

        results[model] = pvals

        # Formatting p-values as strings
        fmt_pvals = pvals.apply(lambda col: col.map(lambda x: f"{x:.4f}" if pd.notna(x) else ""))

        # Binary mask: 1 = significant, 0 = non-significant
        sig_mask = pvals.apply(lambda col: col.map(lambda x: 1 if pd.notna(x) and x < alpha else 0 if pd.notna(x) else float("nan")))

        # Create color matrix: 0 = azzurro, 1 = rosso chiaro
        cmap = ListedColormap(["#cce5ff", "#f4cccc"])

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            sig_mask,
            mask=pvals.isna(),
            annot=fmt_pvals,
            fmt="",
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
            xticklabels=methods,
            yticklabels=methods,
            annot_kws={"color": "black", "fontsize": 9}
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.title(f"Chi-square p-values – {model} with {embedder} embedder", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_folder / f"chi2_significance_{model}_{embedder}.pdf", dpi=300)
        plt.close()

        # LaTeX table with bold for significant comparisons
        latex_table = pvals.apply(
            lambda col: col.map(
                lambda x: f"\\textbf{{{x:.4f}}}" if pd.notna(x) and x < alpha else f"{x:.4f}" if pd.notna(x) else ""
            )
        )
        latex_table.to_latex(output_folder / f"chi2_table_{model}_{embedder}.tex",
                             index=True, header=True, float_format="%.4f",
                             column_format="l" + "c" * len(methods),
                             escape=False)

    return results


if __name__ == "__main__":
    for embedder in ("nomic", "mxbai"):
        # Load and prepare the data
        df = merge_dataframes(CACHE_PATH, embedder=embedder)
        df = df.melt(id_vars=["kind", "method", "model", "embedding"],
                        var_name="metric", value_name="score")
        chi_square_test_by_model(df, output_folder=ANALYSIS_PATH, embedder=embedder)

