"""Exploratory Data Analysis: summary stats, correlation heatmap, distributions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config
from .preprocessing import load_data

def run():
    config.OUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(df.describe(include="all").T); print("\nMissing values:\n", df.isna().sum())
    num = df.select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)): ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im); plt.title("Correlation heatmap"); plt.tight_layout()
    plt.savefig(config.OUT_DIR / "correlation_heatmap.png", dpi=120); plt.close()
    num.hist(bins=30, figsize=(12, 8)); plt.tight_layout()
    plt.savefig(config.OUT_DIR / "distributions.png", dpi=120); plt.close()

if __name__ == "__main__":
    run()
