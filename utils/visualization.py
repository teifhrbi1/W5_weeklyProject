from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def save_distribution_plot(counts: pd.Series, label_col: str, plot_type: str) -> Path:
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = plt.gca()

    if plot_type == "pie":
        ax.pie(counts.values, labels=[str(x) for x in counts.index], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Class Distribution ({label_col})")
    elif plot_type == "bar":
        ax.bar([str(x) for x in counts.index], counts.values)
        ax.set_title(f"Class Distribution ({label_col})")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.xticks(rotation=20, ha="right")
    else:
        raise ValueError("plot_type must be pie or bar")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"dist_{label_col}_{plot_type}_{ts}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def save_histogram_plot(lengths: pd.Series, text_col: str, unit: str) -> Path:
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = plt.gca()
    ax.hist(lengths.dropna().values, bins=30)
    ax.set_title(f"Text Length Histogram ({text_col})")
    ax.set_xlabel(f"Length ({unit})")
    ax.set_ylabel("Frequency")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"hist_{text_col}_{unit}_{ts}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
