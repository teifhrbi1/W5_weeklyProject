from pathlib import Path
from datetime import datetime

import click
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils.data_handler import load_csv, validate_columns

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dirs():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)

def _plot_confmat(cm, labels, title, out_path: Path):
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _get_model(name: str, n_train: int):
    name = name.lower()
    if name == "knn":
        k = max(1, min(5, n_train))
        return KNeighborsClassifier(n_neighbors=k), f"k={k}"
    if name == "lr":
        return LogisticRegression(max_iter=2000), ""
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=42), ""
    raise ValueError("Unknown model. Use knn/lr/rf")

@click.command(name="train")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--label_col", type=str, required=True)
@click.option("--embeddings_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--test_size", type=float, default=0.2, show_default=True)
@click.option("--models", multiple=True, type=click.Choice(["knn", "lr", "rf"]), default=("knn", "lr", "rf"))
@click.option("--save_model", type=click.Path(dir_okay=False, path_type=Path), default=None)
def train(csv_path: Path, label_col: str, embeddings_path: Path, test_size: float, models, save_model):
    _ensure_dirs()

    df = load_csv(csv_path)
    validate_columns(df, [label_col])

    pack = joblib.load(embeddings_path)
    X = pack["X"]
    vectorizer = pack.get("vectorizer", None)  # TF-IDF case
    embedder = pack.get("embedder", None)      # BERT/SBERT/Model2Vec case

    y = df[label_col].astype(str).values
    counts = pd.Series(y).value_counts()
    classes = sorted(counts.index.astype(str).tolist())

    n_samples = int(len(y))
    n_classes = int(len(classes))
    n_test_est = int(round(test_size * n_samples))
    min_count = int(counts.min())

    use_stratify = True
    if min_count < 2:
        use_stratify = False
    if n_test_est < n_classes:
        use_stratify = False

    stratify_arg = y if use_stratify else None
    if not use_stratify:
        click.echo(f"Warning: stratify disabled. Class counts: {counts.to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_arg
    )

    ts = _ts()
    report_path = Path("outputs/reports") / f"training_report_{ts}.md"

    results = []
    best = None

    n_train = int(len(y_train))

    for m in models:
        model, extra = _get_model(m, n_train)
        if extra:
            click.echo(f"Model {m}: {extra}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_path = Path("outputs/visualizations") / f"confmat_{m}_{ts}.png"
        _plot_confmat(cm, classes, f"Confusion Matrix - {m}", cm_path)

        results.append((m, acc, prec, rec, f1, model, str(cm_path)))

        if best is None or f1 > best[4]:
            best = results[-1]

    lines = []
    lines.append(f"## Training Report - {ts}\n\n")
    lines.append("### Dataset Info\n")
    lines.append(f"- Total samples: {n_samples}\n")
    lines.append(f"- Test size: {test_size}\n")
    lines.append(f"- Stratify: {use_stratify}\n")
    lines.append(f"- Class counts: {counts.to_dict()}\n")
    lines.append(f"- Features: {X.shape[1] if hasattr(X, 'shape') else 'N/A'}\n")
    if embedder is not None:
        lines.append(f"- Embedding: {embedder}\n")
    lines.append("\n### Model Performance\n\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for (m, acc, prec, rec, f1, _, _) in results:
        lines.append(f"| {m} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n")
    lines.append("\n### Confusion Matrices\n\n")
    for (m, _, _, _, _, _, cm_path) in results:
        lines.append(f"- {m}: {cm_path}\n")
    lines.append(f"\n### Best Model: {best[0]} ⭐ (F1={best[4]:.4f})\n")

    report_path.write_text("".join(lines), encoding="utf-8")

    click.echo(f"Report saved -> {report_path}")
    click.echo(f"Best model -> {best[0]} (F1={best[4]:.4f})")

    if save_model is not None:
        save_model.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": best[5], "label_col": label_col, "classes": classes}
        if vectorizer is not None:
            payload["vectorizer"] = vectorizer
        if embedder is not None:
            payload["embedder"] = embedder
        joblib.dump(payload, save_model)
        click.echo(f"Saved best model package -> {save_model}")
