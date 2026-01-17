from pathlib import Path
import math
import click
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

from utils.data_handler import load_csv, validate_columns
from utils.arabic_text import clean_arabic_text, load_stopwords, remove_stopwords, normalize_arabic

@click.command(name="pipeline")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--label_col", type=str, required=True)
@click.option("--max_features", type=int, default=5000, show_default=True)
@click.option("--test_size", type=float, default=0.2, show_default=True)
@click.option("--out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs"), show_default=True)
def pipeline(csv_path: Path, text_col: str, label_col: str, max_features: int, test_size: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    validate_columns(df, [text_col, label_col])

    stop_set = load_stopwords(Path("utils/stopwords_ar.txt"))

    df[text_col] = (
        df[text_col].astype(str)
        .apply(clean_arabic_text)
        .apply(lambda x: remove_stopwords(x, stop_set))
        .apply(normalize_arabic)
    )

    y = df[label_col].astype(str).values

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[text_col].astype(str).tolist())

    emb_path = out_dir / "embeddings" / "tfidf.pkl"
    joblib.dump({"X": X, "vectorizer": vectorizer, "text_col": text_col}, emb_path)

    counts = pd.Series(y).value_counts()
    classes = counts.index.astype(str).tolist()
    n_samples = int(len(y))
    n_classes = int(len(classes))

    n_test_est = int(math.ceil(test_size * n_samples)) if test_size < 1 else int(test_size)

    use_stratify = True
    if int(counts.min()) < 2:
        use_stratify = False
    if n_test_est < n_classes:
        use_stratify = False

    strat = y if use_stratify else None
    if not use_stratify:
        click.echo(f"Warning: stratify disabled. n_test={n_test_est}, n_classes={n_classes}, counts={counts.to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=strat
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    model_path = out_dir / "models" / "pipeline_best_model.pkl"
    joblib.dump({"model": model, "vectorizer": vectorizer, "label_col": label_col}, model_path)

    report_path = out_dir / "reports" / "pipeline_report.md"
    report_path.write_text(
        f"# Pipeline Report\n\n"
        f"- CSV: {csv_path}\n"
        f"- Samples: {n_samples}\n"
        f"- Classes: {n_classes} ({', '.join(classes)})\n"
        f"- Test size: {test_size} (estimated n_test={n_test_est})\n"
        f"- Stratify: {use_stratify}\n"
        f"- Features: {X.shape[1]}\n\n"
        f"## Metrics (Logistic Regression)\n"
        f"- Accuracy: {acc:.4f}\n"
        f"- Precision: {prec:.4f}\n"
        f"- Recall: {rec:.4f}\n"
        f"- F1: {f1:.4f}\n\n"
        f"## Outputs\n"
        f"- Embeddings: {emb_path}\n"
        f"- Model: {model_path}\n",
        encoding="utf-8"
    )

    click.echo("Pipeline done")
    click.echo(f"Saved embeddings -> {emb_path}")
    click.echo(f"Saved model -> {model_path}")
    click.echo(f"Saved report -> {report_path}")
