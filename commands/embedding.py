from pathlib import Path
import click
import joblib
import numpy as np

from utils.data_handler import load_csv, validate_columns

@click.group(name="embed")
def embed():
    """Embedding commands."""
    pass

def _sparse_nbytes(X) -> int:
    return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)

@embed.command("tfidf")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--max_features", type=int, default=5000, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def tfidf(csv_path: Path, text_col: str, max_features: int, output: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = load_csv(csv_path)
    validate_columns(df, [text_col])

    texts = df[text_col].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "X": X,
            "vectorizer": vectorizer,
            "text_col": text_col,
            "max_features": max_features,
            "csv_path": str(csv_path),
        },
        output,
    )

    nnz = int(X.nnz)
    shape = X.shape
    mem_mb = _sparse_nbytes(X) / (1024 * 1024)

    click.echo("Embed: TF-IDF")
    click.echo(f"Embedding shape: {shape[0]} x {shape[1]}")
    click.echo(f"Non-zeros (nnz): {nnz}")
    click.echo(f"Approx sparse size: {mem_mb:.2f} MB")
    click.echo(f"Saved -> {output}")

def _require_torch():
    try:
        import torch  # noqa
    except Exception as e:
        raise click.ClickException(
            "Torch is required for this embedding. Install bonus deps:\n"
            "  pip install -r requirements-embeddings-bonus.txt\n"
            f"Original error: {e}"
        )

def _device_from_string(device: str):
    import torch
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)

def _mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted

def _embed_with_transformers(texts, model_name: str, batch_size: int, device: str):
    _require_torch()
    import torch
    from transformers import AutoTokenizer, AutoModel

    dev = _device_from_string(device)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(dev)
    mdl.eval()

    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = mdl(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_emb.append(pooled.detach().cpu().numpy().astype(np.float32))
            if (i // batch_size) % 20 == 0:
                click.echo(f"Progress: {min(i+batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_emb)

def _embed_with_sentence_transformers(texts, model_name: str, batch_size: int, device: str):
    _require_torch()
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        # sentence-transformers expects "cuda"/"cpu"/"mps" depending on environment
        device = "mps" if _device_from_string("auto").type == "mps" else "cuda" if _device_from_string("auto").type == "cuda" else "cpu"

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return emb.astype(np.float32)

def _save_dense(output: Path, X: np.ndarray, meta: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": X, **meta}, output)
    mb = (X.nbytes / (1024 * 1024))
    click.echo(f"Embedding shape: {X.shape[0]} x {X.shape[1]}")
    click.echo(f"Approx size: {mb:.2f} MB")
    click.echo(f"Saved -> {output}")

@embed.command("bert")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--model", type=str, default="UBC-NLP/ARBERTv2", show_default=True)
@click.option("--batch_size", type=int, default=32, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto", show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def bert(csv_path: Path, text_col: str, model: str, batch_size: int, device: str, output: Path):
    df = load_csv(csv_path)
    validate_columns(df, [text_col])
    texts = df[text_col].astype(str).tolist()

    click.echo(f"Embed: BERT (transformers) | model={model} | device={device}")
    X = _embed_with_transformers(texts, model_name=model, batch_size=batch_size, device=device)

    _save_dense(
        output,
        X,
        {
            "text_col": text_col,
            "csv_path": str(csv_path),
            "embedder": {"type": "bert", "model": model, "pooling": "mean"},
        },
    )

@embed.command("sentence-transformer")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", show_default=True)
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto", show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def sentence_transformer(csv_path: Path, text_col: str, model: str, batch_size: int, device: str, output: Path):
    df = load_csv(csv_path)
    validate_columns(df, [text_col])
    texts = df[text_col].astype(str).tolist()

    click.echo(f"Embed: Sentence-Transformer | model={model} | device={device}")
    X = _embed_with_sentence_transformers(texts, model_name=model, batch_size=batch_size, device=device)

    _save_dense(
        output,
        X,
        {
            "text_col": text_col,
            "csv_path": str(csv_path),
            "embedder": {"type": "sentence-transformer", "model": model},
        },
    )

@embed.command("model2vec")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--model", type=str, default="JadwalAlmaa/model2vec-ARBERTv2", show_default=True)
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto", show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def model2vec(csv_path: Path, text_col: str, model: str, batch_size: int, device: str, output: Path):
    """
    Model2Vec embedding (best-effort loader):
    - First tries SentenceTransformer(model)
    - If it fails, falls back to transformers mean pooling.
    """
    df = load_csv(csv_path)
    validate_columns(df, [text_col])
    texts = df[text_col].astype(str).tolist()

    click.echo(f"Embed: Model2Vec | model={model} | device={device}")

    X = None
    try:
        X = _embed_with_sentence_transformers(texts, model_name=model, batch_size=batch_size, device=device)
        loader = "sentence-transformers"
    except Exception:
        X = _embed_with_transformers(texts, model_name=model, batch_size=max(8, batch_size // 2), device=device)
        loader = "transformers(mean-pool)"

    _save_dense(
        output,
        X,
        {
            "text_col": text_col,
            "csv_path": str(csv_path),
            "embedder": {"type": "model2vec", "model": model, "loader": loader},
        },
    )
