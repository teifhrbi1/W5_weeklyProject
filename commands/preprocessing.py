from pathlib import Path
import click

from utils.data_handler import load_csv, validate_columns
from utils.arabic_text import clean_arabic_text, load_stopwords, remove_stopwords, normalize_arabic

@click.group(name="preprocess")
def preprocess():
    """Preprocessing commands."""
    pass

@preprocess.command("remove")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def remove(csv_path: Path, text_col: str, output: Path):
    df = load_csv(csv_path)
    validate_columns(df, [text_col])

    before = df[text_col].astype(str)
    before_mean_words = before.apply(lambda x: len(x.split())).mean()

    df[text_col] = before.apply(clean_arabic_text)

    after = df[text_col].astype(str)
    after_mean_words = after.apply(lambda x: len(x.split())).mean()

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")

    click.echo("Preprocess: remove")
    click.echo(f"Output: {output}")
    click.echo(f"Mean words before: {before_mean_words:.2f}")
    click.echo(f"Mean words after : {after_mean_words:.2f}")

@preprocess.command("stopwords")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def stopwords(csv_path: Path, text_col: str, output: Path):
    df = load_csv(csv_path)
    validate_columns(df, [text_col])

    stop_set = load_stopwords(Path("utils/stopwords_ar.txt"))

    before = df[text_col].astype(str)
    before_mean_words = before.apply(lambda x: len(x.split())).mean()

    df[text_col] = before.apply(lambda x: remove_stopwords(x, stop_set))

    after = df[text_col].astype(str)
    after_mean_words = after.apply(lambda x: len(x.split())).mean()

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")

    click.echo("Preprocess: stopwords")
    click.echo(f"Output: {output}")
    click.echo(f"Mean words before: {before_mean_words:.2f}")
    click.echo(f"Mean words after : {after_mean_words:.2f}")

@preprocess.command("replace")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def replace(csv_path: Path, text_col: str, output: Path):
    df = load_csv(csv_path)
    validate_columns(df, [text_col])

    before = df[text_col].astype(str)
    before_mean_words = before.apply(lambda x: len(x.split())).mean()

    df[text_col] = before.apply(normalize_arabic)

    after = df[text_col].astype(str)
    after_mean_words = after.apply(lambda x: len(x.split())).mean()

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")

    click.echo("Preprocess: replace")
    click.echo(f"Output: {output}")
    click.echo(f"Mean words before: {before_mean_words:.2f}")
    click.echo(f"Mean words after : {after_mean_words:.2f}")

@preprocess.command("all")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def all_steps(csv_path: Path, text_col: str, output: Path):
    df = load_csv(csv_path)
    validate_columns(df, [text_col])

    stop_set = load_stopwords(Path("utils/stopwords_ar.txt"))

    before = df[text_col].astype(str)
    before_mean_words = before.apply(lambda x: len(x.split())).mean()

    df[text_col] = (
        before
        .apply(clean_arabic_text)
        .apply(lambda x: remove_stopwords(x, stop_set))
        .apply(normalize_arabic)
    )

    after = df[text_col].astype(str)
    after_mean_words = after.apply(lambda x: len(x.split())).mean()

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")

    click.echo("Preprocess: all")
    click.echo(f"Output: {output}")
    click.echo(f"Mean words before: {before_mean_words:.2f}")
    click.echo(f"Mean words after : {after_mean_words:.2f}")
