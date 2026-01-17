from pathlib import Path
import click

from utils.data_handler import load_csv, validate_columns
from utils.visualization import save_distribution_plot, save_histogram_plot

from commands.wordcloud_cmd import wordcloud

@click.group()
def eda():
    """EDA commands."""
    pass

@eda.command("distribution")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--label_col", type=str, required=True)
@click.option("--plot_type", type=click.Choice(["pie", "bar"]), default="pie", show_default=True)
def distribution(csv_path: Path, label_col: str, plot_type: str):
    df = load_csv(csv_path)
    validate_columns(df, [label_col])

    labels = df[label_col]
    counts = labels.value_counts(dropna=False)
    total = int(counts.sum())

    click.echo("EDA: Class Distribution")
    click.echo(f"CSV: {csv_path}")
    click.echo(f"Label col: {label_col}")
    click.echo(f"Total samples: {total}")
    for cls, cnt in counts.items():
        pct = (cnt / total) * 100 if total else 0
        click.echo(f"- {cls}: {int(cnt)} ({pct:.2f}%)")

    out_path = save_distribution_plot(counts, label_col, plot_type)
    click.echo(f"Saved plot -> {out_path}")

@eda.command("histogram")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--unit", type=click.Choice(["words", "chars"]), default="words", show_default=True)
def histogram(csv_path: Path, text_col: str, unit: str):
    df = load_csv(csv_path)
    validate_columns(df, [text_col])

    texts = df[text_col].astype(str)
    lengths = texts.apply(lambda x: len(x.split())) if unit == "words" else texts.apply(lambda x: len(x))

    click.echo("EDA: Text Length Histogram")
    click.echo(f"CSV: {csv_path}")
    click.echo(f"Text col: {text_col}")
    click.echo(f"Unit: {unit}")
    click.echo(f"Count: {len(lengths)}")
    click.echo(f"Mean: {lengths.mean():.2f}")
    click.echo(f"Median: {lengths.median():.2f}")
    click.echo(f"Std: {lengths.std(ddof=1):.2f}")
    click.echo(f"Min: {lengths.min():.0f}")
    click.echo(f"Max: {lengths.max():.0f}")

    out_path = save_histogram_plot(lengths, text_col, unit)
    click.echo(f"Saved plot -> {out_path}")

eda.add_command(wordcloud)
