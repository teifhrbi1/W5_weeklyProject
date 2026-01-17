from pathlib import Path
import click
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import arabic_reshaper
from bidi.algorithm import get_display

from utils.data_handler import load_csv, validate_columns

def shape_arabic_text(text: str) -> str:
    tokens = str(text).split()
    shaped = [get_display(arabic_reshaper.reshape(t)) for t in tokens]
    return " ".join(shaped)

@click.command(name="wordcloud")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text_col", type=str, required=True)
@click.option("--label_col", type=str, default=None)
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs/visualizations/wordclouds"))
@click.option("--font_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("/System/Library/Fonts/Supplemental/GeezaPro.ttc"))
def wordcloud(csv_path: Path, text_col: str, label_col: str | None, output_dir: Path, font_path: Path):
    df = load_csv(csv_path)
    validate_columns(df, [text_col] + ([label_col] if label_col else []))

    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_cloud(text: str, out_path: Path):
        text = shape_arabic_text(text)
        wc = WordCloud(
            width=1200,
            height=800,
            background_color="white",
            font_path=str(font_path),
            collocations=False,
        ).generate(text)

        fig = plt.figure()
        ax = plt.gca()
        ax.imshow(wc)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    if label_col:
        for cls, sub in df.groupby(label_col):
            text = " ".join(sub[text_col].astype(str).tolist())
            out_path = output_dir / f"wordcloud_{cls}.png"
            _save_cloud(text, out_path)
            click.echo(f"Saved -> {out_path}")
    else:
        text = " ".join(df[text_col].astype(str).tolist())
        out_path = output_dir / "wordcloud_all.png"
        _save_cloud(text, out_path)
        click.echo(f"Saved -> {out_path}")
