import click

from commands.eda import eda
from commands.preprocessing import preprocess
from commands.embedding import embed
from commands.training import train
from commands.pipeline import pipeline

@click.group()
def cli():
    """Arabic NLP Classification CLI Tool."""
    pass

cli.add_command(eda)
cli.add_command(preprocess)
cli.add_command(embed)
cli.add_command(train)
cli.add_command(pipeline)

if __name__ == "__main__":
    cli()
