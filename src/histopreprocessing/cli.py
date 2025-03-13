import click
from pathlib import Path

from histopreprocessing.histoqc import run_histoqc_task
from histopreprocessing.rename_masks import rename_masks_task
from histopreprocessing.tiling import tile_wsi_task
from histopreprocessing.metadata import write_tiles_metadata
from histopreprocessing.utils import configure_logging

project_dir = Path(__file__).parents[2].resolve()


@click.group()
@click.option("--log-file", type=click.Path(), help="Path to log file.")
def cli(log_file):
    """CLI for histopreprocessing tasks."""
    configure_logging(log_file=log_file)


@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to directory containing the raw WSI to run histoqc on.",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the output directory",
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    help="Number of workers for parallel processing",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to the configuration file",
)
@click.option(
    "--search-pattern",
    type=click.STRING,
    default="*.svs",
    help="String pattern to search for in input-dir.",
)
@click.option(
    "--wsi-id-mapping-style",
    type=click.STRING,
    default="*.svs",
    help="Pick among {}",
)
def run_histoqc(input_dir, output_dir, num_workers, config, search_pattern,
                wsi_id_mapping_style):
    """Run HistoQC on the specified dataset."""
    run_histoqc_task(input_dir, output_dir, num_workers, config,
                     search_pattern, wsi_id_mapping_style)


@cli.command()
@click.argument("dataset")
@click.option("--config",
              "-c",
              type=click.Path(exists=True),
              help="Path to the configuration file")
def rename_masks(dataset, config):
    """Rename mask folders."""
    rename_masks_task(dataset, config)


@cli.command()
@click.argument("dataset")
@click.option("--tile-size", "-t", type=int, help="Tile size")
@click.option("--threshold",
              "-th",
              type=float,
              help="Threshold for mask coverage")
@click.option("--magnification", "-m", type=int, help="Magnification")
@click.option("--num-workers",
              "-n",
              default=1,
              show_default=True,
              help="Number of workers")
@click.option("--config",
              "-c",
              type=click.Path(exists=True),
              help="Path to the configuration file")
@click.option("--debug-id", "-d", help="Debugging WSI ID")
def tile_wsi(dataset, tile_size, threshold, magnification, num_workers, config,
             debug_id):
    """Tile WSIs for the dataset."""
    tile_wsi_task(dataset, tile_size, threshold, magnification, num_workers,
                  config, debug_id)


@cli.command()
@click.option("--config",
              "-c",
              type=click.Path(exists=True),
              help="Path to the configuration file")
def write_metadata(config):
    """Store metadata about tiles."""
    write_tiles_metadata(config)


if __name__ == "__main__":
    cli()
