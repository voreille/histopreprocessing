import click
from pathlib import Path

from histopreprocessing.histoqc import run_histoqc_task
from histopreprocessing.rename_masks import rename_masks_task, write_wsi_paths_to_csv
from histopreprocessing.tiling import (tile_wsi_task,
                                       tile_wsi_superpixel_task_random_overlap,
                                       tile_wsi_superpixel_task_no_overlap)
from histopreprocessing.metadata import write_tiles_metadata_task
from histopreprocessing.utils import configure_logging
from histopreprocessing.superpixels import superpixel_segmentation_task
from histopreprocessing.superpixel_mapping import create_superpixel_tile_mapping_task

project_dir = Path(__file__).parents[2].resolve()


@click.group()
@click.option("--log-file", type=click.Path(), help="Path to log file.")
def cli(log_file):
    """CLI for histopreprocessing tasks."""
    configure_logging(log_file=log_file)


@cli.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True),
)
@click.option(
    "--output-dir",
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
    default=None,
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
    default="TCGA",
)
@click.option("--force",
              is_flag=True,
              show_default=True,
              default=False,
              help="To rerun HistoQC if the files already exist.")
def run_histoqc(input_dir, output_dir, num_workers, config, search_pattern,
                wsi_id_mapping_style, force):
    """Run HistoQC on the specified dataset."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    run_histoqc_task(
        input_dir,
        output_dir,
        config,
        num_workers=num_workers,
        search_pattern=search_pattern,
        wsi_id_mapping_style=wsi_id_mapping_style,
        force=force,
    )
    output_csv = output_dir / "raw_wsi_path.csv"

    write_wsi_paths_to_csv(
        input_dir,
        output_dir,
        output_csv,
        wsi_id_mapping_style,
    )


@cli.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True),
)
@click.option(
    "--wsi-id-mapping-style",
    type=click.STRING,
    default="TCGA",
)
def rename_masks(input_dir, wsi_id_mapping_style):
    """Rename mask folders."""
    rename_masks_task(input_dir, wsi_id_mapping_style)


@cli.command()
@click.option(
    "--masks-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the output of HistoQC",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to the output directory",
)
@click.option(
    "--tile-size",
    "-t",
    type=int,
    default=224,
    show_default=True,
    help="Tile size",
)
@click.option("--threshold",
              type=float,
              default=0.8,
              show_default=True,
              help="Threshold for mask coverage")
@click.option(
    "--magnification",
    type=int,
    default=10,
    show_default=True,
    help="Magnification",
)
@click.option("--num-workers",
              default=1,
              show_default=True,
              help="Number of workers")
@click.option("--save-overlay",
              is_flag=True,
              show_default=True,
              default=True,
              help="Wether to save overlays of the tiles")
@click.option("--save-masks",
              is_flag=True,
              show_default=True,
              default=True,
              help="wether to save the tiled masks")
def tile_wsi(
    masks_dir,
    output_dir,
    tile_size,
    threshold,
    magnification,
    num_workers,
    save_overlay,
    save_masks,
):
    """Tile WSIs for the dataset."""
    tile_wsi_task(
        masks_dir,
        output_dir,
        tile_size=tile_size,
        threshold=threshold,
        num_workers_wsi=1,
        num_workers_tiles=num_workers,
        save_tile_overlay=save_overlay,
        save_masks=save_masks,
        magnification=magnification,
    )


@cli.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True),
)
def write_tiles_metadata(input_dir):
    write_tiles_metadata_task(input_dir)


@cli.command()
@click.option(
    "--raw-wsi-dir",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--masks-dir",
    type=click.Path(exists=True),
    required=True,
    help="",
)
@click.option(
    "--output-dir",
    required=True,
    help="",
)
@click.option("--num-workers",
              default=1,
              show_default=True,
              help="Number of workers")
@click.option("--average-tile-size",
              default=672,
              show_default=True,
              help="Number of workers")
@click.option("--save-overlay",
              is_flag=True,
              show_default=True,
              default=True,
              help="Wether to save overlays of the tiles")
def superpixel_segmentation(
    raw_wsi_dir,
    masks_dir,
    output_dir,
    num_workers,
    average_tile_size,
    save_overlay,
):
    """Tile WSIs for the dataset."""

    superpixel_segmentation_task(
        raw_data_dir=raw_wsi_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        save_overlay=save_overlay,
        average_tile_size=average_tile_size,
        num_workers=num_workers,
    )


@cli.command()
@click.option(
    "--raw-wsi-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the output of HistoQC",
)
@click.option(
    "--superpixel-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the output of HistoQC",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to the output directory",
)
@click.option(
    "--tile-size",
    "-t",
    type=int,
    default=224,
    show_default=True,
    help="Tile size",
)
@click.option("--threshold",
              type=float,
              default=0.8,
              show_default=True,
              help="Threshold for mask coverage")
@click.option(
    "--magnification",
    type=int,
    default=10,
    show_default=True,
    help="Magnification",
)
@click.option("--num-workers",
              default=1,
              show_default=True,
              help="Number of workers")
@click.option("--save-overlay",
              is_flag=True,
              show_default=True,
              default=True,
              help="Wether to save overlays of the tiles")
@click.option("--save-masks",
              is_flag=True,
              show_default=True,
              default=True,
              help="wether to save the tiled masks")
def tile_wsi_from_superpixel_no_overlap(
    raw_wsi_dir,
    superpixel_dir,
    output_dir,
    tile_size,
    threshold,
    magnification,
    num_workers,
    save_overlay,
    save_masks,
):
    """Tile WSIs for the dataset."""
    tile_wsi_superpixel_task_no_overlap(
        raw_wsi_dir,
        superpixel_dir,
        output_dir,
        tile_size=tile_size,
        threshold=threshold,
        num_workers_wsi=1,
        num_workers_tiles=num_workers,
        save_tile_overlay=save_overlay,
        save_masks=save_masks,
        magnification=magnification,
    )


@cli.command()
@click.option(
    "--raw-wsi-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the output of HistoQC",
)
@click.option(
    "--superpixel-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the output of HistoQC",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to the output directory",
)
@click.option(
    "--tile-size",
    "-t",
    type=int,
    default=224,
    show_default=True,
    help="Tile size",
)
@click.option("--threshold",
              type=float,
              default=0.8,
              show_default=True,
              help="Threshold for mask coverage")
@click.option(
    "--magnification",
    type=int,
    default=10,
    show_default=True,
    help="Magnification",
)
@click.option("--num-workers",
              default=1,
              show_default=True,
              help="Number of workers")
@click.option("--save-overlay",
              is_flag=True,
              show_default=True,
              default=True,
              help="Wether to save overlays of the tiles")
@click.option("--average-superpixel-area",
              type=float,
              default=486800,
              show_default=True,
              help="Average superpixel area in micrometer "
              "present in the dataset, used in the "
              "computation to draw less from small superpixel")
@click.option("--average-n-tiles",
              type=click.INT,
              default=25,
              show_default=True,
              help="Average number of tiles to draw from each superpixel.")
def tile_wsi_from_superpixel_random_overlap(
    raw_wsi_dir,
    superpixel_dir,
    output_dir,
    tile_size,
    threshold,
    magnification,
    num_workers,
    save_overlay,
    average_superpixel_area,
    average_n_tiles,
):
    """Tile WSIs for the dataset."""
    tile_wsi_superpixel_task_random_overlap(
        raw_wsi_dir,
        superpixel_dir,
        output_dir,
        tile_size=tile_size,
        threshold=threshold,
        num_workers_wsi=1,
        num_workers_tiles=num_workers,
        save_tile_overlay=save_overlay,
        magnification=magnification,
        average_superpixel_area=average_superpixel_area,
        average_n_tiles=average_n_tiles,
    )


def validate_is_json(ctx, param, value):
    allowed_extensions = {".json"}  # Adjust as needed
    if value and not value.lower().endswith(tuple(allowed_extensions)):
        raise click.BadParameter(
            f"File must have one of the following extensions: {', '.join(allowed_extensions)}"
        )
    return value


@cli.command()
@click.option(
    "--tiles-dir",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output-file",
    type=click.Path(),
    required=True,
    callback=validate_is_json,
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    help="Number of workers for parallel processing",
)
def create_superpixel_tile_mapping(tiles_dir, output_file, num_workers):
    create_superpixel_tile_mapping_task(tiles_dir,
                                        output_file,
                                        num_workers=num_workers)


if __name__ == "__main__":
    cli()
