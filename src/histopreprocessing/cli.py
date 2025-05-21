import click
from pathlib import Path

from histopreprocessing.histoqc import run_histoqc_task
from histopreprocessing.rename_masks import rename_masks_task, write_wsi_paths_to_csv
from histopreprocessing.tiling import (
    tile_wsi_task,
    tile_wsi_superpixel_task_random_overlap,
    tile_wsi_superpixel_task_no_overlap,
)
from histopreprocessing.metadata import write_tiles_metadata_task
from histopreprocessing.utils import configure_logging
from histopreprocessing.superpixels import superpixel_segmentation_task
from histopreprocessing.superpixel_mapping import create_superpixel_tile_mapping_task

project_dir = Path(__file__).parents[2].resolve()


def validate_is_json(ctx, param, value):
    allowed_extensions = {".json"}  # Adjust as needed
    if value and not value.lower().endswith(tuple(allowed_extensions)):
        raise click.BadParameter(
            f"File must have one of the following extensions: {', '.join(allowed_extensions)}"
        )
    return value


@click.group()
@click.option(
    "--log-file", type=click.Path(), help="Optional file path to save log output."
)
def cli(log_file):
    """Command-line interface for executing histopathology pre-processing tasks."""
    configure_logging(log_file=log_file)


@cli.command()
@click.option(
    "--raw-wsi-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing the raw WSI files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory where output files will be saved.",
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    help="Number of parallel worker processes.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to a configuration file with custom settings."
    "If not provided a default config will be chosen.",
)
@click.option(
    "--search-pattern",
    type=click.STRING,
    default="*.svs",
    show_default=True,
    help="Filename pattern to filter input WSIs.",
)
@click.option(
    "--wsi-id-mapping-style",
    type=click.STRING,
    default="TCGA",
    show_default=True,
    help="Identifier mapping style for WSIs."
    "For now the only possibilites or TCGA or CPTAC.",
)
@click.option(
    "--force",
    is_flag=True,
    show_default=True,
    default=False,
    help="Force rerun of HistoQC even if output files already exist.",
)
def run_histoqc(
    raw_wsi_dir,
    output_dir,
    num_workers,
    config,
    search_pattern,
    wsi_id_mapping_style,
    force,
):
    """Execute HistoQC on the specified dataset to save mask and generate a CSV with raw WSI paths."""
    raw_wsi_dir = Path(raw_wsi_dir)
    output_dir = Path(output_dir)
    run_histoqc_task(
        raw_wsi_dir,
        output_dir,
        config,
        num_workers=num_workers,
        search_pattern=search_pattern,
        wsi_id_mapping_style=wsi_id_mapping_style,
        force=force,
    )
    output_csv = output_dir / "raw_wsi_path.csv"

    write_wsi_paths_to_csv(
        raw_wsi_dir,
        output_dir,
        output_csv,
        wsi_id_mapping_style,
    )


@cli.command()
@click.option(
    "--masks-dir",
    type=click.Path(exists=True),
)
@click.option(
    "--wsi-id-mapping-style",
    type=click.STRING,
    default="TCGA",
    show_default=True,
    help="Identifier mapping style used for renaming mask folders.",
)
def rename_masks(masks_dir, wsi_id_mapping_style):
    """Rename mask directories based on the provided WSI identifier mapping style."""
    rename_masks_task(masks_dir, wsi_id_mapping_style)


@cli.command()
@click.option(
    "--raw-wsi-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing the raw WSI files.",
)
@click.option(
    "--masks-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing HistoQC output masks.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Destination directory for generated tiles.",
)
@click.option(
    "--tile-size",
    "-t",
    type=int,
    default=224,
    show_default=True,
    help="Size (in pixels) of each tile.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Minimum mask coverage threshold.",
)
@click.option(
    "--magnification",
    type=int,
    default=10,
    show_default=True,
    help="Magnification level for tiling.",
)
@click.option(
    "--num-workers-tiles",
    default=1,
    show_default=True,
    help="Number of parallel worker processes to process tiles.",
)
@click.option(
    "--num-workers-wsi",
    default=1,
    show_default=True,
    help="Number of parallel worker processes to process WSIs.",
)
@click.option(
    "--save-overlay",
    is_flag=True,
    show_default=True,
    default=True,
    help="Save overlay images for visualizing tile boundaries.",
)
@click.option(
    "--save-masks",
    is_flag=True,
    show_default=True,
    default=True,
    help="Save generated mask tiles.",
)
@click.option(
    "--wsi-id-mapping-style",
    type=click.STRING,
    default="TCGA",
    show_default=True,
    help="Identifier mapping style used for renaming mask folders.",
)
def tile_wsi(
    raw_wsi_dir,
    masks_dir,
    output_dir,
    tile_size,
    threshold,
    magnification,
    num_workers_tiles,
    num_workers_wsi,
    save_overlay,
    save_masks,
    wsi_id_mapping_style,
):
    """Generate tiles from whole slide images (WSIs) using HistoQC mask outputs."""
    tile_wsi_task(
        raw_wsi_dir,
        masks_dir,
        output_dir,
        tile_size=tile_size,
        threshold=threshold,
        num_workers_tiles=num_workers_tiles,
        num_workers_wsi=num_workers_wsi,
        save_tile_overlay=save_overlay,
        save_masks=save_masks,
        magnification=magnification,
        wsi_id_mapping_style=wsi_id_mapping_style,
    )


@cli.command()
@click.option(
    "--tiles-dir",
    type=click.Path(exists=True),
)
@click.option(
    "--output-json",
    type=click.Path(),
    required=True,
    callback=validate_is_json,
    help="Destination JSON file path for saving the tiles path mapping.",
)
def write_tiles_metadata(tiles_dir, output_json):
    """Generate metadata for the tiles present in the specified dataset directory."""
    write_tiles_metadata_task(tiles_dir, output_json)


@cli.command()
@click.option(
    "--raw-wsi-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing raw whole slide images.",
)
@click.option(
    "--masks-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing the corresponding mask images.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Directory to save the segmented superpixel outputs.",
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    help="Number of parallel worker processes.",
)
@click.option(
    "--average-tile-size",
    default=672,
    show_default=True,
    help="Average tile size (in pixels) used for segmentation.",
)
@click.option(
    "--save-overlay",
    is_flag=True,
    show_default=True,
    default=True,
    help="Save overlay images for superpixel segmentation.",
)
def superpixel_segmentation(
    raw_wsi_dir,
    masks_dir,
    output_dir,
    num_workers,
    average_tile_size,
    save_overlay,
):
    """Perform superpixel segmentation on WSIs and generate tile overlays."""
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
    help="Directory containing raw whole slide images.",
)
@click.option(
    "--superpixel-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with superpixel segmentation outputs.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Destination directory for the tiled output.",
)
@click.option(
    "--tile-size",
    "-t",
    type=int,
    default=224,
    show_default=True,
    help="Size (in pixels) of each generated tile.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Minimum mask coverage threshold.",
)
@click.option(
    "--magnification",
    type=int,
    default=10,
    show_default=True,
    help="Magnification level for tiling.",
)
@click.option(
    "--num-workers-tiles",
    default=1,
    show_default=True,
    help="Number of parallel worker processes to process tiles.",
)
@click.option(
    "--num-workers-wsi",
    default=1,
    show_default=True,
    help="Number of parallel worker processes to process WSIs.",
)
@click.option(
    "--save-overlay",
    is_flag=True,
    show_default=True,
    default=True,
    help="Save overlay images of the generated tiles.",
)
@click.option(
    "--save-masks",
    is_flag=True,
    show_default=True,
    default=True,
    help="Save the tiled mask images.",
)
@click.option(
    "--wsi-id-mapping-style",
    type=click.STRING,
    default="TCGA",
    show_default=True,
    help="Identifier mapping style used for renaming mask folders.",
)
def tile_wsi_from_superpixel_no_overlap(
    raw_wsi_dir,
    superpixel_dir,
    output_dir,
    tile_size,
    threshold,
    magnification,
    num_workers_tiles,
    num_workers_wsi,
    save_overlay,
    save_masks,
    wsi_id_mapping_style,
):
    """Generate non-overlapping tiles from WSIs using superpixel segmentation outputs."""
    tile_wsi_superpixel_task_no_overlap(
        raw_wsi_dir,
        superpixel_dir,
        output_dir,
        tile_size=tile_size,
        threshold=threshold,
        num_workers_wsi=num_workers_wsi,
        num_workers_tiles=num_workers_tiles,
        save_tile_overlay=save_overlay,
        save_masks=save_masks,
        magnification=magnification,
        wsi_id_mapping_style=wsi_id_mapping_style,
    )


@cli.command()
@click.option(
    "--raw-wsi-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing raw whole slide images.",
)
@click.option(
    "--superpixel-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with superpixel segmentation outputs.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Destination directory for the tiled output.",
)
@click.option(
    "--tile-size",
    "-t",
    type=int,
    default=224,
    show_default=True,
    help="Size (in pixels) of each tile.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Minimum mask coverage threshold.",
)
@click.option(
    "--magnification",
    type=int,
    default=10,
    show_default=True,
    help="Magnification level for tiling.",
)
@click.option(
    "--num-workers-tiles",
    default=1,
    show_default=True,
    help="Number of parallel worker processes to process tiles.",
)
@click.option(
    "--num-workers-wsi",
    default=1,
    show_default=True,
    help="Number of parallel worker processes to process WSIs.",
)
@click.option(
    "--save-overlay",
    is_flag=True,
    show_default=True,
    default=True,
    help="Save overlay images of the generated tiles.",
)
@click.option(
    "--average-superpixel-area",
    type=float,
    default=486800,
    show_default=True,
    help="Average area (in µm²) of superpixels should "
    "be computed on the whole dataset, used to "
    "compute the number of tiles to draw.",
)
@click.option(
    "--average-n-tiles",
    type=click.INT,
    default=25,
    show_default=True,
    help="Average number of tiles to extract per superpixel.",
)
@click.option(
    "--wsi-id-mapping-style",
    type=click.STRING,
    default="TCGA",
    show_default=True,
    help="Identifier mapping style used for renaming mask folders.",
)
def tile_wsi_from_superpixel_random_overlap(
    raw_wsi_dir,
    superpixel_dir,
    output_dir,
    tile_size,
    threshold,
    magnification,
    num_workers_tiles,
    num_workers_wsi,
    save_overlay,
    average_superpixel_area,
    average_n_tiles,
    wsi_id_mapping_style,
):
    """Generate tiles from WSIs using a random overlap method based on superpixel segmentation outputs."""
    tile_wsi_superpixel_task_random_overlap(
        raw_wsi_dir,
        superpixel_dir,
        output_dir,
        tile_size=tile_size,
        threshold=threshold,
        num_workers_tiles=num_workers_tiles,
        num_workers_wsi=num_workers_wsi,
        save_tile_overlay=save_overlay,
        magnification=magnification,
        average_superpixel_area=average_superpixel_area,
        average_n_tiles=average_n_tiles,
        wsi_id_mapping_style=wsi_id_mapping_style,
    )


@cli.command()
@click.option(
    "--tiles-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing tile images.",
)
@click.option(
    "--output-json",
    type=click.Path(),
    required=True,
    callback=validate_is_json,
    help="Destination JSON file path for saving the tiles path mapping as well as superpixels metadata.",
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    help="Number of parallel worker processes.",
)
def create_superpixel_tile_mapping(tiles_dir, output_json, num_workers):
    """Create a mapping between superpixels and their corresponding tiles and save it as a JSON file."""
    create_superpixel_tile_mapping_task(tiles_dir, output_json, num_workers=num_workers)


if __name__ == "__main__":
    cli()
