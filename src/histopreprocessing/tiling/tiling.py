import logging
from pathlib import Path
from multiprocessing.pool import ThreadPool, Pool

import pandas as pd
from tqdm import tqdm

from .wsi_tiler import WSITilerWithMask

logger = logging.getLogger(__name__)


def process_wsi(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=10,
    tile_size=224,
    threshold=0.8,
    num_workers_tiles=12,
    save_masks=False,
    save_tile_overlay=False,
    save_metadata=True,
):
    logger.info(f"Starting tiling for WSI {wsi_path.name}")
    try:
        tile_processor = WSITilerWithMask(
            wsi_path,
            mask_path,
            output_dir,
            magnification=magnification,
            tile_size=tile_size,
            threshold=threshold,
            save_masks=save_masks,
            save_tile_overlay=save_tile_overlay,
        )

        coordinates = tile_processor.get_coordinates()

        if num_workers_tiles > 1:
            with ThreadPool(processes=num_workers_tiles) as pool:
                results = list(
                    tqdm(pool.imap_unordered(tile_processor, coordinates),
                         total=len(coordinates),
                         desc="Processing tiles"))
        else:
            results = [tile_processor(coord) for coord in tqdm(coordinates)]

        if any(isinstance(res, Exception) for res in results):
            raise RuntimeError(
                f"Error encountered in tile processing for {wsi_path.name}")

        logger.info(f"Tiling completed for WSI {wsi_path.name}")

    except Exception as e:
        logger.error(f"Error processing WSI {wsi_path.name}: {e}")
        return False

    if save_tile_overlay:
        tile_processor.save_overlay()
    if save_metadata:
        tile_processor.save_metadata()
    return True


def tile_wsi_task(
    masks_dir,
    output_dir,
    tile_size=224,
    threshold=0.8,
    num_workers_wsi=4,
    num_workers_tiles=12,
    save_tile_overlay=False,
    save_masks=False,
    magnification=10,
):
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    mask_files = [f for f in masks_dir.rglob("*mask_use.png")]
    df = pd.read_csv(Path(masks_dir) / "raw_wsi_path.csv")
    wsi_paths_mapping = {
        row["WSI_ID"]: Path(row["Path"])
        for _, row in df.iterrows()
    }

    # Create a list of arguments for parallel processing
    wsi_args = [
        (
            wsi_paths_mapping[mask_path.stem.split(".")[0]],  # WSI path
            mask_path,  # Mask path
            output_dir,
            magnification,
            tile_size,
            threshold,
            num_workers_tiles,
            save_masks,  # save_mask
            save_tile_overlay,
        ) for mask_path in mask_files
    ]

    with Pool(processes=num_workers_wsi) as pool:
        results = list(
            tqdm(pool.starmap(process_wsi, wsi_args),
                 total=len(wsi_args),
                 desc="Processing WSIs"))

    for wsi_path, result in zip(wsi_paths_mapping.values(), results):
        if not result:
            logger.warning(f"Processing failed for WSI {wsi_path.name}")
