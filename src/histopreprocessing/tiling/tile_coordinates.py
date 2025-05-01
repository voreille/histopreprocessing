import logging
from itertools import product
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import numpy as np
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm

from histopreprocessing.wsi_id_mapping import WSI_ID_MAPPING_DICT

from ..utils import map_masks_to_wsi
from .contours import extract_patch_coords_from_mask
from .wsi_tiler import WSITilerWithMask

logger = logging.getLogger(__name__)

MAGNIFICTION_TO_MPP = {
    5: 2.0,
    10: 1.0,
    20: 0.5,
    40: 0.25,
}


def tile_coordinates_task(
    raw_wsi_dir,
    masks_dir,
    output_dir,
    tile_size=224,
    num_workers_wsi=4,
    num_workers_tiles=12,
    save_tile_overlay=False,
    save_masks=False,
    magnification=10,
    wsi_id_mapping_style="TCGA",
):
    raw_wsi_dir = Path(raw_wsi_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    mask_files = [f for f in masks_dir.rglob("*mask_use.png")]
    filename_to_wsi_id = WSI_ID_MAPPING_DICT[wsi_id_mapping_style]

    if len(mask_files) == 0:
        raise ValueError(f"No HistoQC masks found in {masks_dir}")

    test_mask_name = mask_files[0].name.removesuffix(".svs_mask_use.png")
    if test_mask_name != filename_to_wsi_id(mask_files[0].name):
        raise ValueError(
            f"The masks in {masks_dir} were not renamed "
            "you must run the command histopreprocessing "
            "rename-masks on that folder."
        )

    logger.info("Searching for matching WSI path")
    wsi_paths_mapping = map_masks_to_wsi(mask_files, raw_wsi_dir, filename_to_wsi_id)
    logger.info("Searching for matching WSI path - DONE")

    # Create a list of arguments for parallel processing
    wsi_args = [
        (
            wsi_paths_mapping[mask_path],
            mask_path,  # Mask path
            output_dir,
            magnification,
            tile_size,
            num_workers_tiles,
            save_masks,  # save_mask
            save_tile_overlay,
        )
        for mask_path in mask_files
    ]

    with Pool(processes=num_workers_wsi) as pool:
        results = list(
            tqdm(
                pool.starmap(get_tile_coordinates_per_wsi, wsi_args),
                total=len(wsi_args),
                desc="Processing WSIs",
            )
        )

    for wsi_path, result in zip(wsi_paths_mapping.values(), results):
        if not result:
            logger.warning(f"Processing failed for WSI {wsi_path.name}")


def get_tile_coordinates_per_wsi(
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
        wsi = OpenSlide(wsi_path)
        mask_array = np.array(Image.open(mask_path)) != 0
        coords = extract_patch_coords_from_mask(
            wsi,
            mask_array,
            target_mpp=MAGNIFICTION_TO_MPP[magnification],
            patch_size=tile_size,
            step_size=tile_size,
            min_tissue_ratio=0.5,
            tissue_area_threshold=100,
            hole_area_threshold=16,
            max_holes=8,
            ref_patch_size=512,
            mpp_selection_mode="closest_mpp",  # or "highest_mpp"
            segmentation_downsample=64,
        )

    except Exception as e:
        logger.error(f"Error processing WSI {wsi_path.name}: {e}")
        return False

    if save_tile_overlay:
        tile_processor.save_overlay()
    if save_metadata:
        tile_processor.save_metadata()
    return True
