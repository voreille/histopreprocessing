import logging
from multiprocessing.pool import ThreadPool, Pool

from tqdm import tqdm

from histolung.data.wsi_tiler import WSITilerWithSuperPixelMaskWithOverlap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_wsi(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=10,
    tile_size=224,
    threshold=0.8,
    num_workers_tiles=12,
    save_tile_overlay=False,
    save_metadata=True,
    average_superpixel_area=1000,
    average_n_tiles=10,
):
    logger.info(f"Starting tiling for WSI {wsi_path.name}")
    try:
        tile_processor = WSITilerWithSuperPixelMaskWithOverlap(
            wsi_path,
            mask_path,
            output_dir,
            magnification=magnification,
            tile_size=tile_size,
            threshold=threshold,
            save_tile_overlay=save_tile_overlay,
            average_superpixel_area=average_superpixel_area,
            average_n_tiles=average_n_tiles,
        )
        superpixel_labels = tile_processor.labels

        if num_workers_tiles > 1:

            with ThreadPool(processes=num_workers_tiles) as pool:
                # results = list(
                #     tqdm(pool.imap_unordered(tile_processor,
                #                              superpixel_labels),
                #          total=len(superpixel_labels),
                #          desc="Processing superpixels"))

                results = list(
                    pool.imap_unordered(
                        tile_processor,
                        superpixel_labels,
                    ))

        else:
            results = [
                # tile_processor(label) for label in tqdm(superpixel_labels)
                tile_processor(label) for label in superpixel_labels
            ]

        if any(isinstance(res, Exception) for res in results):
            raise RuntimeError(
                f"Error encountered in tile processing for {wsi_path.name}")

        logger.info(f"Tiling completed for WSI {wsi_path.name}")

        if save_tile_overlay:
            tile_processor.save_overlay()
        if save_metadata:
            tile_processor.save_metadata()

    except Exception as e:
        logger.error(f"Error processing WSI {wsi_path.name}: {e}")
        return False

    return True


def process_wsi_wrapper(args):
    """Unpacks tuple args before calling process_wsi"""
    return process_wsi(*args)


def get_wsi_path_from_mask_path(mask_path, raw_data_dir):
    wsi_id = mask_path.stem.replace("_segments", "")
    match = list(raw_data_dir.rglob(f"{wsi_id}*.svs"))
    if len(match) > 1:
        raise ValueError(f"multiple matching *.svs for the wsi_id: {wsi_id}")

    if len(match) == 0:
        raise ValueError(f"No matching *.svs for wsi_id: {wsi_id}")

    return match[0]


def tile_wsi_superpixel_task_random_overlap(
    raw_data_dir,
    masks_dir,
    output_dir,
    tile_size=224,
    threshold=0.8,
    num_workers_wsi=4,
    num_workers_tiles=12,
    save_tile_overlay=False,
    magnification=10,
    save_metadata=True,
    average_superpixel_area=1000,
    average_n_tiles=10,
):

    mask_files = [f for f in masks_dir.rglob("*.tiff")]

    # Create a list of arguments for parallel processing
    wsi_args = [(
        get_wsi_path_from_mask_path(mask_path, raw_data_dir),
        mask_path,
        output_dir,
        magnification,
        tile_size,
        threshold,
        num_workers_tiles,
        save_tile_overlay,
        save_metadata,
        average_superpixel_area,
        average_n_tiles,
    ) for mask_path in mask_files]

    if num_workers_wsi > 1:
        with Pool(processes=num_workers_wsi) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_wsi_wrapper, wsi_args),
                    total=len(wsi_args),
                    desc="Processing WSIs",
                ))

    else:
        results = [
            process_wsi(*args)
            for args in tqdm(wsi_args, desc="Processing WSIs")
        ]

    for wsi_arg, result in zip(wsi_args, results):
        wsi_path = wsi_arg[0]
        if not result:
            logger.warning(f"Processing failed for WSI {wsi_path.name}")
