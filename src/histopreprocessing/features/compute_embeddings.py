# TODO: Move functions to utils.py

import json
import os
from pathlib import Path
import logging

import click
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histopreprocessing.features.foundation_models import load_model
from histopreprocessing.features.torch_datasets import TileAndCoordDataset


logger = logging.getLogger(__name__)


def get_device(gpu_id=None):
    """Select the appropriate device for computation."""
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")  # Default to first GPU
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device


def compute_and_store_embeddings_per_wsi(
    tile_dir,
    model,
    model_name=None,
    embedding_dim=None,
    preprocess=None,
    batch_size=128,
    num_workers=None,
    output_filepath=None,
    autocast_dtype=None,
):
    """Compute embeddings dynamically based on model type and save temp checkpoints."""

    if autocast_dtype is not None:
        logger.info(f"Using autocast with dtype {autocast_dtype}")

    tile_paths = list((tile_dir).glob("tiles/*.png"))
    wsi_id = tile_dir.stem
    with open(tile_dir / f"{wsi_id}__metadata.json", "r") as f:
        tiles_metadata = json.load(f)

    dataset = TileAndCoordDataset(tile_paths, preprocess=preprocess)
    num_workers = min(4, os.cpu_count() // 2) if num_workers is None else num_workers
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep workers alive
        pin_memory=True,
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"starting processing tiles with num_workers={num_workers}")

    mode = "w"
    attr_dict = {
        "embeddings": {
            "description": "Embeddings for each tile",
            "shape": (None, embedding_dim),
        },
        "coordinates": {
            "description": "Coordinates of each tile",
            "shape": (None, 2),
        },
        "model_name": {
            "description": "Model name used for embedding computation",
            "value": model_name,
        },
    }
    attr_dict.update(tiles_metadata)

    for batch_idx, (batch_images, batch_coordinates) in enumerate(
        tqdm(dataloader, desc=f"Processing Tiles of WSI {wsi_id}", unit="batch")
    ):
        with torch.inference_mode():
            batch_images = batch_images.to(device, non_blocking=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    batch_embeddings = (
                        model(batch_images).cpu().numpy().astype(np.float32)
                    )
            else:
                batch_embeddings = model(batch_images).cpu().numpy().astype(np.float32)
            batch_coordinates = batch_coordinates.cpu().numpy().astype(np.int32)
            asset_dict = {
                "embeddings": batch_embeddings,
                "coordinates": batch_coordinates,
            }
            save_hdf5(
                output_filepath,
                asset_dict,
                global_attr_dict=attr_dict,
                mode=mode,
            )
            mode = "a"
            attr_dict = None

    return output_filepath


def save_hdf5(output_path, asset_dict, global_attr_dict=None, mode="a"):
    """
    Save data to an HDF5 file.

    Parameters:
        output_path (str): Path to the HDF5 file.
        asset_dict (dict): Dictionary of datasets to save.
        global_attr_dict (dict): Dictionary of global attributes for the WSI.
        mode (str): File mode ('w' for write, 'a' for append).
    """
    file = h5py.File(output_path, mode)

    # Add global attributes to the file
    if global_attr_dict is not None:
        for attr_key, attr_val in global_attr_dict.items():
            # Serialize unsupported types to JSON strings
            if isinstance(attr_val, (dict, list)):
                attr_val = json.dumps(attr_val)
            file.attrs[attr_key] = attr_val

    # Add datasets and their attributes
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val

    file.close()
    return output_path


@click.command()
@click.option(
    "--model-name",
    type=str,
    default="UNI2",
    help="Model name to use for embedding computation",
)
@click.option(
    "--tile-dirs-path",
    help="Path to the directory containing tile directories",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--output-dir",
    default="data/processed/embeddings/UNI2/",
    help="Output directory for storing embeddings",
)
@click.option("--gpu-id", default=0, help="GPU ID to use for inference")
@click.option("--batch-size", default=512, help="Batch size for inference")
@click.option(
    "--num-workers",
    default=8,
    type=click.INT,
    help="Number of workers for DataLoader",
)
@click.option(
    "--use-autocast",
    is_flag=True,
    default=False,
    help="Enable mixed precision inference using autocast",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Enable mixed precision inference using autocast",
)
def main(
    model_name,
    tile_dirs_path,
    output_dir,
    gpu_id,
    batch_size,
    num_workers,
    use_autocast,
    force,
):
    # Load Model
    device = get_device(gpu_id)
    model, preprocess, embedding_dim, autocast_dtype = load_model(model_name, device)

    if not use_autocast:
        autocast_dtype = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_dirs_path = Path(tile_dirs_path)

    for tile_dir in tile_dirs_path.iterdir():
        wsi_id = tile_dir.stem
        output_filepath = Path(output_dir) / f"{wsi_id}.h5"
        if output_filepath.exists() and not force:
            logger.info(
                f"File {output_filepath} already exists. Skipping, use --force to overwrite."
            )
            continue

        logger.info(f"Processing {tile_dir}")
        if not tile_dir.is_dir():
            logger.info(f"{tile_dir} is not a directory. Skipping.")
            continue

        try:
            compute_and_store_embeddings_per_wsi(
            tile_dir,
            model,
            model_name=model_name,
            embedding_dim=embedding_dim,
            preprocess=preprocess,
            batch_size=batch_size,
            num_workers=num_workers,
            output_filepath=output_filepath,
            autocast_dtype=autocast_dtype,
            )
        except Exception as e:
            logger.error(f"Error processing {tile_dir}: {e}")
            continue


if __name__ == "__main__":
    main()
