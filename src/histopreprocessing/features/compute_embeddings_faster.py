# TODO: Move functions to utils.py

import json
import os
from pathlib import Path
import logging
from time import perf_counter


import click
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histopreprocessing.features.foundation_models import load_model
from histopreprocessing.features.torch_datasets import TileDataset


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


def compute_and_store_embeddings(
    tile_paths,
    model,
    model_name=None,
    preprocess=None,
    batch_size=128,
    num_workers=None,
    output_dir=None,
    autocast_dtype=None,
    save_every_n_batches=1,
):
    """Compute embeddings dynamically based on model type and save temp checkpoints."""

    if autocast_dtype is not None:
        logger.info(f"Using autocast with dtype {autocast_dtype}")

    dataset = TileDataset(tile_paths, preprocess=preprocess)
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

    all_embeddings = []
    all_tilepaths = []
    for batch_idx, (batch_images, batch_tilepaths) in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            batch_images = batch_images.to(device, non_blocking=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    batch_embeddings = (
                        model(batch_images).cpu().numpy().astype(np.float32)
                    )
            else:
                batch_embeddings = model(batch_images).cpu().numpy().astype(np.float32)

        all_embeddings.append(batch_embeddings)
        all_tilepaths.extend(batch_tilepaths)
        if save_every_n_batches > 0 and batch_idx % save_every_n_batches == 0:
            time_start = perf_counter()
            save_embeddings(
                output_dir,
                batch_embeddings,
                batch_tilepaths,
                model_name=model_name,
            )
            logger.info(
                f"Saved embeddings for batch {batch_idx} in {perf_counter() - time_start:.2f} seconds"
            )
            all_embeddings = []
            all_tilepaths = []


def get_coordinates_from_tile_path(tile_path):
    positions = tile_path.stem.split("__")[1]

    # Extract x and y coordinates
    pos_parts = positions.split("_")
    x_coord = int(pos_parts[0].replace("x", ""))
    y_coord = int(pos_parts[1].replace("y", ""))

    return x_coord, y_coord


def save_embeddings(
    output_dir,
    batch_embeddings,
    batch_tilepaths,
    model_name,
):
    """Save embeddings to HDF5 file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_tilepaths = [Path(p) for p in batch_tilepaths]
    embedding_dim = batch_embeddings.shape[1]

    big_dict = {}
    for idx, tile_path in enumerate(batch_tilepaths):
        wsi_id = str(tile_path.parents[1].stem)
        if wsi_id not in big_dict:
            big_dict[wsi_id] = {"embeddings": [], "coordinates": [], "tilepaths": []}
        big_dict[wsi_id]["embeddings"].append(batch_embeddings[idx, :])
        big_dict[wsi_id]["coordinates"].append(
            get_coordinates_from_tile_path(tile_path)
        )
        big_dict[wsi_id]["tilepaths"].append(str(tile_path))

    for wsi_id, asset_dict in big_dict.items():
        hdf5_path = output_dir / f"{wsi_id}.h5"

        if not hdf5_path.exists():
            tile_dir = Path(asset_dict["tilepaths"][0]).parents[1]
            with open(tile_dir / f"{wsi_id}__metadata.json", "r") as f:
                tiles_metadata = json.load(f)
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
                "total_tiles": {
                    "description": "Total number of tiles",
                    "value": len(list((tile_dir / "tiles").glob("*.png"))),
                },
                "current_number_tiles": {
                    "description": "Current number of saved tiles, used for resuming",
                    "value": 0,
                },
            }
            attr_dict.update(tiles_metadata)
        else:
            mode = "a"
            attr_dict = None

        save_hdf5(
            hdf5_path,
            {
                "embeddings": np.array(asset_dict["embeddings"]),
                "coordinates": np.array(asset_dict["coordinates"]),
            },
            global_attr_dict=attr_dict,
            mode=mode,
        )
        update_hdf5_completion(hdf5_path)


def update_hdf5_completion(hdf5_path):
    with h5py.File(hdf5_path, "a") as f:
        total = f.attrs.get("total_tiles", None)
        current = f["embeddings"].shape[0]
        f.attrs["current_number_tiles"] = current
        if total is not None and current == total:
            f.attrs["is_complete"] = True
        else:
            f.attrs["is_complete"] = False


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
    "--tile-paths-json",
    help="Path to JSON file containing tile paths",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
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
@click.option("--save-every-n-batches", default=1, help="Save every n batches")
def main(
    model_name,
    tile_paths_json,
    output_dir,
    gpu_id,
    batch_size,
    num_workers,
    use_autocast,
    save_every_n_batches,
):
    # Load Model
    device = get_device(gpu_id)
    model, preprocess, _, autocast_dtype = load_model(model_name, device)

    if not use_autocast:
        autocast_dtype = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(tile_paths_json, "r") as f:
        tile_paths = json.load(f)

    compute_and_store_embeddings(
        tile_paths,
        model,
        model_name=model_name,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        autocast_dtype=autocast_dtype,
        save_every_n_batches=save_every_n_batches,
    )


if __name__ == "__main__":
    main()
