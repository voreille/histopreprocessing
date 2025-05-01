import os
from pathlib import Path
import json

import click
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm

from histopreprocessing.features.foundation_models import load_model
from histopreprocessing.features.torch_datasets import TileDataset


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


def compute_embeddings(
    tile_paths,
    model,
    preprocess=None,
    batch_size=128,
    num_workers=None,
    autocast_dtype=torch.float16,
):
    """Compute embeddings dynamically based on model type and save temp checkpoints."""

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

    embeddings = []

    for batch_idx, (batch_images, _) in enumerate(
        tqdm(dataloader, desc="Processing Tiles", unit="batch")
    ):
        batch_images = batch_images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            with torch.inference_mode():
                batch_embeddings = model(batch_images).detach().cpu().numpy()

        embeddings.extend(list(batch_embeddings))

    embeddings = np.vstack(embeddings)
    return embeddings


def compute_and_store_embeddings(
    model_name,
    tile_paths_json,
    output_filepath,
    gpu_id=0,
    batch_size=32,
    num_workers=0,
):
    """Precompute and store WSI embeddings in a single HDF5 file."""
    # Load metadata

    with open(tile_paths_json, "r") as f:
        tile_paths = json.load(f)

    total_number_tiles = len(tile_paths)
    print(f"Total number of tiles: {total_number_tiles}")

    # Load Model
    device = get_device(gpu_id)
    model, preprocess, embedding_dim, autocast_dtype = load_model(model_name, device)

    # Compute embeddings
    embeddings = compute_embeddings(
        tile_paths,
        model,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        autocast_dtype=autocast_dtype,
    )
    np.savez_compressed(
        output_filepath,
        embeddings=embeddings,
        tile_paths=tile_paths,
    )


@click.command()
@click.option(
    "--model-name",
    type=str,
    default="UNI2",
    help="Model name to use for embedding computation",
)
@click.option(
    "--raw-wsi-dir",
    help="raw WSI directory",
)
@click.option(
    "--output-filepath",
    default="data/processed/embeddings/embedding.npz",
    help="Output file path for storing embeddings",
)
@click.option("--gpu-id", default=0, help="GPU ID to use for inference")
@click.option("--batch-size", default=256, help="Batch size for inference")
@click.option(
    "--num-workers",
    default=None,
    type=click.INT,
    help="Number of workers for DataLoader",
)
def main(
    model_name,
    raw_wsi_dir,
    wsi_tile_dirs_json,
    output_filepath,
    gpu_id,
    batch_size,
    num_workers,
):
    compute_and_store_embeddings(
        model_name,
        tile_paths_json,
        output_filepath,
        gpu_id=gpu_id,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
