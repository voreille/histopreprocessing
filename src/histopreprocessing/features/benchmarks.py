import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histopreprocessing.features.foundation_models import load_model
from histopreprocessing.features.torch_datasets import (
    TileDataset,
    TileDatasetFromRawWSI,
)


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


def compute_embeddings_from_raw_wsi_dir(
    raw_wsi_path,
    coordinates,
    model,
    preprocess=None,
    batch_size=128,
    num_workers=None,
    autocast_dtype=torch.float16,
    tile_size=224,
    tile_level=0,
    device=None,
):
    """Compute embeddings dynamically based on model type and save temp checkpoints."""

    dataset = TileDatasetFromRawWSI(
        raw_wsi_path,
        coordinates,
        tile_size_at_level0=tile_size,
        tile_level=tile_level,
        preprocess=preprocess,
    )
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

    if device is None:
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


def compute_embeddings(
    tile_paths,
    model,
    preprocess=None,
    batch_size=128,
    num_workers=None,
    autocast_dtype=torch.float16,
    device=None,
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

    if device is None:
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


def get_tile_coordinates(tile_dir):
    """Get tile coordinates from a DataFrame."""
    wsi_id = tile_dir.name

    df = pd.read_csv(tile_dir / f"{wsi_id}__tiling_results.csv")
    coordinates = []
    # Fix the syntax error in the for loop
    for _, row in df[df["keep"] == 1].iterrows():
        x, y = row["x_level0"], row["y_level0"]
        coordinates.append((x, y))
    return coordinates


def main():
    """Precompute and store WSI embeddings in a single HDF5 file."""
    # Load metadata
    model_name = "dummy"
    gpu_id = 0
    batch_size = 256
    num_workers = 28

    # Load Model
    device = get_device(gpu_id)

    model, preprocess, _, autocast_dtype = load_model(model_name, device)

    tiles_wsi_dir = Path(
        "/mnt/nas7/data/Personal/Valentin/histopath/tiles_20x/cptac_luad/C3N-00580-24"
        # "/home/valentin/workspaces/histopreprocessing/data/tile_dir/C3N-00580-24"
    )
    raw_wsi_path = Path("/mnt/nas6/data/CPTAC/CPTAC-LUAD_v12/LUAD/C3N-00580-24.svs")

    print("Running benchmark using parameters:")
    print(f"Model name: {model_name}")
    print(f"GPU ID: {gpu_id}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Raw WSI path: {raw_wsi_path}")
    print(f"Tile WSI directory: {tiles_wsi_dir}")

    print(f"Looking for tiles in {tiles_wsi_dir}")
    tile_paths = list((tiles_wsi_dir / "tiles").glob("*.png"))
    total_number_tiles = len(tile_paths)

    print(f"Total number of tiles: {total_number_tiles}")

    print(f"Getting tile coordinates from {tiles_wsi_dir}")
    coordinates = get_tile_coordinates(tiles_wsi_dir)
    print(f"Total number of coordinates: {len(coordinates)}")

    # Compute embeddings

    t1_start = perf_counter()
    embeddings = compute_embeddings_from_raw_wsi_dir(
        raw_wsi_path,
        coordinates,
        model,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        autocast_dtype=autocast_dtype,
        tile_size=224,  # Hardcoded
        tile_level=0,
        device=device,
    )
    t1_stop = perf_counter()
    print(
        "Elapsed time computing embeddings from tiles from OpenSlide raw WSI",
        t1_stop - t1_start,
    )
    print(f"embeddings shape: {embeddings.shape}")

    t1_start = perf_counter()
    embeddings = compute_embeddings(
        tile_paths,
        model,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        autocast_dtype=autocast_dtype,
        device=device,
    )
    t1_stop = perf_counter()
    print(
        "Elapsed time computing embeddings from tiles on the disk", t1_stop - t1_start
    )
    print(f"embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
