import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def find_tiles_in_folder(folder):
    """Finds all tile files inside a given folder."""
    return list(folder.rglob(
        "*.png"))  # Recursively find all .png tiles in this folder


def process_tile(tile_path):
    """Extract superpixel_id and return it with the path"""
    tile_name = tile_path.stem  # Example: [wsi_id]__l[superpixel_label]__x[x]_y[y]
    superpixel_id = "__".join(
        tile_name.split("__")[:2])  # Extract [wsi_id]__l[superpixel_label]
    return superpixel_id, str(tile_path)


def create_superpixel_tile_mapping_task(tile_dir,
                                        output_json,
                                        wsi_ids_to_discard=None,
                                        num_workers=8):
    """
    Generates a JSON file mapping each superpixel to its tile paths in parallel.

    Args:
        tile_dir (str or Path): Directory containing the tiles.
        output_json (str): Path to save the JSON mapping.
        wsi_ids_to_discard (list): List of WSI IDs to exclude.
        num_workers (int): Number of parallel workers for scanning.
    """
    tile_dir = Path(tile_dir)

    # Step 1: Find all tile-containing folders, filtering out unwanted WSIs
    print("Scanning for tile folders...")
    tile_folders = []
    if wsi_ids_to_discard is None:
        wsi_ids_to_discard = set()

    for wsi_folder in tile_dir.iterdir():
        if not wsi_folder.is_dir():
            continue  # Skip if it's a file

        wsi_id = wsi_folder.name  # Extract WSI ID
        if wsi_id in wsi_ids_to_discard:
            continue  # Skip unwanted WSIs early

        tiles_path = wsi_folder / "tiles"
        if tiles_path.is_dir():
            tile_folders.append(tiles_path)

    print(f"Found {len(tile_folders)} tile folders after filtering.")

    # Step 2: Find all tiles in parallel
    print(f"Scanning tiles from {len(tile_folders)} folders in parallel...")
    tile_paths = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for paths in tqdm(executor.map(find_tiles_in_folder, tile_folders),
                          total=len(tile_folders),
                          desc="Finding tiles"):
            tile_paths.extend(paths)  # Flatten results

    print(f"Total tiles found: {len(tile_paths)}")

    superpixel_map = {}

    # Step 3: Process tiles in parallel to extract superpixel IDs
    print("Processing tiles in parallel...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(process_tile, tile_paths),
                 total=len(tile_paths),
                 desc="Processing tiles"))

    # Step 4: Aggregate results
    for superpixel_id, tile_path in results:
        if superpixel_id not in superpixel_map:
            superpixel_map[superpixel_id] = []
        superpixel_map[superpixel_id].append(tile_path)

    # Step 5: Convert to JSON format
    superpixel_list = [{
        "superpixel_id": k,
        "tile_paths": v
    } for k, v in superpixel_map.items()]

    # Step 6: Save JSON
    with open(output_json, "w") as f:
        json.dump(superpixel_list, f, indent=4)

    print(f"Superpixel-tile mapping saved to {output_json}")
