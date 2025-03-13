import logging
from pathlib import Path, PurePath
import shutil
import tempfile

import pandas as pd

from histopreprocessing.wsi_id_mapping import WSI_ID_MAPPING_DICT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rename_masks_task(input_dir, wsi_id_mapping_style):
    """Rename mask folders according to dataset conventions"""

    input_dir = Path(input_dir)
    filename_to_wsi_id_mapping = WSI_ID_MAPPING_DICT[wsi_id_mapping_style]

    logger.info(f"Renaming HistQC output in: {input_dir}")
    rename_masks_with_copy(input_dir, filename_to_wsi_id_mapping)


# Define dataset-specific renaming functions
def rename_tcga(folder_name):
    """Extracts the TCGA ID by splitting at the first period."""
    return folder_name.split(".")[0]


def rename_cptac(folder_name):
    """Extracts the CPTAC ID by splitting at the first period."""
    return folder_name.split(".")[0]  # Example for CPTAC


# Configuration dictionary for datasets and their renaming functions
RENAMING_RULES = {
    "tcga_luad": rename_tcga,
    "tcga_lusc": rename_tcga,
    "cptac_lusc": rename_cptac,
    "cptac_luad": rename_cptac,
    # Add other datasets and their corresponding functions here
}

# Base directory and other configuration
base_dir = Path("data/interim/masks/")
datasets = ["tcga_luad", "tcga_lusc", "cptac_lscc"]  # List your datasets here


def reformat_results_tsv(results_tsv_path, dataset, dataset_config):
    with open(results_tsv_path, 'r') as file:
        start_line = next(i for i, line in enumerate(file)
                          if line.startswith('#dataset:filename'))

    df = pd.read_csv(results_tsv_path, sep="\t", skiprows=start_line)
    df = df.rename(columns={"#dataset:filename": "filename"})


def rename_files_in_folder(folder, renaming_function):
    """
    Rename all PNG files within a folder using the renaming scheme.

    Args:
        folder (Path): The folder containing the files to rename.
        renaming_function (function): The renaming function for the dataset.
    """
    for file in folder.glob("*.png"):
        original_name = file.name
        # Extract the new base name from the folder
        base_name = renaming_function(folder.name)
        # Reconstruct the new filename
        new_name = base_name + "." + ".".join(original_name.split(".")[-2:])
        new_path = folder / new_name

        # Rename the file
        file.rename(new_path)
        logger.info(f"Renamed file: {file} -> {new_path}")


def rename_masks_with_copy(masks_dir, renaming_function):
    """
    Copies and renames mask directories and files safely.

    Args:
        masks_dir (Path): Base directory containing masks for the dataset.
        dataset_name (str): Name of the dataset (e.g., "tcga_luad").
    """

    # Define paths
    masks_temp_dir = Path(tempfile.mkdtemp())

    # Create a temporary directory for renamed files
    masks_temp_dir.mkdir(parents=False, exist_ok=True)

    # Copy metadata files (error.log, results.tsv)
    for filename in ["error.log", "results.tsv", "raw_wsi_path.csv"]:
        file_path = masks_dir / filename
        if file_path.exists():
            shutil.copy(file_path, masks_temp_dir / filename)
            logger.info(f"Copied {filename} to {masks_temp_dir / filename}")

    # Rename and copy WSI folders into the temporary directory
    wsi_folders = [f for f in masks_dir.iterdir() if f.is_dir()]

    for folder in wsi_folders:
        new_folder_name = renaming_function(folder.name)
        new_folder_path = masks_temp_dir / new_folder_name

        # Copy folder contents to the new folder
        if not new_folder_path.exists():
            shutil.copytree(folder, new_folder_path)
            logger.info(f"Copied and renamed {folder} to {new_folder_path}")
            rename_files_in_folder(new_folder_path, renaming_function)
        else:
            logger.warning(
                f"Skipping {folder} as {new_folder_path} already exists.")

    # Validation step
    original_count = sum(1 for _ in masks_dir.rglob("*"))
    copied_count = sum(1 for _ in masks_temp_dir.rglob("*"))

    if original_count == copied_count:
        logger.info(f"Validation successful: All files are accounted "
                    f"for. Original count: {original_count}, "
                    f"Copied count: {copied_count}.")
    else:
        logger.error(f"Validation failed: File counts do not match. "
                     f"Original count: {original_count}, "
                     f"Copied count: {copied_count}.")
        return

    shutil.rmtree(masks_dir)
    masks_dir.mkdir()
    for item in masks_temp_dir.iterdir():
        shutil.move(str(item), str(masks_dir / item.name))

    shutil.rmtree(masks_temp_dir)

    logger.info(f"Moved renamed content to {masks_dir}")


OPENS_SLIDE_EXTENSIONS = {
    ".svs", ".tiff", ".vms", ".vmu", ".ndpi", ".scn", ".mrxs", ".tif"
}


def write_wsi_paths_to_csv(
    raw_data_dir: str,
    masks_dir: str,
    output_csv: str,
    wsi_id_mapping_style: str,
):
    logging.info(f"Writing raw WSI paths to {output_csv}")
    raw_data_dir = Path(raw_data_dir)
    masks_dir = Path(masks_dir)

    # Find all mask files
    mask_files = [f for f in masks_dir.rglob("*mask_use.png")]
    wsi_ids = {f.name.split(".")[0]
               for f in mask_files}  # Use a set for efficient lookup

    # Define the renaming function for the dataset
    rename_func = WSI_ID_MAPPING_DICT.get(wsi_id_mapping_style)

    # Map WSI IDs to their paths
    path_to_wsi_id = {
        rename_func(f.name): f
        for f in raw_data_dir.rglob("*") if f.is_file()
        and f.suffix.lower() in OPENS_SLIDE_EXTENSIONS  # Filter by extension
        and rename_func(f.name) in wsi_ids
    }

    # Check for any missing WSI IDs
    missing_wsi_ids = wsi_ids - set(path_to_wsi_id.keys())
    if missing_wsi_ids:
        print(
            f"Warning: The following WSI IDs are missing in raw data: {missing_wsi_ids}"
        )

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(list(path_to_wsi_id.items()), columns=["WSI_ID", "Path"])
    df.to_csv(output_csv, index=False)
    logging.info(f"Writing raw WSI paths to {output_csv} - DONE")
