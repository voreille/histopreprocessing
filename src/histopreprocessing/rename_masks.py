import logging
from pathlib import Path, PurePath
import shutil

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def rename_masks_with_copy(masks_dir, dataset_name):
    """
    Copies and renames mask directories and files safely.

    Args:
        masks_dir (Path): Base directory containing masks for the dataset.
        dataset_name (str): Name of the dataset (e.g., "tcga_luad").
    """
    # Get the renaming function based on the dataset
    renaming_function = RENAMING_RULES.get(dataset_name)

    # If no renaming function is defined, skip this dataset
    if not renaming_function:
        logger.warning(
            f"No renaming rule for dataset {dataset_name}. Skipping.")
        raise ValueError(
            f"No renaming rule for dataset {dataset_name}. Skipping.")

    # Define paths
    masks_out_dir = masks_dir / "output"
    masks_temp_dir = masks_dir / "masks_temp"
    masks_final_dir = masks_dir / "masks"

    # Create a temporary directory for renamed files
    masks_temp_dir.mkdir(parents=False, exist_ok=True)

    # Copy metadata files (error.log, results.tsv)
    for filename in ["error.log", "results.tsv"]:
        file_path = masks_out_dir / filename
        if file_path.exists():
            shutil.copy(file_path, masks_dir / filename)
            logger.info(f"Copied {filename} to {masks_dir / filename}")

    # Rename and copy WSI folders into the temporary directory
    wsi_folders = [f for f in masks_out_dir.iterdir() if f.is_dir()]

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
    original_count = sum(1 for _ in masks_out_dir.rglob("*"))
    copied_count = sum(1 for _ in masks_temp_dir.rglob("*"))

    # Adjust the copied count to include the moved files (error.log, results.tsv)
    adjusted_copied_count = copied_count + 2

    if original_count == adjusted_copied_count:
        logger.info(f"Validation successful: All files are accounted "
                    f"for. Original count: {original_count}, "
                    f"Copied count (adjusted): {adjusted_copied_count}.")
    else:
        logger.error(f"Validation failed: File counts do not match. "
                     f"Original count: {original_count}, "
                     f"Copied count (adjusted): {adjusted_copied_count}.")
        return

    # Move temporary renamed files to final location
    if masks_final_dir.exists():
        shutil.rmtree(masks_final_dir)  # Remove existing directory
        logger.warning(f"Existing directory {masks_final_dir} removed.")
    masks_temp_dir.rename(masks_final_dir)
    logger.info(f"Renamed files finalized in {masks_final_dir}.")

    # Cleanup: Remove original output directory
    shutil.rmtree(masks_out_dir)
    logger.info(f"Deleted original directory {masks_out_dir}.")


OPENS_SLIDE_EXTENSIONS = {
    ".svs", ".tiff", ".vms", ".vmu", ".ndpi", ".scn", ".mrxs", ".tif"
}


def write_wsi_paths_to_csv(
    raw_data_dir: str,
    masks_dir: str,
    output_csv: str,
    dataset: str,
):
    raw_data_dir = Path(raw_data_dir)
    masks_dir = Path(masks_dir)

    # Find all mask files
    mask_files = [f for f in masks_dir.rglob("*mask_use.png")]
    wsi_ids = {f.name.split(".")[0]
               for f in mask_files}  # Use a set for efficient lookup

    # Define the renaming function for the dataset
    rename_func = RENAMING_RULES.get(dataset)
    if not rename_func:
        raise ValueError(f"No renaming function found for dataset: {dataset}")

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
    print(f"CSV file saved to {output_csv}")


def write_wsi_paths_to_csv_old(results_tsv: str, output_csv: str, dataset: str,
                               config: dict):
    """
    Extracts paths to WSI (Whole Slide Image) files from a results.tsv file, applies renaming rules, 
    and saves the paths with their corresponding WSI IDs to a CSV file.

    Args:
        results_tsv (str): Path to the `results.tsv` file containing metadata, including command-line arguments.
        output_csv (str): Path to the CSV file where the WSI paths and IDs will be saved.
        dataset (str): Name of the dataset, used to apply dataset-specific renaming rules.
        config (dict): Configuration dictionary containing the `data_dir` key, which specifies the base directory of the raw data.

    Raises:
        ValueError: If the `results.tsv` file does not contain a `#command_line_args` line.
    """
    results_tsv_path = Path(results_tsv)

    # Read the results.tsv file
    with results_tsv_path.open('r') as file:
        lines = file.readlines()

    # Extract the #command_line_args line
    command_line_args_line = next(
        (line for line in lines if line.startswith("#command_line_args")),
        None)
    if not command_line_args_line:
        raise ValueError("No #command_line_args line found in results.tsv.")

    # Resolve the raw data directory
    raw_data_dir = Path(config["data_dir"]).resolve()

    # Extract and process command-line arguments
    command_line_args = command_line_args_line.split("\t")[1].split(" ")
    file_paths = [
        raw_data_dir / Path(*PurePath(arg).parts[2:])
        for arg in command_line_args
    ]

    # Filter valid WSI files
    wsi_files = [
        path for path in file_paths if path.is_file() and path.suffix != ".ini"
    ]

    # Apply renaming rules to generate WSI IDs
    rename_func = RENAMING_RULES.get(dataset)
    path_to_wsi_id = {rename_func(str(path.name)): path for path in wsi_files}

    # Create and save the DataFrame
    df = pd.DataFrame(list(path_to_wsi_id.items()), columns=["WSI_ID", "Path"])
    df.to_csv(output_csv, index=False)