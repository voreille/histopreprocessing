import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Accepted OpenSlide file extensions (case-insensitive).
ACCEPTED_OPENSLIDE_EXTENSIONS = {
    ".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".vms", ".vmu",
    ".svslide", ".isyntax"
}


def configure_logging(log_file=None):
    """Configure logging to print to stdout and optionally save to a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler (logs to stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optional: Log to file if specified
    if log_file:
        log_file_path = Path(log_file).resolve()
        log_file_path.parent.mkdir(parents=True,
                                   exist_ok=True)  # Ensure directory exists
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info("Logging initialized.")


def build_wsi_index(raw_data_dir: Path,
                    filename_to_wsi_id: callable,
                    wsi_extension: str = "all"):
    """Build an index mapping WSI IDs to file paths using filtered extensions."""
    if wsi_extension == "all":
        accepted_extensions = ACCEPTED_OPENSLIDE_EXTENSIONS
    else:
        accepted_extensions = {wsi_extension}
    index = {}
    for path in raw_data_dir.rglob("*"):
        if path.suffix.lower() in accepted_extensions:
            # Assume the file stem (without extension) is the WSI id.
            wsi_id = filename_to_wsi_id(path.stem)
            if wsi_id in index:
                raise ValueError(f"Duplicate WSI id found for {wsi_id}")
            index[wsi_id] = path
    return index


def map_masks_to_wsi(masks_list,
                     raw_data_dir: Path,
                     filename_to_wsi_id: callable,
                     wsi_extension: str = "all"):
    """Map each mask file to its corresponding WSI file based on the WSI id."""
    index = build_wsi_index(raw_data_dir,
                            filename_to_wsi_id,
                            wsi_extension=wsi_extension)
    mapping = {}
    for mask_path in masks_list:
        # Remove the known suffix to extract the wsi id.
        wsi_id = filename_to_wsi_id(mask_path.stem)
        if wsi_id not in index:
            raise ValueError(f"No matching file for wsi_id: {wsi_id}")
        mapping[mask_path] = index[wsi_id]
    return mapping
