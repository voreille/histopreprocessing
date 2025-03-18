from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_tiles_metadata_task(input_dir, output_json):
    """Store metatdata about tiles path, label for the dataloader"""
    input_dir = Path(input_dir)
    output_json = Path(output_json)

    wsi_data = []
    for wsi_id_dir in input_dir.iterdir():
        if wsi_id_dir.is_dir():
            patch_dir = wsi_id_dir / "tiles"
            if patch_dir.exists():
                patch_files = [str(p) for p in patch_dir.glob("*.png")]
                if len(patch_files) == 0:
                    logger.warning(f"WSI with "
                                   f"ID {wsi_id_dir.name} has not tiles "
                                   f"thus it has been discarded.")
                    continue
                wsi_data.append({
                    "wsi_id": wsi_id_dir.name,
                    "patch_dir": str(patch_dir),
                    "patch_files": patch_files,
                })

    with open(output_json, 'w') as f:
        json.dump(wsi_data, f, indent=4)

    logger.info(f"Writing tiles metadata to {output_json}")
