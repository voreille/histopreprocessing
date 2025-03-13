import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_tiles_metadata(config, tiles_basedir):
    """Store metatdata about tiles path, label for the dataloader"""
    tiles_basedir = Path(config["tiles_basedir"])
    preprocessed_datasets = [
        f.name for f in tiles_basedir.iterdir() if f.is_dir()
    ]
    wsi_data = []
    for dataset in preprocessed_datasets:
        dataset_config = config["datasets"].get(dataset)
        tiled_data_dir = tiles_basedir / dataset
        label = dataset_config.get("label")
        for wsi_id_dir in tiled_data_dir.iterdir():
            if wsi_id_dir.is_dir():
                patch_dir = wsi_id_dir / "tiles"
                if patch_dir.exists():
                    patch_files = [str(p) for p in patch_dir.glob("*.png")]
                    if len(patch_files) == 0:
                        logger.warning(f"WSI in dataset {dataset} and with "
                                       f"ID {wsi_id_dir.name} has not tiles "
                                       f"thus it has been discarded.")
                        continue
                    wsi_data.append({
                        "wsi_id": wsi_id_dir.name,
                        "label": label,
                        "patch_dir": str(patch_dir),
                        "patch_files": patch_files,
                    })
    output_json = tiles_basedir / "tiles_metadata.json"
    with open(output_json, 'w') as f:
        json.dump(wsi_data, f, indent=4)

    logger.info(f"Writing tiles metadata to {output_json}")
