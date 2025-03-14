import logging
from pathlib import Path
import multiprocessing
import warnings

import openslide
import numpy as np
from PIL import Image
from skimage.io import imsave
from skimage import segmentation
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

# OpenSlide supported extensions
OPENSLIDE_EXTENSIONS = [
    ".svs", ".ndpi", ".tif", ".tiff", ".vms", ".vmu", ".scn", ".mrxs", ".bif"
]


def generate_overlay(wsi_image, segments):
    """Generate an overlay image of superpixel segmentation on top of a resized WSI."""
    unique_segments = np.unique(segments)
    num_labels = len(unique_segments)

    # Generate random colors for each superpixel
    random_colors = np.random.randint(0,
                                      255,
                                      size=(num_labels, 3),
                                      dtype=np.uint8)

    # Create a colored segmentation image
    colored_segments = np.zeros((segments.shape[0], segments.shape[1], 3),
                                dtype=np.uint8)
    for i, label in enumerate(unique_segments):
        mask = segments == label
        colored_segments[mask] = random_colors[i]

    # Convert to PIL image
    seg_img = Image.fromarray(colored_segments)

    # Blend with original image using transparency
    wsi_image = wsi_image.convert("RGBA")  # Ensure 4-channel RGBA
    seg_img = seg_img.convert("RGBA")

    # Apply transparency (adjust alpha channel)
    alpha = 0.4  # Transparency level
    return Image.blend(wsi_image, seg_img, alpha)


def process_single_mask(args):
    """Function to process a single mask, used for parallel processing."""
    mask_path, raw_data_dir, output_segments, output_overlay, average_tile_size, save_overlay, kwargs = args

    wsi_filename = mask_path.stem
    wsi_id = wsi_filename.replace(".svs_mask_use", "")

    # Find matching WSI
    matching_wsis = [
        f for f in raw_data_dir.rglob(f"*{wsi_id}*")
        if f.suffix in OPENSLIDE_EXTENSIONS
    ]

    if len(matching_wsis) > 1:
        logger.error(
            f"Multiple WSIs match ID '{wsi_id}', please check the dataset.")
        return
    if len(matching_wsis) == 0:
        logger.warning(f"No WSI found for '{wsi_id}', skipping...")
        return

    wsi_slide = openslide.OpenSlide(matching_wsis[0])
    image_resized, segments = superpixel_segmentation_one_image(
        wsi_slide, mask_path, average_tile_size=average_tile_size, **kwargs)

    # Save segmentation mask and filter low contras warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            ".*low contrast image.*",
            UserWarning,
        )
        imsave(output_segments / f"{wsi_id}_segments.tiff",
               segments.astype(np.uint16))  # TIFF format

    # Save overlay image
    if save_overlay:
        overlay_image = generate_overlay(image_resized, segments)
        overlay_image.save(output_overlay / f"{wsi_id}__overlay.png")


def superpixel_segmentation_task(
        raw_data_dir,
        masks_dir,
        output_dir,
        average_tile_size=672,
        save_overlay=True,
        num_workers=None,  # Set to None for automatic CPU allocation
        **kwargs):
    """
    Perform superpixel segmentation for all WSIs found in raw_data_dir using their corresponding masks.
    Uses multiprocessing to speed up execution.
    """
    raw_data_dir = Path(raw_data_dir).resolve()
    masks_dir = Path(masks_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Ensure output directories exist
    output_segments = output_dir / "segments"
    output_overlay = output_dir / "overlay"
    output_segments.mkdir(parents=True, exist_ok=True)
    output_overlay.mkdir(parents=True, exist_ok=True)

    masks_path = list(masks_dir.rglob("*svs_mask_use.png"))

    # Prepare arguments for multiprocessing
    args_list = [(mask_path, raw_data_dir, output_segments, output_overlay,
                  average_tile_size, save_overlay, kwargs)
                 for mask_path in masks_path]

    # Define number of processes (default: all available CPUs)
    num_workers = num_workers or min(multiprocessing.cpu_count(),
                                     len(masks_path))

    logger.info(f"Starting parallel processing with {num_workers} workers.")

    # Run processing in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(
            tqdm(pool.imap_unordered(process_single_mask, args_list),
                 total=len(args_list),
                 desc="Processing WSIs",
                 leave=True))

    logger.info("Superpixel segmentation completed.")


def superpixel_segmentation_one_image(slide,
                                      mask_path,
                                      average_tile_size=672,
                                      **kwargs):
    """
    Perform SLIC superpixel segmentation on a WSI while restricting segmentation
    to a region defined by a binary mask.
    """
    # Load the mask (HistoQC-generated) and get its dimensions
    mask = np.array(Image.open(mask_path).convert("L"))  # Convert to grayscale
    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape  # (H, W) from PIL, OpenSlide uses (W, H)

    # Get the microns per pixel (MPP) values
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)

    if mpp_x is None or mpp_y is None:
        logger.error(f"MPP values are missing from metadata in {mask_path}.")
        raise ValueError(
            "Microns per pixel (MPP) values are missing from the slide metadata."
        )

    # Convert MPP values to float
    mpp_x = float(mpp_x)
    mpp_y = float(mpp_y)

    # Compute number of superpixels
    S_image = slide.dimensions[0] * slide.dimensions[1] * mpp_x * mpp_y
    S_target = (average_tile_size)**2  # Target tile area
    num_segments = int(S_image / S_target)  # Ensure an integer count

    # Get the thumbnail dimensions from OpenSlide
    original_thumbnail = slide.get_thumbnail((width, height))
    original_size = original_thumbnail.size  # (width, height)

    # Check if resizing is needed
    if original_size != (width, height):
        image_resized = original_thumbnail.resize((width, height),
                                                  Image.Resampling.LANCZOS)
        logger.info(
            f"Resizing performed for {mask_path.stem}, original size {original_size}, new size {image_resized.size}."
        )
    else:
        image_resized = original_thumbnail

    # Convert PIL image to NumPy
    image_np = np.array(image_resized)

    # Perform SLIC superpixel segmentation
    segments = segmentation.slic(image_np, n_segments=num_segments, **kwargs)

    # Apply the mask to keep only relevant segments
    segments *= mask  # Zero out regions outside the mask

    return image_resized, segments  # Return resized WSI & segmented regions
