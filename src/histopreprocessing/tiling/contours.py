import logging
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_patch_coords_from_mask(
    wsi,
    binary_mask,
    target_mpp=0.5,
    patch_size=256,
    step_size=256,
    min_tissue_ratio=0.5,
    tissue_area_threshold=100,
    hole_area_threshold=16,
    max_holes=8,
    ref_patch_size=512,
    mpp_selection_mode="closest_mpp",  # or "highest_mpp"
    segmentation_downsample=64,
):
    """
    Extract patch coordinates from a binary mask at a specific resolution.

    Args:
        wsi: OpenSlide object representing the WSI
        binary_mask: Binary numpy array where tissue regions are marked as 1
        target_mpp: Target microns-per-pixel resolution (default: 0.5, ~20x magnification)
        patch_size: Desired patch size at the target resolution
        step_size: Step size at target resolution
        min_tissue_ratio: Minimum tissue coverage ratio to include a patch (0.0-1.0)
        tissue_area_threshold: Minimum area for a tissue contour (scaled by reference patch)
        hole_area_threshold: Minimum area for a hole contour (scaled by reference patch)
        max_holes: Maximum number of holes to keep per contour
        ref_patch_size: Reference patch size for scaling area thresholds
        mpp_selection_mode: How to select pyramid level based on target MPP:
                    "closest" - choose the level with MPP closest to target
                    "higher_resolution" - choose the highest resolution level with MPP <= target

    Returns:
        dict: Contains:
            'coords': List of tuples (x, y) for patch extraction (at level 0)
            'patch_level': Level to use for patch extraction
            'patch_size': Size of patches to extract at the selected level
            'downsampling_factor': Downsampling factor from level 0 to the selected level
    """
    # Get WSI properties
    level_dimensions = wsi.level_dimensions
    level_downsamples = wsi.level_downsamples

    # Find best level for patch extraction based on target MPP
    patch_level, base_mpp = get_level_for_target_mpp(
        wsi, target_mpp=target_mpp, mpp_selection_mode=mpp_selection_mode
    )

    actual_downsample = level_downsamples[patch_level]
    actual_mpp = base_mpp * actual_downsample

    logger.info(f"Selected level {patch_level} with MPP {actual_mpp:.3f} µm/pixel")
    logger.info(f"Target MPP was {target_mpp:.3f} µm/pixel")

    # Find best level for segmentation (looking for ~64x downsampling)
    segmentation_level = wsi.get_best_level_for_downsample(segmentation_downsample)

    # Resize binary mask to match the segmentation level
    seg_w, seg_h = level_dimensions[segmentation_level]

    if binary_mask.shape[0] != seg_h or binary_mask.shape[1] != seg_w:
        logger.info(f"Resizing mask from {binary_mask.shape} to {seg_h, seg_w}")
        binary_mask = cv2.resize(
            binary_mask.astype(np.uint8),
            (seg_w, seg_h),
            interpolation=cv2.INTER_NEAREST,
        )

    # Get the downsample factor at segmentation level
    seg_scale = level_downsamples[segmentation_level]
    if isinstance(seg_scale, tuple):
        seg_scale_factor = seg_scale[0] * seg_scale[1]
    else:
        seg_scale_factor = seg_scale**2

    # Process binary mask to get tissue and hole contours with proper scaling
    tissue_contours, hole_contours = process_binary_mask_to_contours(
        binary_mask,
        ref_patch_size=ref_patch_size,
        tissue_area_threshold=tissue_area_threshold,
        hole_area_threshold=hole_area_threshold,
        max_holes=max_holes,
        level_downsample=seg_scale_factor,
    )

    if not tissue_contours:
        logger.warning("No valid tissue contours found in the mask.")
        return {
            "coords": [],
            "patch_level": patch_level,
            "patch_size": patch_size,
            "patch_size_at_level_0": 0,
            "downsampling_factor": actual_downsample,
        }

    # Calculate scale factor between segmentation level and patch level
    if isinstance(level_downsamples[patch_level], tuple) and isinstance(
        level_downsamples[segmentation_level], tuple
    ):
        seg_to_patch_scale_x = (
            level_downsamples[patch_level][0] / level_downsamples[segmentation_level][0]
        )
        seg_to_patch_scale_y = (
            level_downsamples[patch_level][1] / level_downsamples[segmentation_level][1]
        )
        seg_to_patch_scale = (seg_to_patch_scale_x, seg_to_patch_scale_y)
    else:
        seg_to_patch_scale = (
            level_downsamples[patch_level] / level_downsamples[segmentation_level]
        )
        seg_to_patch_scale = (seg_to_patch_scale, seg_to_patch_scale)

    # Calculate patch and step size at segmentation level
    patch_size_at_seg_level = int(patch_size / seg_to_patch_scale[0])
    step_size_at_seg_level = int(step_size / seg_to_patch_scale[0])

    logger.info(
        f"Patch size at seg level: {patch_size_at_seg_level}, Step size: {step_size_at_seg_level}"
    )

    # Function to check if a patch at given coordinates has sufficient tissue
    def is_valid_patch(
        x, y, tissue_contours, hole_contours, patch_size_at_seg_level, min_tissue_ratio
    ):
        # Create a test mask for this patch
        patch_mask = np.zeros(
            (patch_size_at_seg_level, patch_size_at_seg_level), dtype=np.uint8
        )

        # Draw all tissue contours that intersect this patch
        for contour in tissue_contours:
            # Translate contour to patch coordinates
            shifted_contour = contour - [x, y]
            cv2.drawContours(patch_mask, [shifted_contour], -1, 1, -1)

        # Remove holes
        for hole_group in hole_contours:
            for hole in hole_group:
                # Translate contour to patch coordinates
                shifted_hole = hole - [x, y]
                cv2.drawContours(patch_mask, [shifted_hole], -1, 0, -1)

        # Calculate tissue coverage
        coverage = np.sum(patch_mask) / (
            patch_size_at_seg_level * patch_size_at_seg_level
        )
        return coverage >= min_tissue_ratio

    # For multiprocessing
    def process_patch_candidate(args):
        (
            x,
            y,
            tissue_contours,
            hole_contours,
            patch_size_at_seg_level,
            min_tissue_ratio,
        ) = args
        if is_valid_patch(
            x,
            y,
            tissue_contours,
            hole_contours,
            patch_size_at_seg_level,
            min_tissue_ratio,
        ):
            # If valid, return the coordinates scaled to level 0
            scale_to_0 = level_downsamples[segmentation_level]
            if isinstance(scale_to_0, tuple):
                x0 = int(x * scale_to_0[0])
                y0 = int(y * scale_to_0[1])
            else:
                x0 = int(x * scale_to_0)
                y0 = int(y * scale_to_0)
            return (x0, y0)
        return None

    # Generate candidate coordinates at segmentation level
    w, h = level_dimensions[segmentation_level]
    candidate_coords = []

    for y in range(0, h - patch_size_at_seg_level + 1, step_size_at_seg_level):
        for x in range(0, w - patch_size_at_seg_level + 1, step_size_at_seg_level):
            candidate_coords.append((x, y))

    logger.info(f"Evaluating {len(candidate_coords)} candidate patches")

    # Use multiprocessing to evaluate patches
    num_workers = min(cpu_count(), 12)  # Limit to 12 cores max

    coord_args = [
        (
            x,
            y,
            tissue_contours,
            hole_contours,
            patch_size_at_seg_level,
            min_tissue_ratio,
        )
        for x, y in candidate_coords
    ]

    with Pool(num_workers) as pool:
        valid_coords = pool.map(process_patch_candidate, coord_args)

    # Filter out None values
    valid_coords = [coord for coord in valid_coords if coord is not None]

    logger.info(f"Found {len(valid_coords)} valid patch coordinates")

    # Calculate the final patch size to use with wsi.read_region
    if isinstance(actual_downsample, tuple):
        patch_size_at_level_0 = int(patch_size * actual_downsample[0])
    else:
        patch_size_at_level_0 = int(patch_size * actual_downsample)

    return {
        "coords": valid_coords,  # Coordinates at level 0
        "patch_level": patch_level,  # Level to use for reading patches
        "patch_size": patch_size,  # Size to use with read_region
        "patch_size_at_level_0": patch_size_at_level_0,  # Size at level 0
        "downsampling_factor": actual_downsample,  # Actual downsampling factor
    }


def get_level_for_target_mpp(wsi, target_mpp=0.5, mpp_selection_mode="closest"):
    """
    Get the WSI level that has the closest MPP to the target value.

    Args:
        wsi: OpenSlide object
        target_mpp: Target microns-per-pixel resolution
        mpp_selection_mode: How to select the level:
                    "closest" - level with MPP closest to target
                    "higher_resolution" - highest resolution level with MPP <= target

    Returns:
        int: Best level index
    """
    if mpp_selection_mode not in ["closest_mpp", "higher_resolution"]:
        raise ValueError(
            f"Invalid mpp_selection_mode: {mpp_selection_mode}. "
            "Choose 'closest_mpp' or 'higher_resolution'."
        )

    try:
        base_mpp_x = float(wsi.properties.get("openslide.mpp-x", 0.25))
        base_mpp_y = float(wsi.properties.get("openslide.mpp-y", 0.25))
        base_mpp = (base_mpp_x + base_mpp_y) / 2  # Average MPP
    except Exception as e:
        logger.warning(
            f"MPP information not available, assuming 0.25 µm/pixel at level 0 (error: {e})"
        )
        base_mpp = 0.25

    if mpp_selection_mode == "higher_resolution":
        target_downsample = target_mpp / base_mpp
        return wsi.get_best_level_for_downsample(target_downsample), base_mpp

    # Find level with MPP closest to target
    best_level = 0
    min_mpp_diff = float("inf")

    for level in range(len(wsi.level_downsamples)):
        level_downsample = wsi.level_downsamples[level]
        # Handle tuple or scalar
        if isinstance(level_downsample, tuple):
            level_scale = (level_downsample[0] + level_downsample[1]) / 2
        else:
            level_scale = level_downsample

        level_mpp = base_mpp * level_scale
        mpp_diff = abs(level_mpp - target_mpp)

        if mpp_diff < min_mpp_diff:
            min_mpp_diff = mpp_diff
            best_level = level

    actual_mpp = base_mpp * wsi.level_downsamples[best_level]
    if isinstance(actual_mpp, tuple):
        actual_mpp = (actual_mpp[0] + actual_mpp[1]) / 2

    logger.info(
        f"Selected level {best_level} with MPP {actual_mpp:.3f} µm/pixel (target: {target_mpp:.3f})"
    )

    return best_level, base_mpp


def process_binary_mask_to_contours(
    binary_mask,
    ref_patch_size=512,
    tissue_area_threshold=100,
    hole_area_threshold=16,
    max_holes=8,
    level_downsample=None,
):
    """
    Process a binary mask into filtered contours using CLAM's approach with clearer parameter names

    Args:
        binary_mask: Binary numpy array where tissue regions are marked as 1
        ref_patch_size: Reference patch size for scaling the filter parameters, 512 in CLAM
        tissue_area_threshold: Minimum area for a tissue contour (scaled by reference patch)
        hole_area_threshold: Minimum area for a hole contour (scaled by reference patch)
        max_holes: Maximum number of holes to keep per contour
        level_downsample: Downsample factor for the current level

    Returns:
        tuple: (tissue_contours, hole_contours)
    """
    # Ensure binary mask is uint8 type
    binary_mask = binary_mask.astype(np.uint8)

    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    # If no contours found, return empty lists
    if len(contours) == 0 or hierarchy is None:
        return [], []

    # Process hierarchy array to match CLAM's format
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # Scale the filter parameters based on the reference patch size and current level
    # This follows CLAM's approach for scaling parameters by level
    scale = 1.0
    if level_downsample is not None:
        # Convert to float if it's a tuple
        if isinstance(level_downsample, tuple):
            scale = level_downsample[0] * level_downsample[1]
        else:
            scale = level_downsample

    scaled_ref_patch_area = int(ref_patch_size**2 / scale)

    # Scale filter parameters by reference patch area
    filter_params = {
        "a_t": tissue_area_threshold * scaled_ref_patch_area,
        "a_h": hole_area_threshold * scaled_ref_patch_area,
        "max_n_holes": max_holes,
    }

    # Filter contours using CLAM's approach
    tissue_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)

    return tissue_contours, hole_contours


def filter_contours(contours, hierarchy, filter_params):
    """
    Filter contours using CLAM's sophisticated area-based approach

    Args:
        contours: List of contours from cv2.findContours
        hierarchy: Hierarchy array from cv2.findContours
        filter_params: Dictionary with filtering parameters:
            - a_t: Area threshold for tissue contours
            - a_h: Area threshold for holes
            - max_n_holes: Maximum number of holes to keep per contour

    Returns:
        tuple: (tissue_contours, hole_contours)
            - tissue_contours: List of filtered tissue contours
            - hole_contours: List of lists, each containing holes for a tissue contour
    """
    filtered = []

    # Find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # Loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # Actual contour
        cont = contours[cont_idx]

        # Indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)

        # Take contour area (includes holes)
        a = cv2.contourArea(cont)

        # Calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]

        # Actual area of foreground contour region (subtract holes)
        a = a - np.array(hole_areas).sum()

        # Skip contours with zero area
        if a == 0:
            continue

        # Keep contour if it exceeds area threshold
        if a >= filter_params["a_t"]:
            filtered.append(cont_idx)
            all_holes.append(holes)

    # Extract foreground contours
    tissue_contours = [contours[cont_idx] for cont_idx in filtered]

    # Process and filter holes
    hole_contours = []
    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]

        # Sort holes by area (largest first)
        unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)

        # Take max_n_holes largest holes
        unfiltered_holes = unfiltered_holes[: filter_params["max_n_holes"]]
        filtered_holes = []

        # Filter holes by area threshold
        for hole in unfiltered_holes:
            if cv2.contourArea(hole) > filter_params["a_h"]:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return tissue_contours, hole_contours
