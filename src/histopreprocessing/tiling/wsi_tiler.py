import logging
from pathlib import Path
from itertools import product
from multiprocessing.pool import ThreadPool
import json

import numpy as np
import pandas as pd
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import uniform_filter

DEBUG = False


class WSITilerWithMask:

    def __init__(self,
                 wsi_path,
                 mask_path,
                 output_dir,
                 magnification=10,
                 tile_size=224,
                 threshold=0.8,
                 save_masks=False,
                 raise_error_mag=True,
                 save_tile_overlay=False):
        # Initialize parameters and objects
        self.wsi = OpenSlide(wsi_path)
        self.mask_array = np.array(Image.open(mask_path)) != 0
        self.wsi_id = Path(mask_path).stem.split(".")[0]
        self.output_dir = Path(output_dir) / self.wsi_id
        self.tiles_output_dir = self.output_dir / "tiles"

        self.output_dir_mask = self.output_dir / "mask" if save_masks else None
        self.magnification = magnification
        self.tile_size = tile_size
        self.threshold = threshold
        self.save_masks = save_masks
        self.save_tile_overlay = save_tile_overlay
        self.raise_error_mag = raise_error_mag
        self.patch_metadata = []
        self.x_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-x", "nan"))
        self.y_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-y", "nan"))

        # Directories setup
        self.output_dir.mkdir(exist_ok=True)
        self.tiles_output_dir.mkdir(exist_ok=True)
        if save_masks:
            self.output_dir_mask.mkdir(exist_ok=True)

        # Scaling calculations
        self.base_magnification = self.get_base_magnification()
        self.downsample_factor = self.base_magnification / magnification
        self.level, self.level_tile_size = self.get_best_level()

        self.level0_tile_size = int(np.ceil(tile_size *
                                            self.downsample_factor))

        # Mask alignment scale factor
        self.mask_scale_factor = self._get_mask_scale_factor()
        self.mask_tile_size = int(
            round(self.level0_tile_size / self.mask_scale_factor))

        # Create the overlay for visualizing selected tiles on WSI with mask if enabled
        if save_tile_overlay:
            mask_height, mask_width = self.mask_array.shape

            # Create WSI thumbnail at the mask scale
            self.wsi_thumbnail = self.wsi.get_thumbnail(
                (mask_width, mask_height)).convert("RGB")

            # mask_width, mask_height = self.wsi_thumbnail.size

            # # Resize mask to match thumbnail size and create an RGB mask
            # mask_resized = Image.fromarray(
            #     (self.mask_array * 255).astype(np.uint8)).resize(
            #         (mask_width, mask_height), Image.NEAREST)

            # # Convert mask to binary (0 or 1) for multiplication
            # mask_binary = np.array(mask_resized) / 255
            # mask_rgb = np.stack([mask_binary] * 3, axis=-1)

            # # Apply the mask by multiplying with the thumbnail

            # thumbnail_array = np.array(self.wsi_thumbnail)
            # masked_thumbnail = (thumbnail_array * mask_rgb).astype(np.uint8)
            # self.wsi_thumbnail = Image.fromarray(masked_thumbnail)

    def _get_mask_scale_factor(self):
        # Calculate separate scale factors for width and height
        wsi_width, wsi_height = self.wsi.dimensions
        mask_width, mask_height = self.mask_array.T.shape

        scale_factor_width = wsi_width / mask_width
        scale_factor_height = wsi_height / mask_height

        # Check if the scale factors are close to each other
        if not np.isclose(scale_factor_width, scale_factor_height, rtol=1e-3):
            raise ValueError(
                f"Scale factors for width ({scale_factor_width}) and height ({scale_factor_height}) differ significantly."
            )

        # Use the mean scale factor if they are close
        return np.mean([scale_factor_width, scale_factor_height])

    def get_base_magnification(self):
        """Retrieve base magnification from WSI metadata or infer it from MPP."""
        # Check if magnification is available
        magnification_keys = [
            "aperio.AppMag",
            "openslide.objective-power",
            "hamamatsu.XResolution",
            "hamamatsu.ObjectiveMagnification",
        ]
        for key in magnification_keys:
            mag = self.wsi.properties.get(key)
            if mag:
                return float(mag)

        if self.raise_error_mag:
            raise ValueError(
                f"Magnification metadata is missing for WSI {self.wsi_path.name}. "
                "Please ensure the WSI has magnification information or "
                "set `raise_error_mag` to False to attempt inference.")

        # Attempt to infer magnification based on MPP if not available
        mpp_x = float(self.wsi.properties.get("openslide.mpp-x", "nan"))
        reference_mpp_40x = 0.25  # Assume 0.25 microns/pixel for 40x as a reference

        if not np.isnan(mpp_x):
            estimated_magnification = reference_mpp_40x / mpp_x * 40
            logging.warning(
                f"Inferred magnification from MPP as {estimated_magnification:.2f}x based on MPP: {mpp_x}."
            )
            return estimated_magnification
        else:
            logging.warning("Base magnification not found; defaulting to 40x.")
            return 40.0

    def get_best_level(self):
        """Find the best level in OpenSlide for the desired magnification."""
        level = self.wsi.get_best_level_for_downsample(self.downsample_factor)
        downsample_level = self.wsi.level_downsamples[level]
        level_tile_size = int(
            np.ceil(self.tile_size * self.downsample_factor /
                    downsample_level))

        if level_tile_size != self.tile_size:
            downsample_level = self.wsi.level_downsamples[level + 1]
            level_p1_tile_size = int(
                np.ceil(self.tile_size * self.downsample_factor /
                        downsample_level))
            if level_p1_tile_size == self.tile_size:
                level += 1
                level_tile_size = level_p1_tile_size
        return level, level_tile_size

    def get_coordinates(self):
        """
        Return the coordinate of each potential complete 
        squared tile at level 0
        """
        return list(
            product(
                range(
                    0,
                    self.wsi.dimensions[1] - self.level0_tile_size + 1,
                    self.level0_tile_size,
                ),
                range(
                    0,
                    self.wsi.dimensions[0] - self.level0_tile_size + 1,
                    self.level0_tile_size,
                ),
            ))

    def __call__(self, coords):
        y, x = coords  # Coordinates for the tile

        tile_id = f'{self.wsi_id}__x{x}_y{y}'
        # Calculate the corresponding region in the mask
        mask_x = int(x // self.mask_scale_factor)
        mask_y = int(y // self.mask_scale_factor)

        # Get the mask patch (clipping to avoid out of bounds)
        mask_patch = self.mask_array[mask_y:mask_y + self.mask_tile_size,
                                     mask_x:mask_x + self.mask_tile_size]
        coverage = np.mean(mask_patch > 0)

        # If coverage meets the threshold, save the WSI tile
        keep = 0
        if coverage >= self.threshold:

            keep = 1
            tile = self.wsi.read_region(
                (x, y),
                self.level,
                (self.level_tile_size, self.level_tile_size),
            )
            if self.level_tile_size != self.tile_size:
                tile = tile.resize((self.tile_size, self.tile_size),
                                   Image.LANCZOS)
            tile = tile.convert('RGB')  # Convert to RGB if necessary

            # Save the tile as a PNG
            tile_path = self.tiles_output_dir / f"{tile_id}.png"
            tile.save(tile_path)

            # Outline the selected tiles on the thumbnail overlay
            if self.save_tile_overlay:
                outline = Image.new("RGBA",
                                    (self.mask_tile_size, self.mask_tile_size),
                                    (0, 255, 0, 128))
                self.wsi_thumbnail.paste(outline, (mask_x, mask_y), outline)

            # Optionally save mask tile
            if self.save_masks:
                mask_patch_image = Image.fromarray(
                    (mask_patch * 255).astype(np.uint8))
                mask_patch_image = mask_patch_image.resize(
                    (self.tile_size, self.tile_size), Image.NEAREST)
                mask_tile_filename = f'{self.wsi_id}__{x}_y{y}.png'
                mask_tile_path = self.output_dir_mask / mask_tile_filename
                mask_patch_image.save(mask_tile_path)

        # Save metadata
        self.patch_metadata.append({
            "tile_id":
            tile_id,
            "x_level0":
            x,
            "y_level0":
            y,
            "x_current_level":
            int(x // self.downsample_factor),
            "y_current_level":
            int(y // self.downsample_factor),
            "row":
            int(y // self.level0_tile_size),
            "column":
            int(x // self.level0_tile_size),
            "keep":
            keep,
            "mask_coverage_ratio":
            coverage,
        })

    def save_overlay(self):
        if self.save_tile_overlay:
            overlay_path = self.output_dir / f"{self.wsi_id}__tile_overlay.png"
            self.wsi_thumbnail.save(overlay_path)

    def save_metadata(self):
        """
        Save metadata to a CSV file using pandas.
        """
        # Convert the metadata list to a pandas DataFrame
        patch_metadata_df = pd.DataFrame(self.patch_metadata).sort_values(
            by=['row', 'column'], ascending=[True, True])

        # Save the DataFrame to a CSV file
        patch_metadata_df.to_csv(
            self.output_dir / f"{self.wsi_id}__tiling_results.csv",
            index=False,
        )

        metadata = {
            "tile_magnification": self.magnification,
            "base_magnification": self.base_magnification,
            "x_px_size_tile": self.x_px_size_level0 * self.downsample_factor,
            "y_px_size_tile": self.y_px_size_level0 * self.downsample_factor,
            "x_px_size_base": self.x_px_size_level0,
            "y_px_size_base": self.y_px_size_level0,
        }
        with open(self.output_dir / f"{self.wsi_id}__metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logging.info(f"Metadata saved to {self.output_dir}")


class WSITilerWithSuperPixelMask(WSITilerWithMask):

    def __init__(self,
                 wsi_path,
                 mask_path,
                 output_dir,
                 magnification=10,
                 tile_size=224,
                 threshold=0.8,
                 save_masks=False,
                 raise_error_mag=True,
                 save_tile_overlay=False):
        # Initialize parameters and objects
        self.wsi = OpenSlide(wsi_path)

        self.mask_array = np.array(Image.open(mask_path))
        labels = np.unique(self.mask_array)
        labels = labels[labels != 0]
        self.labels = labels

        self.wsi_id = Path(mask_path).stem.replace("_segments", "")
        self.output_dir = Path(output_dir) / self.wsi_id
        self.tiles_output_dir = self.output_dir / "tiles"

        self.output_dir_mask = self.output_dir / "mask" if save_masks else None
        self.magnification = magnification
        self.tile_size = tile_size
        self.threshold = threshold
        self.save_masks = save_masks
        self.save_tile_overlay = save_tile_overlay
        self.raise_error_mag = raise_error_mag
        self.patch_metadata = []
        self.x_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-x", "nan"))
        self.y_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-y", "nan"))

        # Directories setup
        self.output_dir.mkdir(exist_ok=True)
        self.tiles_output_dir.mkdir(exist_ok=True)
        if save_masks:
            self.output_dir_mask.mkdir(exist_ok=True)

        # Scaling calculations
        self.base_magnification = self.get_base_magnification()
        self.downsample_factor = self.base_magnification / magnification
        self.level, self.level_tile_size = self.get_best_level()

        self.level0_tile_size = int(np.ceil(tile_size *
                                            self.downsample_factor))

        # Mask alignment scale factor
        self.mask_scale_factor = self._get_mask_scale_factor()
        self.mask_tile_size = int(
            round(self.level0_tile_size / self.mask_scale_factor))

        # Create the overlay for visualizing selected tiles on WSI with mask if enabled
        if save_tile_overlay:
            mask_height, mask_width = self.mask_array.shape

            # Create WSI thumbnail at the mask scale
            self.wsi_thumbnail = self.wsi.get_thumbnail(
                (mask_width, mask_height)).convert("RGB")
            thumbnail_size = self.wsi_thumbnail.size  # (width, height)

            # Check if resizing is needed
            if thumbnail_size != (mask_width, mask_height):
                self.wsi_thumbnail = self.wsi_thumbnail.resize(
                    (mask_width, mask_height), Image.Resampling.LANCZOS)
                logging.info(
                    f"Resizing performed for {mask_path.stem}, original size"
                    f" {thumbnail_size}, new size {self.wsi_thumbnail.size}.")

    def save_overlay(self):
        """
        Generate and save an overlay image where each superpixel label has a unique color.
        """
        if not self.save_tile_overlay:
            return

        overlay_path = self.output_dir / f"{self.wsi_id}__tile_overlay.png"

        # Generate random colors for each label
        np.random.seed(42)  # Fix seed for consistent colors
        label_colors = {
            label: tuple(np.random.randint(0, 255, 3))
            for label in self.labels
        }

        # Create an overlay mask
        overlay = np.zeros(
            (self.mask_array.shape[0], self.mask_array.shape[1], 3),
            dtype=np.uint8)

        for label in self.labels:
            mask = self.mask_array == label
            overlay[mask] = label_colors[label]  # Assign label-specific color

        # Convert overlay to a PIL image
        overlay_img = Image.fromarray(overlay.astype(np.uint8))

        # Ensure WSI thumbnail is in RGBA mode for blending
        self.wsi_thumbnail = self.wsi_thumbnail.convert("RGBA")
        overlay_img = overlay_img.convert("RGBA")

        # Blend the overlay with the WSI thumbnail
        alpha = 0.4  # Transparency level
        blended_overlay = Image.blend(self.wsi_thumbnail, overlay_img, alpha)

        # Save overlay image
        blended_overlay.save(overlay_path)
        logging.info(f"Overlay saved: {overlay_path}")

    def __call__(self, coords):
        y, x = coords  # Coordinates for the tile

        # Calculate the corresponding region in the mask
        mask_x = int(x // self.mask_scale_factor)
        mask_y = int(y // self.mask_scale_factor)

        # Get the mask patch (clipping to avoid out of bounds)
        mask_patch = self.mask_array[mask_y:mask_y + self.mask_tile_size,
                                     mask_x:mask_x + self.mask_tile_size]
        coverage = np.mean(mask_patch > 0)

        # If coverage meets the threshold, save the WSI tile
        keep = 0
        if coverage >= self.threshold:
            labels, counts = np.unique(mask_patch, return_counts=True)
            ind = np.argmax(counts)
            label_coverage = counts[ind] / self.mask_tile_size**2
            if label_coverage < self.threshold:
                return
            label = labels[ind]

            tile_id = f'{self.wsi_id}__l{label}__x{x}_y{y}'

            keep = 1
            tile = self.wsi.read_region(
                (x, y),
                self.level,
                (self.level_tile_size, self.level_tile_size),
            )
            if self.level_tile_size != self.tile_size:
                tile = tile.resize((self.tile_size, self.tile_size),
                                   Image.LANCZOS)
            tile = tile.convert('RGB')  # Convert to RGB if necessary

            # Save the tile as a PNG
            tile_path = self.tiles_output_dir / f"{tile_id}.png"
            tile.save(tile_path)

            # Outline the selected tiles on the thumbnail overlay
            if self.save_tile_overlay:
                outline = Image.new("RGBA",
                                    (self.mask_tile_size, self.mask_tile_size),
                                    (0, 255, 0, 128))
                self.wsi_thumbnail.paste(outline, (mask_x, mask_y), outline)

            # Optionally save mask tile
            if self.save_masks:
                mask_patch_image = Image.fromarray(
                    (mask_patch * 255).astype(np.uint8))
                mask_patch_image = mask_patch_image.resize(
                    (self.tile_size, self.tile_size), Image.NEAREST)
                mask_tile_filename = f'{self.wsi_id}__{x}_y{y}.png'
                mask_tile_path = self.output_dir_mask / mask_tile_filename
                mask_patch_image.save(mask_tile_path)

        # Save metadata
            self.patch_metadata.append({
                "tile_id":
                tile_id,
                "x_level0":
                x,
                "y_level0":
                y,
                "x_current_level":
                int(x // self.downsample_factor),
                "y_current_level":
                int(y // self.downsample_factor),
                "row":
                int(y // self.level0_tile_size),
                "column":
                int(x // self.level0_tile_size),
                "keep":
                keep,
                "label":
                label,
                "mask_coverage_ratio":
                coverage,
                "label_coverage_ratio":
                label_coverage,
            })


class WSITilerWithSuperPixelMaskWithOverlap(WSITilerWithMask):

    def __init__(
        self,
        wsi_path,
        mask_path,
        output_dir,
        magnification=10,
        tile_size=224,
        threshold=0.8,
        raise_error_mag=True,
        save_tile_overlay=False,
        average_superpixel_area=4868800,  # in microns**2
        average_n_tiles=10,
        max_iter=1000,
    ):
        # Initialize parameters and objects
        self.wsi = OpenSlide(wsi_path)

        self.mask_array = np.array(Image.open(mask_path))
        labels = np.unique(self.mask_array)
        labels = labels[labels != 0]
        self.labels = labels

        self.wsi_id = Path(mask_path).stem.replace("_segments", "")
        self.output_dir = Path(output_dir) / self.wsi_id
        self.tiles_output_dir = self.output_dir / "tiles"

        self.magnification = magnification
        self.tile_size = tile_size
        self.threshold = threshold
        self.save_tile_overlay = save_tile_overlay
        self.raise_error_mag = raise_error_mag
        self.average_superpixel_area = average_superpixel_area
        self.average_n_tiles = average_n_tiles
        self.max_iter = max_iter

        self.patch_metadata = []
        self.x_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-x", "nan"))
        self.y_px_size_level0 = float(
            self.wsi.properties.get("openslide.mpp-y", "nan"))

        # Directories setup
        self.output_dir.mkdir(exist_ok=True)
        self.tiles_output_dir.mkdir(exist_ok=True)

        # Scaling calculations
        self.base_magnification = self.get_base_magnification()
        self.downsample_factor = self.base_magnification / magnification
        self.level, self.level_tile_size = self.get_best_level()

        self.level0_tile_size = int(np.ceil(tile_size *
                                            self.downsample_factor))

        # Mask alignment scale factor
        self.mask_scale_factor = self._get_mask_scale_factor()
        self.mask_tile_size = int(
            round(self.level0_tile_size / self.mask_scale_factor))

        self.x_px_size_mask = self.x_px_size_level0 * self.mask_scale_factor
        self.y_px_size_mask = self.y_px_size_level0 * self.mask_scale_factor

        # Create the overlay for visualizing selected tiles on WSI with mask if enabled
        if save_tile_overlay:
            mask_height, mask_width = self.mask_array.shape

            # Create WSI thumbnail at the mask scale
            self.wsi_thumbnail = self.wsi.get_thumbnail(
                (mask_width, mask_height)).convert("RGB")
            thumbnail_size = self.wsi_thumbnail.size  # (width, height)

            # Check if resizing is needed
            if thumbnail_size != (mask_width, mask_height):
                self.wsi_thumbnail = self.wsi_thumbnail.resize(
                    (mask_width, mask_height), Image.Resampling.LANCZOS)
                logging.info(
                    f"Resizing performed for {mask_path.stem}, original size"
                    f" {thumbnail_size}, new size {self.wsi_thumbnail.size}.")

    def save_overlay(self):
        """
        Generate and save an overlay image where each superpixel label has a unique color.
        """
        if not self.save_tile_overlay:
            return

        overlay_path = self.output_dir / f"{self.wsi_id}__tile_overlay.png"

        # Generate random colors for each label
        np.random.seed(42)  # Fix seed for consistent colors
        label_colors = {
            label: tuple(np.random.randint(0, 255, 3))
            for label in self.labels
        }

        # Create an overlay mask
        overlay = np.zeros(
            (self.mask_array.shape[0], self.mask_array.shape[1], 3),
            dtype=np.uint8)

        for label in self.labels:
            mask = self.mask_array == label
            overlay[mask] = label_colors[label]  # Assign label-specific color

        # Convert overlay to a PIL image
        overlay_img = Image.fromarray(overlay.astype(np.uint8))

        # Ensure WSI thumbnail is in RGBA mode for blending
        self.wsi_thumbnail = self.wsi_thumbnail.convert("RGBA")
        overlay_img = overlay_img.convert("RGBA")

        # Blend the overlay with the WSI thumbnail
        alpha = 0.4  # Transparency level
        blended_overlay = Image.blend(self.wsi_thumbnail, overlay_img, alpha)

        # Save overlay image
        blended_overlay.save(overlay_path)
        logging.info(f"Overlay saved: {overlay_path}")

    def __call__(self, label):
        """Process a single superpixel label, tiling it efficiently."""

        positions = np.where(self.mask_array == label)
        superpixel_area = positions[0].size
        superpixel_area_micron = superpixel_area * self.x_px_size_mask * self.y_px_size_mask
        n_tiles = int(
            np.round(superpixel_area_micron / self.average_superpixel_area *
                     self.average_n_tiles))

        if superpixel_area <= self.threshold * self.mask_tile_size**2:
            return

        min_y, max_y = positions[0].min(), positions[0].max()
        min_x, max_x = positions[1].min(), positions[1].max()

        # Crop mask to the bounding box
        cropped_mask = self.mask_array[min_y:max_y + 1, min_x:max_x + 1]

        # Compute coverage map **only within the cropped region**
        coverage_map = uniform_filter((cropped_mask == label).astype(float),
                                      size=self.mask_tile_size)

        # Find valid positions in the **cropped mask coordinates**
        valid_y, valid_x = np.where(coverage_map >= self.threshold)
        if valid_x.size == 0:
            return

        # Convert cropped coordinates to **global mask coordinates**
        valid_y += min_y - self.mask_tile_size // 2
        valid_x += min_x - self.mask_tile_size // 2

        # Ensure coordinates stay within bounds
        valid_y = np.clip(valid_y, 0,
                          self.mask_array.shape[0] - self.mask_tile_size)
        valid_x = np.clip(valid_x, 0,
                          self.mask_array.shape[1] - self.mask_tile_size)

        # Stack them as valid tile positions
        valid_positions = np.column_stack((valid_y, valid_x))

        # Shuffle and take at most `n_tiles`
        np.random.shuffle(valid_positions)
        valid_positions = valid_positions[:n_tiles]

        # If not enough tiles, sample with replacement
        if len(valid_positions) < n_tiles:
            additional_indices = np.random.choice(len(valid_positions),
                                                  size=n_tiles -
                                                  len(valid_positions),
                                                  replace=True)
            additional_positions = valid_positions[additional_indices]
            valid_positions = np.vstack(
                (valid_positions, additional_positions))

        for mask_y, mask_x in valid_positions:
            x_level0 = int(mask_x * self.mask_scale_factor +
                           np.random.randint(-self.mask_scale_factor //
                                             2, self.mask_scale_factor // 2))
            y_level0 = int(mask_y * self.mask_scale_factor +
                           np.random.randint(-self.mask_scale_factor //
                                             2, self.mask_scale_factor // 2))

            tile_id = f'{self.wsi_id}__l{label}__x{x_level0}_y{y_level0}'

            tile = self.wsi.read_region(
                (x_level0, y_level0),
                self.level,
                (self.level_tile_size, self.level_tile_size),
            )
            if self.level_tile_size != self.tile_size:
                tile = tile.resize((self.tile_size, self.tile_size),
                                   Image.LANCZOS)
            tile = tile.convert('RGB')

            # Save the tile
            tile_path = self.tiles_output_dir / f"{tile_id}.png"
            tile.save(tile_path)

            # Outline the selected tiles on the thumbnail overlay
            if self.save_tile_overlay:
                outline = Image.new("RGBA",
                                    (self.mask_tile_size, self.mask_tile_size),
                                    (0, 255, 0, 128))
                self.wsi_thumbnail.paste(outline, (mask_x, mask_y), outline)

            # Save metadata
            self.patch_metadata.append({
                "tile_id":
                tile_id,
                "x_level0":
                x_level0,
                "y_level0":
                y_level0,
                "x_current_level":
                int(x_level0 // self.downsample_factor),
                "y_current_level":
                int(y_level0 // self.downsample_factor),
                "row":
                int(y_level0 // self.level0_tile_size),
                "column":
                int(x_level0 // self.level0_tile_size),
                "label":
                label,
                "label_coverage_ratio":
                coverage_map[mask_y - min_y,
                             mask_x - min_x],  # Adjusted for cropped mask
            })

    def call_wip(self, label):

        positions = np.where(self.mask_array == label)
        superpixel_area = positions[0].size
        superpixel_area_micron = superpixel_area * self.x_px_size_mask * self.y_px_size_mask
        n_tiles = int(
            np.round(superpixel_area_micron / self.average_superpixel_area *
                     self.average_n_tiles))

        if superpixel_area <= self.threshold * self.mask_tile_size**2:
            return

        coverage_map = uniform_filter((self.mask_array == label).astype(float),
                                      size=self.mask_tile_size)

        # Get valid positions where the coverage is above the threshold
        valid_y, valid_x = np.where(coverage_map >= self.threshold)

        if valid_x.size == 0:
            return

        # Convert positions to **top-left corners** of tiles
        valid_y = valid_y - self.mask_tile_size // 2
        valid_x = valid_x - self.mask_tile_size // 2

        # Ensure the coordinates stay within image bounds
        valid_y = np.clip(valid_y, 0,
                          self.mask_array.shape[0] - self.mask_tile_size)
        valid_x = np.clip(valid_x, 0,
                          self.mask_array.shape[1] - self.mask_tile_size)

        # Stack them as valid tile positions
        valid_positions = np.column_stack((valid_y, valid_x))
        # Shuffle and take at most `n_tiles`
        np.random.shuffle(valid_positions)
        valid_positions = valid_positions[:n_tiles]
        if len(valid_positions) < n_tiles:
            additional_indices = np.random.choice(len(valid_positions),
                                                  size=n_tiles -
                                                  len(valid_positions),
                                                  replace=True)
            additional_positions = valid_positions[additional_indices]
            valid_positions = np.vstack(
                (valid_positions, additional_positions))

        for mask_y, mask_x in valid_positions:

            # Get the mask patch (clipping to avoid out of bounds)
            # mask_patch = self.mask_array[mask_y:mask_y + self.mask_tile_size,
            #                              mask_x:mask_x + self.mask_tile_size]

            x_level0 = int(mask_x * self.mask_scale_factor + np.random.randint(
                -self.mask_scale_factor // 2,
                self.mask_scale_factor // 2,
            ))
            y_level0 = int(mask_y * self.mask_scale_factor + np.random.randint(
                -self.mask_scale_factor // 2,
                self.mask_scale_factor // 2,
            ))

            tile_id = f'{self.wsi_id}__l{label}__x{x_level0}_y{y_level0}'

            tile = self.wsi.read_region(
                (x_level0, y_level0),
                self.level,
                (self.level_tile_size, self.level_tile_size),
            )
            if self.level_tile_size != self.tile_size:
                tile = tile.resize((self.tile_size, self.tile_size),
                                   Image.LANCZOS)
            tile = tile.convert('RGB')  # Convert to RGB if necessary

            # Save the tile as a PNG
            tile_path = self.tiles_output_dir / f"{tile_id}.png"
            tile.save(tile_path)

            # Outline the selected tiles on the thumbnail overlay
            if self.save_tile_overlay:
                outline = Image.new("RGBA",
                                    (self.mask_tile_size, self.mask_tile_size),
                                    (0, 255, 0, 128))
                self.wsi_thumbnail.paste(outline, (mask_x, mask_y), outline)

            # Save metadata
            self.patch_metadata.append({
                "tile_id":
                tile_id,
                "x_level0":
                x_level0,
                "y_level0":
                y_level0,
                "x_current_level":
                int(x_level0 // self.downsample_factor),
                "y_current_level":
                int(y_level0 // self.downsample_factor),
                "row":
                int(y_level0 // self.level0_tile_size),
                "column":
                int(x_level0 // self.level0_tile_size),
                "label":
                label,
                "label_coverage_ratio":
                coverage_map[mask_y, mask_x],
            })


def main(
    wsi_path,
    mask_path,
    output_dir=None,
    magnification=20,
    tile_size=224,
    threshold=0.5,
    num_workers=12,
    save_mask=False,
    save_tile_overlay=False,
):
    tile_processor = WSITilerWithMask(
        wsi_path=wsi_path,
        mask_path=mask_path,
        output_dir=output_dir,
        magnification=magnification,
        tile_size=tile_size,
        threshold=threshold,
        save_masks=save_mask,
        save_tile_overlay=save_tile_overlay,
    )

    # Define coordinates for tiles
    coordinates = tile_processor.get_coordinates()

    if DEBUG:
        for coord in tqdm(coordinates):
            tile_processor(coord)
        tile_processor.save_overlay()
        return

    # Process tiles in parallel using ThreadPool
    with ThreadPool(processes=num_workers) as pool:
        list(
            tqdm(pool.imap_unordered(tile_processor, coordinates),
                 total=len(coordinates),
                 desc="Processing tiles"))

    # Save the tile overlay after processing if enabled
    tile_processor.save_overlay()
    tile_processor.save_metadata(mask_path / "metadata.csv")
