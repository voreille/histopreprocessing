import pyspng
import torch
import torch.nn.functional as F
from openslide import OpenSlide
from PIL import Image
from torch.utils.data import Dataset


def collate_fn_ragged(batch):
    wsi_ids, embeddings, labels = zip(*batch)
    return list(wsi_ids), list(embeddings), torch.stack(labels)


class TileDataset(Dataset):
    def __init__(self, tile_paths, preprocess=None):
        """
        Tile-level dataset that returns individual tile images from a list of paths.

        Args:
            tile_paths (list): List of paths to tile images for a WSI.
            augmentation (callable, optional): augmentation to apply to each tile image.
            transform (callable, optional): Transform to apply to each tile image.
        """
        self.tile_paths = tile_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.tile_paths)

    def _load_image(self, path):
        """Loads an image efficiently using OpenCV."""
        # img = cv2.imread(path)  # OpenCV loads as BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        img = Image.fromarray(img).convert("RGB")  # Convert to PIL
        return img

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        image = self._load_image(tile_path)

        if self.preprocess:
            image = self.preprocess(image)

        return image, str(tile_path)


class TileDatasetFromRawWSI(Dataset):
    def __init__(
        self,
        raw_wsi_path,
        coordinates,
        tile_level=0,
        tile_size_at_level0=224,
        target_tile_size=224,
        preprocess=None,
    ):
        self.wsi = OpenSlide(raw_wsi_path)
        self.coords = coordinates
        self.tile_level = tile_level
        self.tile_size = tile_size_at_level0
        self.target_tile_size = target_tile_size
        self.preprocess = preprocess

        self.check_tile_size()

    def __len__(self):
        return len(self.coords)

    def check_tile_size(self):
        coord = self.coords[0]

        img = self.wsi.read_region(
            coord, self.tile_level, (self.tile_size, self.tile_size)
        ).convert("RGB")
        if img.size != (self.target_tile_size, self.target_tile_size):
            print(
                f"Warning: Image size {img.size} does not match target size {self.target_tile_size}"
            )

    def __getitem__(self, idx):
        coord = self.coords[idx]

        img = self.wsi.read_region(
            coord, self.tile_level, (self.tile_size, self.tile_size)
        ).convert("RGB")
        img = self.preprocess(img)
        return img, str(coord)

class TileDatasetFromRawWSI(Dataset):
    def __init__(
        self,
        raw_wsi_path,
        coordinates,
        tile_level=0,
        tile_size_at_level0=224,
        target_tile_size=224,
        preprocess=None,
        cache_whole_level=True,  # New parameter to enable caching
    ):
        self.wsi = OpenSlide(raw_wsi_path)
        self.coords = coordinates
        self.tile_level = tile_level
        self.tile_size = tile_size_at_level0
        self.target_tile_size = target_tile_size
        self.preprocess = preprocess
        self.cache_whole_level = cache_whole_level
        
        # Cache the entire level in memory if requested
        self.cached_image = None
        self.level_downsample = None
        
        if self.cache_whole_level and self.tile_level > 0:
            print(f"Caching WSI at level {self.tile_level} in memory...")
            # Get the downsample factor for this level
            self.level_downsample = self.wsi.level_downsamples[self.tile_level]
            
            # Get dimensions at this level
            level_dims = self.wsi.level_dimensions[self.tile_level]
            
            # Read the entire level into memory
            self.cached_image = self.wsi.read_region(
                (0, 0), self.tile_level, level_dims
            ).convert("RGB")
            
            print(f"Cached image size: {self.cached_image.size}")
        
        self.check_tile_size()

    def __len__(self):
        return len(self.coords)

    def check_tile_size(self):
        if len(self.coords) == 0:
            print("Warning: No coordinates provided")
            return
            
        coord = self.coords[0]
        img = self.wsi.read_region(
            coord, self.tile_level, (self.tile_size, self.tile_size)
        ).convert("RGB")
        if img.size != (self.target_tile_size, self.target_tile_size):
            print(
                f"Warning: Image size {img.size} does not match target size {self.target_tile_size}"
            )

    def __getitem__(self, idx):
        coord = self.coords[idx]
        
        # Use cached image if available
        if self.cached_image is not None and self.tile_level > 0:
            # Convert level 0 coordinates to coordinates at the cached level
            x_at_level = int(coord[0] / self.level_downsample)
            y_at_level = int(coord[1] / self.level_downsample)
            
            # Calculate the tile size at this level
            tile_size_at_level = int(self.tile_size / self.level_downsample)
            
            # Extract the region from the cached image
            img = self.cached_image.crop((
                x_at_level, 
                y_at_level, 
                x_at_level + tile_size_at_level, 
                y_at_level + tile_size_at_level
            ))
            
            # Resize if needed to match target size
            if img.size != (self.target_tile_size, self.target_tile_size):
                img = img.resize((self.target_tile_size, self.target_tile_size), Image.LANCZOS)
        else:
            # Fall back to standard OpenSlide reading for level 0 or if caching disabled
            img = self.wsi.read_region(
                coord, self.tile_level, (self.tile_size, self.tile_size)
            ).convert("RGB")
        
        if self.preprocess:
            img = self.preprocess(img)
        
        return img, str(coord)