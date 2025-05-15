import pyspng
import torch
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


class TileAndCoordDataset(Dataset):
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

    def get_coordinates_from_tile_path(self, tile_path):
        positions = tile_path.stem.split("__")[1]

        # Extract x and y coordinates
        pos_parts = positions.split("_")
        x_coord = int(pos_parts[0].replace("x", ""))
        y_coord = int(pos_parts[1].replace("y", ""))

        return x_coord, y_coord

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        x_coord, y_coord = self.get_coordinates_from_tile_path(tile_path)
        image = self._load_image(tile_path)

        if self.preprocess:
            image = self.preprocess(image)

        return image, torch.tensor([x_coord, y_coord], dtype=torch.long)


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
