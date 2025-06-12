# src/histopreprocessing/labeling/base.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence


class Labeller(ABC):
    """
    Abstract base class for any tile labeller.
    It takes embeddings or tile representations and outputs labels.
    """

    @abstractmethod
    def label_batch(self, embeddings: np.ndarray) -> Sequence[str]:
        """
        Given a batch of embeddings (or representations), return predicted labels.

        Args:
            embeddings (np.ndarray): A batch of N x D embeddings (float32 or float64)

        Returns:
            Sequence[str]: A list of N labels, one per embedding.
        """
        pass
