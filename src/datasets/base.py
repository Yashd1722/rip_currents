"""
Base Abstract Dataset Class for Modular Dataset Extension.

Inherit from BaseDataset to add new dataset handlers (e.g. SeaDronesSee, RipVIS, RipAID, Custom).
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def filter_candidates(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Filter dataset candidates according to specified criteria."""
        pass

    @abstractmethod
    def load_crops(self, limit: int = 50) -> list[dict[str, Any]]:
        """Extract or load reference crops for compositing."""
        pass
