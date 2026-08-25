"""
RipVIS Background Dataset Handler.

Manages background ocean/rip-current frames extracted from Hugging Face Irikos/RipVIS.
"""

from pathlib import Path
from typing import Any
import random
import cv2

from config import config
from datasets.base import BaseDataset


class RipVISDataset(BaseDataset):
    """Handler for RipVIS background frames."""

    def __init__(self) -> None:
        super().__init__("RipVIS")
        self.dataset_dir = config.DATASET_DIR

    def get_background_frames(self, limit: int | None = None) -> list[Path]:
        """Collect all extracted RipVIS background frame image paths."""
        video_dirs = sorted(self.dataset_dir.glob("video_*_frames"))
        all_frames = []
        for v_dir in video_dirs:
            frames = sorted(v_dir.glob("*.jpg"))
            all_frames.extend(frames)

        print(f"[{self.name}] Found {len(all_frames)} background frames across {len(video_dirs)} videos.")
        if limit:
            random.seed(42)
            all_frames = random.sample(all_frames, min(limit, len(all_frames)))

        return all_frames

    def filter_candidates(self, **kwargs: Any) -> list[dict[str, Any]]:
        """RipVIS background frames do not require filtering."""
        frames = self.get_background_frames()
        return [{"frame_path": str(f)} for f in frames]

    def load_crops(self, limit: int = 50) -> list[dict[str, Any]]:
        """RipVIS is a background source; foreground crops are pulled from reference pool."""
        return []
