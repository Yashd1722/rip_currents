"""
Central Configuration for Diffusion & GAN Person Generation & Detection.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Project paths
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATASET_DIR: Path = ROOT_DIR / "dataset"
    ANNOTATIONS_DIR: Path = DATASET_DIR / "annotations"
    CROPS_DIR: Path = (Path(__file__).resolve().parent.parent / "crops") if (Path(__file__).resolve().parent.parent / "crops" / "manifest.json").exists() else (Path(__file__).resolve().parent.parent / "dataset" / "reference_crops")
    IMAGES_DIR: Path = DATASET_DIR / "reference_images"
    OUTPUT_DIR: Path = DATASET_DIR / "experiments"

    # SeaDronesSee WebDAV details
    WEBDAV_BASE: str = "https://cloud.cs.uni-tuebingen.de/public.php/webdav/Uncompressed%20Version/"
    WEBDAV_AUTH: tuple[str, str] = ("ZZxX65FGnQ8zjBP", "")

    # SeaDronesSee metadata filter thresholds
    MAX_ALTITUDE_M: float = 30.0
    PITCH_MIN_DEG: float = 15.0
    PITCH_MAX_DEG: float = 75.0
    MIN_BBOX_HEIGHT_PX: int = 80

    # Person Generation parameters
    GEN_PERSON_SIZE: int = 512          # Generation resolution for person model (512x512)
    DEFAULT_TARGET_HEIGHT_PX: int = 45 # Final target swimmer height on background (20-60px)

    # Hardware
    DEVICE: str = "cuda"


config = Config()
