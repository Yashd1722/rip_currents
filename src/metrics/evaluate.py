"""
Quality & Realism Evaluation Metrics (SSIM & LPIPS).
"""

from typing import Any
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

try:
    import torch
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

from config import config


class Evaluator:
    """Evaluates composite image realism against ground truth frame."""

    def __init__(self, device: str = config.DEVICE) -> None:
        self.device = device
        self.lpips_fn = None
        if HAS_LPIPS and torch.cuda.is_available() and device != "cpu":
            try:
                self.lpips_fn = lpips.LPIPS(net="alex").to(device)
            except Exception:
                pass

    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Compute Structural Similarity Index (SSIM)."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray1, gray2, full=True)
        if mask is not None:
            if mask.shape != gray1.shape:
                mask = cv2.resize(mask, (gray1.shape[1], gray1.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 127
            if np.any(mask_bool):
                return float(np.mean(diff[mask_bool]))

        return float(score)

    def compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute LPIPS perceptual distance."""
        if self.lpips_fn is None:
            return -1.0

        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0

        t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.lpips_fn(t1, t2).item()
        return float(dist)

    def evaluate(self, original_img: np.ndarray, composite_img: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Compute full SSIM, masked SSIM, and LPIPS metrics."""
        return {
            "ssim_full": self.compute_ssim(original_img, composite_img),
            "ssim_masked": self.compute_ssim(original_img, composite_img, mask=mask),
            "lpips": self.compute_lpips(original_img, composite_img),
        }
