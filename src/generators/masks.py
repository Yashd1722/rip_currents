"""
Four-Region Mask Decomposition (Body, Submerged Smear, Wake Ring, Shadow).
"""

from typing import Any
import cv2
import numpy as np


def generate_4region_masks(
    body_mask: np.ndarray,
    sun_azimuth_deg: float = 45.0,
    smear_factor: float = 1.5,
    wake_factor: float = 0.6,
) -> dict[str, Any]:
    """
    Decompose swimmer body mask into 4 interaction regions.

    Returns dict with keys: 'body', 'smear', 'wake', 'shadow', 'union', 'gt_bbox'.
    """
    if body_mask.ndim == 3:
        body_mask = cv2.cvtColor(body_mask, cv2.COLOR_BGR2GRAY)

    _, body = cv2.threshold(body_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = body.shape
        return {
            "body": body,
            "smear": np.zeros_like(body),
            "wake": np.zeros_like(body),
            "shadow": np.zeros_like(body),
            "union": body,
            "gt_bbox": (0, 0, w, h),
        }

    bx, by, bw, bh = cv2.boundingRect(np.vstack(contours))

    # 1. Submerged Smear (vertical downward dilation + soft blur)
    smear_shift = max(3, int(bh * smear_factor))
    kernel_smear = np.zeros((smear_shift, 3), dtype=np.uint8)
    kernel_smear[:, 1] = 1
    smear = cv2.threshold(cv2.dilate(body, kernel_smear), 50, 255, cv2.THRESH_BINARY)[1]

    # Ground truth bounding box = Body + Submerged Smear
    gt_mask = cv2.bitwise_or(body, smear)
    gt_cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_bbox = cv2.boundingRect(np.vstack(gt_cnts)) if gt_cnts else (bx, by, bw, bh)

    # 2. Wake Ring (isotropic dilation minus body)
    wake_k = max(3, int(bw * wake_factor) // 2 * 2 + 1)
    wake_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wake_k, wake_k))
    wake = cv2.subtract(cv2.dilate(body, wake_kernel), body)

    # 3. Shadow (offset ellipse along sun azimuth vector)
    rad = np.radians(sun_azimuth_deg)
    dx, dy = int(np.cos(rad) * bh * 0.4), int(np.sin(rad) * bh * 0.4)
    shadow = np.zeros_like(body)
    cv2.ellipse(shadow, (bx + bw // 2 + dx, by + bh + dy // 2), (max(4, int(bw * 0.7)), max(3, int(bh * 0.3))), sun_azimuth_deg, 0, 360, 255, -1)

    # Combined inpainting canvas mask
    union = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(body, smear), wake), shadow)

    return {
        "body": body,
        "smear": smear,
        "wake": wake,
        "shadow": shadow,
        "union": union,
        "gt_bbox": gt_bbox,
    }
