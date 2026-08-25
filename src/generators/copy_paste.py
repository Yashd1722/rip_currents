"""
Copy-Paste Baseline Compositer.
"""

from typing import Any
import cv2
import numpy as np


def match_color_histogram(src_bgr: np.ndarray, tpl_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Match LAB color channel statistics of source crop to target background patch."""
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tpl_lab = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is not None and np.any(mask > 127):
        valid = src_lab[mask > 127]
        src_mean = np.mean(valid, axis=0, keepdims=True).reshape(1, 1, 3)
        src_std = np.maximum(np.std(valid, axis=0, keepdims=True).reshape(1, 1, 3), 1e-5)
    else:
        src_mean = np.mean(src_lab, axis=(0, 1), keepdims=True)
        src_std = np.maximum(np.std(src_lab, axis=(0, 1), keepdims=True), 1e-5)

    tpl_mean = np.mean(tpl_lab, axis=(0, 1), keepdims=True)
    tpl_std = np.std(tpl_lab, axis=(0, 1), keepdims=True)

    matched = (src_lab - src_mean) * (tpl_std / src_std) + tpl_mean
    return cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def copy_paste_swimmer(
    bg_img: np.ndarray,
    swimmer_crop: np.ndarray,
    body_mask: np.ndarray,
    placement_xy: tuple[int, int],
    target_height_px: int = 45,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Composite swimmer crop onto background at specified position and target height."""
    bg_h, bg_w = bg_img.shape[:2]
    c_h, c_w = swimmer_crop.shape[:2]

    scale = target_height_px / float(max(1, c_h))
    new_w, new_h = max(4, int(c_w * scale)), max(4, target_height_px)

    res_crop = cv2.resize(swimmer_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    res_mask = cv2.resize(body_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    px, py = placement_xy
    x1, y1 = max(0, px - new_w // 2), max(0, py - new_h // 2)
    x2, y2 = min(bg_w, x1 + new_w), min(bg_h, y1 + new_h)

    actual_w, actual_h = x2 - x1, y2 - y1
    if actual_w <= 0 or actual_h <= 0:
        return bg_img.copy(), (0, 0, 0, 0)

    crop_patch = res_crop[:actual_h, :actual_w]
    mask_patch = res_mask[:actual_h, :actual_w]
    bg_patch = bg_img[y1:y2, x1:x2]

    crop_patch = match_color_histogram(crop_patch, bg_patch, mask_patch)

    alpha = (mask_patch.astype(np.float32) / 255.0)[:, :, None]
    blended = (crop_patch.astype(np.float32) * alpha + bg_patch.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    res_bg = bg_img.copy()
    res_bg[y1:y2, x1:x2] = blended

    gt_bbox = (x1, y1, actual_w, actual_h)
    return res_bg, gt_bbox
