"""
Main Pipeline Runner for People-in-Water Generation & Dataset Storage.

Generates swimmers into ocean water using:
1. Dual Guidance (Text prompt + IP-Adapter visual conditioning)
2. 384x384 Cropped Patch Output (centered on swimmer)
3. Saves training images + YOLO labels to training_results/generated_images/
4. Saves testing images + YOLO labels to testing_results/testing_images/
"""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from config import config
from datasets import SeaDronesSeeDataset, RipVISDataset
from generators import DiffusionPersonGenerator


def write_yolo_label(label_path: Path, bbox_xywh: tuple[int, int, int, int], img_w: int = 384, img_h: int = 384):
    """Write standard YOLO format normalized label file."""
    bx, by, bw, bh = bbox_xywh
    xc = (bx + bw / 2.0) / float(img_w)
    yc = (by + bh / 2.0) / float(img_h)
    wn = bw / float(img_w)
    hn = bh / float(img_h)

    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


def main() -> None:
    print("=== Starting Swimmer Generation & Dataset Storage Pipeline ===")

    train_img_dir = Path("training_results/generated_images")
    train_lbl_dir = Path("training_results/generated_images/labels")
    test_img_dir = Path("testing_results/testing_images")
    test_lbl_dir = Path("testing_results/testing_images/labels")

    for d in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir, Path("debug")]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Loading RipVIS background frames & reference crops...")
    ripvis = RipVISDataset()
    bg_frames = ripvis.get_background_frames(limit=30)

    seadrone = SeaDronesSeeDataset()
    crop_records = seadrone.load_crops(limit=20)

    if not bg_frames or not crop_records:
        print("Required dataset inputs missing. Exiting.")
        return

    print("\n[Step 2] Initializing SDXL Diffusion Person Generator with Dual Guidance...")
    generator = DiffusionPersonGenerator(device=config.DEVICE)

    # Generate Training Samples
    print(f"\n[Step 3] Generating Training Images into {train_img_dir}/...")
    N_TRAIN_SAMPLES = 4
    for idx in range(N_TRAIN_SAMPLES):
        bg_p = bg_frames[idx % len(bg_frames)]
        bg_img = cv2.imread(str(bg_p))
        if bg_img is None:
            continue

        rec = crop_records[idx % len(crop_records)]
        ref_crop = cv2.imread(rec["crop_path"])

        px = int(bg_img.shape[1] * (0.3 + 0.4 * (idx / max(1, N_TRAIN_SAMPLES))))
        py = int(bg_img.shape[0] * (0.3 + 0.4 * (idx / max(1, N_TRAIN_SAMPLES))))

        patch_384, bbox_384 = generator.generate_person_in_water(
            bg_img=bg_img,
            placement_xy=(px, py),
            reference_crop=ref_crop,
            target_person_height=50 + (idx * 5) % 40,
            strength=0.98,
        )

        img_out_p = train_img_dir / f"train_swimmer_{idx:03d}.jpg"
        lbl_out_p = train_lbl_dir / f"train_swimmer_{idx:03d}.txt"

        cv2.imwrite(str(img_out_p), patch_384)
        write_yolo_label(lbl_out_p, bbox_384, img_w=384, img_h=384)
        print(f" Saved Training Image [{idx:02d}]: {img_out_p.name} (bbox: {bbox_384})")

    # Generate Testing Samples
    print(f"\n[Step 4] Generating Testing Images into {test_img_dir}/...")
    N_TEST_SAMPLES = 4
    for idx in range(N_TEST_SAMPLES):
        bg_p = bg_frames[(idx + 15) % len(bg_frames)]
        bg_img = cv2.imread(str(bg_p))
        if bg_img is None:
            continue

        rec = crop_records[(idx + 10) % len(crop_records)]
        ref_crop = cv2.imread(rec["crop_path"])

        px = int(bg_img.shape[1] * (0.4 + 0.3 * (idx / max(1, N_TEST_SAMPLES))))
        py = int(bg_img.shape[0] * (0.4 + 0.3 * (idx / max(1, N_TEST_SAMPLES))))

        patch_384, bbox_384 = generator.generate_person_in_water(
            bg_img=bg_img,
            placement_xy=(px, py),
            reference_crop=ref_crop,
            target_person_height=45 + (idx * 8) % 35,
            strength=0.98,
        )

        img_out_p = test_img_dir / f"test_swimmer_{idx:03d}.jpg"
        lbl_out_p = test_lbl_dir / f"test_swimmer_{idx:03d}.txt"

        cv2.imwrite(str(img_out_p), patch_384)
        write_yolo_label(lbl_out_p, bbox_384, img_w=384, img_h=384)
        print(f" Saved Testing Image [{idx:02d}]: {img_out_p.name} (bbox: {bbox_384})")

    print("\n=======================================================")
    print("      DATASET STORAGE COMPLETED SUCCESSFULLY")
    print("=======================================================")
    print(f"Training Images Saved At : {train_img_dir.resolve()}")
    print(f"Training Labels Saved At : {train_lbl_dir.resolve()}")
    print(f"Testing Images Saved At  : {test_img_dir.resolve()}")
    print(f"Testing Labels Saved At  : {test_lbl_dir.resolve()}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
