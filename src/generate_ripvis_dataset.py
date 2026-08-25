"""
Generate People in Water Synthetic Dataset on RipVIS Background Frames using Diffusion / GAN models.
"""

from typing import Any
import argparse
import json
import random
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from config import config
from datasets import SeaDronesSeeDataset, RipVISDataset
from generators import copy_paste_swimmer, DiffusionPersonGenerator


def generate_ripvis_synthetic_dataset(
    num_samples: int = 100,
    method: str = "diffusion", # "diffusion" or "copypaste"
    output_name: str = "ripvis_person_diffusion",
) -> Path:
    """Generate synthetic person dataset on RipVIS background frames."""
    out_dir = config.DATASET_DIR / "ripvis_synthetic" / output_name
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Generating People in Water on RipVIS Dataset ({method.upper()}) ===")
    print(f"Target count: {num_samples} samples | Output: {out_dir}")

    # Load RipVIS background frames
    ripvis = RipVISDataset()
    bg_frames = ripvis.get_background_frames(limit=num_samples * 2)
    if not bg_frames:
        raise RuntimeError("No RipVIS background frames found in dataset/video_*_frames")

    # Load reference swimmer crops
    seadrone = SeaDronesSeeDataset()
    crops = seadrone.load_crops(limit=100)
    if not crops:
        raise RuntimeError("No swimmer reference crops available.")

    generator = DiffusionPersonGenerator(device=config.DEVICE) if method == "diffusion" else None

    dataset_index = []

    for i in tqdm(range(num_samples), desc="Generating synthetic person frames"):
        bg_path = random.choice(bg_frames)
        bg_img = cv2.imread(str(bg_path))
        if bg_img is None:
            continue

        bg_h, bg_w = bg_img.shape[:2]

        crop_rec = random.choice(crops)
        crop_img = cv2.imread(crop_rec["crop_path"])
        if crop_img is None:
            continue

        # Extract body mask
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, body_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # Pick random placement on verified water surface (rejecting sky & sand with 200px border margin safety)
        from generators.diffusion import water_mask_simple
        wmask = water_mask_simple(bg_img)
        margin = 200
        
        # Zero out margins
        wmask[:margin, :] = False
        wmask[-margin:, :] = False
        wmask[:, :margin] = False
        wmask[:, -margin:] = False

        water_y, water_x = np.where(wmask)
        if len(water_y) > 0:
            rand_idx = random.randint(0, len(water_y) - 1)
            py, px = int(water_y[rand_idx]), int(water_x[rand_idx])
        else:
            px = random.randint(margin, max(margin + 1, bg_w - margin))
            py = random.randint(margin, max(margin + 1, bg_h - margin))

        # Perspective placement scale rule: higher in frame (top) = smaller, lower in frame (bottom) = larger
        row_frac = max(0.0, min(1.0, float(py) / float(bg_h)))
        target_h = int(25 + row_frac * (140 - 25))  # Exact Phase 2/3 scale calibration (25px to 140px)

        if method == "diffusion" and generator is not None:
            # Generate person directly inside RipVIS water scene using Generative Inpainting with Resolution Fix
            synth_img, gt_bbox = generator.generate_person_in_water(
                bg_img=bg_img,
                placement_xy=(px, py),
                reference_crop=crop_img,
                target_person_height=target_h,
                strength=0.98,
            )
        else:
            # Baseline Copy-Paste
            target_h = random.randint(25, 60)
            synth_img, gt_bbox = copy_paste_swimmer(
                bg_img=bg_img,
                swimmer_crop=crop_img,
                body_mask=body_mask,
                placement_xy=(px, py),
                target_height_px=target_h,
            )

        img_filename = f"ripvis_person_{i:05d}.jpg"
        save_img_path = img_dir / img_filename
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_img_path), synth_img)

        # Write YOLO label format
        bx, by, bw, bh = gt_bbox
        xc = (bx + bw / 2.0) / float(bg_w)
        yc = (by + bh / 2.0) / float(bg_h)
        wn = bw / float(bg_w)
        hn = bh / float(bg_h)

        lbl_filename = f"ripvis_person_{i:05d}.txt"
        save_lbl_path = lbl_dir / lbl_filename
        with open(save_lbl_path, "w", encoding="utf-8") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        dataset_index.append({
            "id": i,
            "file_name": img_filename,
            "image_path": str(save_img_path),
            "label_path": str(save_lbl_path),
            "bg_source": str(bg_path),
            "bbox": gt_bbox,
            "width": bg_w,
            "height": bg_h,
        })

    # Save dataset index JSON
    index_path = out_dir / "dataset.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(dataset_index, f, indent=2)

    print(f"\nDataset generation complete! Generated {len(dataset_index)} samples at {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic people-in-water on RipVIS dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of synthetic images to generate")
    parser.add_argument("--method", type=str, choices=["diffusion", "copypaste"], default="diffusion")
    parser.add_argument("--output_name", type=str, default="ripvis_person_diffusion")
    args = parser.parse_args()

    generate_ripvis_synthetic_dataset(
        num_samples=args.num_samples,
        method=args.method,
        output_name=args.output_name,
    )
