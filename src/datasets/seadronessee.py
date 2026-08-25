"""
SeaDronesSee Dataset Handler (Multi-Swimmer Metadata Filtering & WebDAV Downloader).

Features:
- Groups ALL valid swimmers per image (supports images with multiple swimmers).
- Draws ALL swimmer bounding boxes per full-resolution debug frame.
- Strict ZERO contamination check with instances_val.json.
"""

from typing import Any
import json
from pathlib import Path
import cv2
import numpy as np
import requests
from tqdm import tqdm

from config import config
from datasets.base import BaseDataset


class SeaDronesSeeDataset(BaseDataset):
    def __init__(self, annotation_file: str = "instances_train.json") -> None:
        super().__init__("SeaDronesSee")
        self.annotation_path = config.ANNOTATIONS_DIR / annotation_file

    def filter_candidates(
        self,
        max_alt_m: float = config.MAX_ALTITUDE_M,
        pitch_min_deg: float = config.PITCH_MIN_DEG,
        pitch_max_deg: float = config.PITCH_MAX_DEG,
        min_bbox_h: int = config.MIN_BBOX_HEIGHT_PX,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Filter candidates by altitude, gimbal pitch, and bbox height."""
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotations file missing: {self.annotation_path}")

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        val_json_path = config.ANNOTATIONS_DIR / "instances_val.json"
        val_img_ids = set()
        if val_json_path.exists():
            with open(val_json_path, "r", encoding="utf-8") as vf:
                val_data = json.load(vf)
                val_img_ids = {img["id"] for img in val_data.get("images", [])}

        img_dict = {img["id"]: img for img in data.get("images", [])}
        cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}

        matching = []
        for ann in data.get("annotations", []):
            if cat_map.get(ann["category_id"]) not in ["swimmer", "floater", "person"]:
                continue
            img_meta = img_dict.get(ann["image_id"])
            if not img_meta:
                continue

            # ZERO CONTAMINATION ASSERTION
            assert ann["image_id"] not in val_img_ids, f"CRITICAL ERROR: Image ID {ann['image_id']} CONTAMINATED from VAL split!"

            meta = img_meta.get("meta") or {}
            alt = meta.get("height_above_takeoff(meter)", 0.0)
            pitch = meta.get("gimbal_pitch(degrees)", 0.0)
            h = ann["bbox"][3]

            if alt <= max_alt_m and pitch_min_deg <= pitch <= pitch_max_deg and h >= min_bbox_h:
                matching.append({
                    "ann_id": ann["id"],
                    "image_id": ann["image_id"],
                    "file_name": img_meta["file_name"],
                    "bbox": ann["bbox"],
                    "alt_m": alt,
                    "pitch_deg": pitch,
                })

        print(f"[{self.name}] Filtered {len(matching)} annotations across {len(set(m['file_name'] for m in matching))} images.")
        return matching[:limit]

    def _download_image(self, file_name: str, dest_path: Path) -> bool:
        """Download single image over WebDAV."""
        if dest_path.exists():
            return True
        url = config.WEBDAV_BASE + "images/train/" + file_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(url, auth=config.WEBDAV_AUTH, stream=True, timeout=30)
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                return True
        except Exception as exc:
            print(f"Error downloading {file_name}: {exc}")
        return False

    def load_crops(self, limit: int = 200, padding_ratio: float = 0.5) -> list[dict[str, Any]]:
        """Download required source images or load existing harvested crops from crops/manifest.json."""
        for manifest_path in [Path("crops/manifest.json"), config.CROPS_DIR / "manifest.json"]:
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
                    res = []
                    base_dir = manifest_path.parent
                    for entry in manifest_data:
                        crop_file = entry.get("crop") or entry.get("crop_name")
                        if crop_file:
                            full_crop_path = base_dir / crop_file
                            if full_crop_path.exists():
                                res.append({
                                    "crop_path": str(full_crop_path),
                                    "category": entry.get("category", "swimmer"),
                                    "bbox": entry.get("bbox", [0, 0, 100, 100]),
                                })
                    if res:
                        print(f"[{self.name}] Loaded {len(res)} harvested swimmer crops from {manifest_path}")
                        return res[:limit]
                except Exception as exc:
                    print(f"[{self.name}] Warning loading {manifest_path}: {exc}")

        matching = self.filter_candidates(limit=limit)

        config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        config.CROPS_DIR.mkdir(parents=True, exist_ok=True)
        debug_dir = config.OUTPUT_DIR / "debug_boxed"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Group annotations by image file_name so multi-swimmers per image are all processed
        by_image: dict[str, list[dict[str, Any]]] = {}
        for ann in matching:
            by_image.setdefault(ann["file_name"], []).append(ann)

        crops = []
        image_count = 0

        for file_name, annos in tqdm(by_image.items(), desc="Harvesting Multi-Swimmer Crops"):
            img_path = config.IMAGES_DIR / file_name
            if not self._download_image(file_name, img_path):
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            vis = img.copy()

            # Draw ALL swimmer bounding boxes in this image
            for rec in annos:
                x, y, w, h = map(int, rec["bbox"])
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)

                # Add context padding for crop
                pad_w = int(w * padding_ratio)
                pad_h = int(h * padding_ratio)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)

                crop_img = img[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue

                crop_name = f"crop_{rec['ann_id']}_{file_name}"
                crop_path = config.CROPS_DIR / crop_name
                cv2.imwrite(str(crop_path), crop_img)

                rec_copy = dict(rec)
                rec_copy["crop_path"] = str(crop_path)
                rec_copy["crop_bbox"] = [int(x - x1), int(y - y1), int(w), int(h)]
                crops.append(rec_copy)

            # Save 10 debug multi-boxed full frames
            if image_count < 10:
                cv2.imwrite(str(debug_dir / f"01_multi_boxed_{image_count:02d}_{file_name}"), vis)
                image_count += 1

        print(f"[{self.name}] Harvested {len(crops)} crops across {len(by_image)} unique images saved to {config.CROPS_DIR}")
        print(f"[{self.name}] Multi-swimmer debug boxed frames saved to {debug_dir}")
        return crops
