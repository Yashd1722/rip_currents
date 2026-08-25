"""
SeaDronesSee Validation Set Downloader and Swimmer Crop Harvester.

Filters instances_val.json metadata:
- altitude_m <= 30.0
- gimbal_pitch in range 15° .. 75°
- category == swimmer / floater / person
- bbox height >= 80px

Downloads matching uncompressed validation images over WebDAV and extracts validation swimmer crops.
"""

import json
from pathlib import Path
import cv2
import requests
from tqdm import tqdm

from config import config

WEBDAV_VAL_BASE = "https://cloud.cs.uni-tuebingen.de/public.php/webdav/Uncompressed%20Version/images/val/"

VAL_IMAGES_DIR = config.DATASET_DIR / "reference_val_images"
VAL_CROPS_DIR = config.DATASET_DIR / "reference_val_crops"


def download_val_image(file_name: str, dest_path: Path) -> bool:
    """Download single validation image over WebDAV."""
    if dest_path.exists():
        return True

    url = WEBDAV_VAL_BASE + file_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, auth=config.WEBDAV_AUTH, stream=True, timeout=30)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True
    except Exception as exc:
        print(f"Error downloading val image {file_name}: {exc}")
    return False


def harvest_validation_set(padding_ratio: float = 0.5) -> list[dict]:
    """Filter instances_val.json and harvest validation swimmer crops."""
    val_ann_path = config.ANNOTATIONS_DIR / "instances_val.json"
    if not val_ann_path.exists():
        raise FileNotFoundError(f"Validation annotation file missing: {val_ann_path}")

    with open(val_ann_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    img_dict = {img["id"]: img for img in val_data.get("images", [])}
    cat_map = {c["id"]: c["name"] for c in val_data.get("categories", [])}

    matching = []
    for ann in val_data.get("annotations", []):
        if cat_map.get(ann["category_id"]) not in ["swimmer", "floater", "person"]:
            continue
        img_meta = img_dict.get(ann["image_id"])
        if not img_meta:
            continue

        meta = img_meta.get("meta") or {}
        alt = meta.get("height_above_takeoff(meter)", 0.0)
        pitch = meta.get("gimbal_pitch(degrees)", 0.0)
        h = ann["bbox"][3]

        if alt <= config.MAX_ALTITUDE_M and config.PITCH_MIN_DEG <= pitch <= config.PITCH_MAX_DEG and h >= config.MIN_BBOX_HEIGHT_PX:
            matching.append({
                "ann_id": ann["id"],
                "image_id": ann["image_id"],
                "file_name": img_meta["file_name"],
                "width": img_meta["width"],
                "height": img_meta["height"],
                "bbox": ann["bbox"],
                "alt_m": alt,
                "pitch_deg": pitch,
            })

    print(f"[SeaDronesSee Val] Filtered {len(matching)} annotations across {len(set(m['file_name'] for m in matching))} validation images.")

    VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VAL_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    by_image: dict[str, list[dict]] = {}
    for ann in matching:
        by_image.setdefault(ann["file_name"], []).append(ann)

    extracted = []
    for file_name, img_annos in tqdm(by_image.items(), desc="Harvesting Val Crops"):
        img_path = VAL_IMAGES_DIR / file_name
        if not download_val_image(file_name, img_path):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        for ann in img_annos:
            x, y, w, h = ann["bbox"]
            pad_w = int(w * padding_ratio)
            pad_h = int(h * padding_ratio)
            x1, y1 = max(0, int(x - pad_w)), max(0, int(y - pad_h))
            x2, y2 = min(img_w, int(x + w + pad_w)), min(img_h, int(y + h + pad_h))

            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            crop_name = f"val_crop_{ann['ann_id']}_{file_name}"
            crop_path = VAL_CROPS_DIR / crop_name
            cv2.imwrite(str(crop_path), crop_img)

            rec_copy = dict(ann)
            rec_copy["crop_path"] = str(crop_path)
            rec_copy["crop_bbox"] = [int(x - x1), int(y - y1), int(w), int(h)]
            extracted.append(rec_copy)

    index_p = VAL_CROPS_DIR / "index.json"
    with open(index_p, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)

    print(f"[SeaDronesSee Val] Successfully harvested {len(extracted)} validation crops to {VAL_CROPS_DIR}")
    return extracted


if __name__ == "__main__":
    harvest_validation_set()
