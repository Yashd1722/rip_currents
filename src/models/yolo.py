"""
YOLO Object Detector Module (Format Conversion, Training, and Validation Evaluation).
"""

from pathlib import Path
from typing import Any
import yaml

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from config import config


class YOLODetector:
    """Modular wrapper for YOLO detector training, testing, and validation evaluation."""

    def __init__(self, model_name: str = "yolo11n.pt") -> None:
        self.model_name = model_name

    def prepare_dataset(
        self,
        dataset_name: str,
        train_records: list[dict[str, Any]],
        val_records: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Create standard YOLO directory structure and dataset.yaml config."""
        ds_dir = config.DATASET_DIR / "yolo_datasets" / dataset_name
        img_tr = ds_dir / "images" / "train"
        lbl_tr = ds_dir / "labels" / "train"
        img_val = ds_dir / "images" / "val"
        lbl_val = ds_dir / "labels" / "val"

        for d in [img_tr, lbl_tr, img_val, lbl_val]:
            d.mkdir(parents=True, exist_ok=True)

        def write_records(recs: list[dict[str, Any]], img_dir: Path, lbl_dir: Path):
            import shutil
            for rec in recs:
                src_p = Path(rec.get("image_path") or rec.get("crop_path", ""))
                if not src_p.exists():
                    continue

                dest_img = img_dir / src_p.name
                if not dest_img.exists():
                    shutil.copy2(src_p, dest_img)

                txt_name = dest_img.stem + ".txt"
                lbl_p = lbl_dir / txt_name

                w_img = rec.get("width", 3840)
                h_img = rec.get("height", 2160)

                if "crop_bbox" in rec:
                    bx, by, bw, bh = rec["crop_bbox"]
                    # If crop image, width/height is crop image size
                    if src_p.exists():
                        import cv2
                        c_img = cv2.imread(str(src_p))
                        if c_img is not None:
                            h_img, w_img = c_img.shape[:2]
                else:
                    bx, by, bw, bh = rec.get("bbox", [0, 0, 10, 10])

                xc = (bx + bw / 2.0) / float(max(1, w_img))
                yc = (by + bh / 2.0) / float(max(1, h_img))
                wn = bw / float(max(1, w_img))
                hn = bh / float(max(1, h_img))

                with open(lbl_p, "w", encoding="utf-8") as f:
                    f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        write_records(train_records, img_tr, lbl_tr)
        if val_records:
            write_records(val_records, img_val, lbl_val)

        yaml_data = {
            "path": str(ds_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "swimmer"},
        }

        yaml_path = ds_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        return yaml_path

    def train(
        self,
        yaml_path: Path,
        experiment_name: str,
        epochs: int = 15,
        imgsz: int = 1024,
    ) -> tuple[Path, dict[str, float]]:
        """Train YOLO detector model and return weights path & metrics."""
        if not HAS_YOLO:
            print("[YOLODetector] Ultralytics package unavailable; skipping training.")
            return Path(""), {"mAP50": 0.0, "mAP50-95": 0.0}

        model = YOLO(self.model_name)
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            device=config.DEVICE if config.DEVICE != "cpu" else "cpu",
            name=experiment_name,
            project="runs",
            verbose=True,
        )

        weights_path = Path(results.save_dir) / "weights" / "best.pt"

        metrics = model.val()
        train_metrics = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        return weights_path, train_metrics

    def evaluate_testing(
        self,
        weights_path: Path,
        val_yaml_path: Path,
    ) -> dict[str, float]:
        """Test and evaluate detector performance on held-out validation set."""
        if not HAS_YOLO or not weights_path.exists():
            print(f"[YOLODetector] Weights path missing at {weights_path}; returning default testing metrics.")
            return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}

        model = YOLO(str(weights_path))
        metrics = model.val(data=str(val_yaml_path))

        return {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
