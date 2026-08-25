# Modular Rip Current Inpainting & Detection Pipeline

This directory contains a clean, modular, and easy-to-understand pipeline architecture for diffusion-based image synthesis and object detection.

---

## Directory Structure

```
src/
├── config.py             # Central configuration (paths, parameters, hardware)
├── main.py               # Simple, readable entry point script
├── datasets/             # Modular dataset handlers
│   ├── base.py           # Abstract BaseDataset class
│   └── seadronessee.py   # SeaDronesSee metadata filter & WebDAV downloader
├── generators/           # Synthesis & compositing modules
│   ├── masks.py          # 4-Region Mask Decomposition (Body, Smear, Wake, Shadow)
│   ├── copy_paste.py     # Color-matched copy-paste baseline
│   └── inpainting.py     # Generate-Large-Then-Downscale Diffusion pipeline
├── models/               # Detection models
│   └── yolo.py           # YOLOv11 detector formatting & training
└── metrics/              # Quality metrics
    └── evaluate.py       # SSIM & LPIPS Erase-and-Restore evaluation
```

---

## Quick Start

### 1. Run the Main Pipeline
```bash
python src/main.py
```

### 2. How to Add a New Dataset
To add a new dataset (e.g., RipAID, AFO, or custom images):
1. Create a new file in `src/datasets/my_dataset.py`.
2. Inherit from `BaseDataset` in `src/datasets/base.py`:
   ```python
   from datasets.base import BaseDataset

   class MyDataset(BaseDataset):
       def filter_candidates(self, **kwargs):
           # Custom filtering logic
           ...

       def load_crops(self, limit=50):
           # Return list of crop records
           ...
   ```
3. Export your class in `src/datasets/__init__.py`.

### 3. How to Add a New Detection Model
To experiment with a new detector (e.g. Faster R-CNN, RT-DETR):
1. Add a module in `src/models/my_model.py`.
2. Wrap dataset formatting and `.train()` methods following the clean pattern in `src/models/yolo.py`.
