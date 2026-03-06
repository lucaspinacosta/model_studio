# Model Studio Workflow

This repository contains a YOLO workflow with:

- Inference GUI for images/videos.
- Polygon labeling GUI and YOLO-seg dataset export.
- Pseudo-label + train pipeline (`Optimize` tab).
- Direct training (`detect` or `obb`) in `Training` tab.
- `.pt -> .onnx` conversion in `Converter` tab.

## Project Layout

- `scripts/python/`: GUI and Python automation scripts.
- `scripts/bash/`: shell scripts (including AMD OBB training launcher).
- `runs/`: training outputs.
- `models/`: model artifacts (ONNX, checkpoints, exports).
- `pt/`: local `.pt` models.
- `Backup/`: archived datasets/images/models.
- `requirements.rocm.txt`: ROCm stack Python requirements.
- `requirements.nvidia.txt`: NVIDIA/CUDA stack Python requirements.

## Environment Setup

### AMD ROCm

```bash
python -m venv .venv-rocm --system-site-packages
source .venv-rocm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.rocm.txt
```

### NVIDIA CUDA

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.nvidia.txt
```

## Launch GUI

```bash
.venv-rocm/bin/python scripts/python/model_viewer_gui.py
```

Use `.venv/bin/python` on NVIDIA if needed.

## GUI Tabs

### Inference

- Load model (`.pt`, `.onnx`, etc.), image folder, or video.
- Adjustable confidence threshold.

### Labeling

- Draw polygon annotations per class.
- Export YOLO-seg dataset with train/test/valid splits.

### Optimize

- Pseudo-label unlabeled images, then train.
- Platform profiles: AMD / NVIDIA / CPU.
- Uses `scripts/python/pseudo_label_and_train.py`.

### Training

- Task selector: `detect` or `obb`.
- Platform selector: `AMD (ROCm)` or `NVIDIA (CUDA)`.
- Runs task-aware training command and output paths (`runs/detect` or `runs/obb`).

### Converter

- Convert `.pt` to `.onnx` with task/imgsz/opset/runtime controls.
- Output folder must already exist (converter does not create folders).

## Dataset Path Behavior

The GUI now resolves dataset paths from your current structure and falls back across common locations:

- `data/...`
- `Backup/Data/...`

When switching training task (`obb` <-> `detect`), dataset YAML defaults are updated automatically if current value is old/invalid.

## CLI: AMD OBB Training

```bash
./scripts/bash/train_obb_amd.sh --device 0 --batch 8
```

The script now auto-resolves default OBB dataset YAML from available `data/` or `Backup/Data/` locations.

## CLI: Direct Ultralytics Training

### Detect

```bash
.venv-rocm/bin/yolo detect train \
  data=/absolute/path/to/data.yaml \
  model=yolo11n.pt \
  epochs=100 imgsz=640 batch=8 device=0 workers=8 \
  project=runs/detect name=my_detect_run
```

### OBB

```bash
.venv-rocm/bin/yolo obb train \
  data=/absolute/path/to/data.yaml \
  model=yolo11n-obb.pt \
  epochs=100 imgsz=640 batch=8 device=0 workers=8 \
  project=runs/obb name=my_obb_run
```

## Notes

- ONNX inference models should be loaded with the correct task (`detect`, `obb`, etc.).
- For AMD/ROCm setups, validation/AMP settings may need conservative values depending on torchvision/ROCm compatibility.
