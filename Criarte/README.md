# Criarte YOLO Workflow

This repository supports two training flows:

- YOLO `detect` from COCO conversion scripts.
- YOLO `obb` (oriented bounding boxes) from `data/SandwichPanel.v7i.yolov8-obb`.

## Project Layout

- `scripts/python/`: Python automation scripts and GUI.
- `scripts/bash/`: shell automation scripts.
- `data/`: datasets.
- `runs/`: training outputs.
- `models/`: ONNX model artifacts.
- `pt/`: `.pt` model artifacts.
- `Backup/`: archived datasets and runs.

## Requirements Files

- `requirements.nvidia.txt`: backup copy of the NVIDIA requirements.
- `requirements.rocm.txt`: Python packages for AMD/ROCm workflow.

## OBB Dataset

Dataset YAML used for OBB training:

```bash
data/SandwichPanel.v7i.yolov8-obb/data.yaml
```

## Train OBB Model (AMD GPU / ROCm)

Use this when your machine has an AMD GPU.

### 1) Install ROCm torch (Arch Linux)

```bash
sudo pacman -S --needed python-pytorch-rocm python-torchvision rocminfo rocm-hip-runtime
```

### 2) Create ROCm venv

```bash
cd /home/$USER/Documents/Criarte
python -m venv .venv-rocm --system-site-packages
source .venv-rocm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.rocm.txt
```

### 3) Verify GPU visibility

```bash
.venv-rocm/bin/python -c "import torch; print('gpu=', torch.cuda.is_available(), 'count=', torch.cuda.device_count(), 'hip=', torch.version.hip)"
```

### 4) Train

```bash
./scripts/bash/train_obb_amd.sh --device 0 --batch 8
```

Notes:

- Script sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for RX 6700-class compatibility.
- Script defaults to `--amp False --val False` to avoid ROCm `torchvision::nms` issues during checks/final validation.
- Weights are saved in `runs/obb/<name>/weights/best.pt`.

### 5) Validate on CPU (recommended on this ROCm stack)

```bash
.venv-rocm/bin/yolo obb val \
  model=/home/$USER/Documents/Criarte/runs/obb/sandwich_panel_obb_v15/weights/best.pt \
  data=/home/$USER/Documents/Criarte/data/SandwichPanel.v7i.yolov8-obb/data.yaml \
  device=cpu
```

### 6) Export to ONNX

```bash
.venv-rocm/bin/yolo export \
  model=/home/$USER/Documents/Criarte/runs/obb/sandwich_panel_obb_v15/weights/best.pt \
  format=onnx imgsz=640
```

## Train OBB Model (NVIDIA GPU / CUDA)

Use this on your NVIDIA machine.

### 1) Create venv and install NVIDIA requirements

```bash
cd /home/$USER/Documents/Criarte
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Train directly with Ultralytics

```bash
.venv/bin/yolo obb train \
  data=/home/$USER/Documents/Criarte/data/SandwichPanel.v7i.yolov8-obb/data.yaml \
  model=yolo11n-obb.pt \
  epochs=100 imgsz=640 batch=16 device=0 \
  project=/home/$USER/Documents/Criarte/runs/obb \
  name=sandwich_panel_obb_nvidia_v1
```

### 3) Validate

```bash
.venv/bin/yolo obb val \
  model=/home/$USER/Documents/Criarte/runs/obb/sandwich_panel_obb_nvidia_v1/weights/best.pt \
  data=/home/$USER/Documents/Criarte/data/SandwichPanel.v7i.yolov8-obb/data.yaml \
  device=0
```

## Existing COCO -> Detect Flow

TScript only, no GUI:

```bash
./scripts/bash/retrain_with_new_images.sh \
  --coco-json Backup/Data/SandwichPanel.coco/train/_annotations.coco.json \
  --images-dir Backup/Data/SandwichPanel.coco/train \
  --output-dir data/sandwich_panel_only_yolo \
  --include-class sandwich_panel \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cpu \
  --workers 0 \
  --name sandwich_panel_only_full
```

## GUI

```bash
.venv/bin/python scripts/python/model_viewer_gui.py
```

 `Inference`: model inference for images/videos.

- `Optimize`: pseudo-label + optimize pipeline (renamed from previous Training tab).
  - Includes platform subtabs: `AMD`, `NVIDIA`, `CPU`.
  - Sends platform profile options to optimization script:
    - `--platform` (`amd|nvidia|cpu`)
    - `--hsa-gfx` (AMD only)
    - `--amp true|false`
    - `--train-val true|false`
- `Training`: direct OBB training with platform selector (`AMD (ROCm)` / `NVIDIA (CUDA)`).


# OBS

This software still under validation and development.

Attention to the hardcoded path. This is for a expecific project but still valid and flexible for other models. 

##### The use of GUI is strongly recomended!
