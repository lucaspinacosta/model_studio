#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  YOLO_BIN="${ROOT_DIR}/.venv/bin/yolo"
elif [[ -x "${ROOT_DIR}/.venv-rocm/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv-rocm/bin/python"
  YOLO_BIN="${ROOT_DIR}/.venv-rocm/bin/yolo"
else
  PYTHON_BIN="${ROOT_DIR}/yolovenv/bin/python3"
  YOLO_BIN="${ROOT_DIR}/yolovenv/bin/yolo"
fi

COCO_JSON="${ROOT_DIR}/Backup/Data/SandwichPanel.coco/train/_annotations.coco.json"
IMAGES_DIR="${ROOT_DIR}/Backup/Data/SandwichPanel.coco/train"
OUTPUT_DIR="${ROOT_DIR}/data/sandwich_panel_yolo"
MODEL="yolo11n.yaml"
EPOCHS=100
IMGSZ=640
BATCH=16
VAL_RATIO=0.15
DEVICE="cpu"
RUN_NAME="sandwich_panel_retrain"
WORKERS=4
INCLUDE_CLASS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coco-json) COCO_JSON="$2"; shift 2 ;;
    --images-dir) IMAGES_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --include-class) INCLUDE_CLASS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -x "${YOLO_BIN}" ]]; then
  echo "YOLO CLI not found at ${YOLO_BIN}"
  exit 1
fi

echo "Preparing YOLO dataset..."
PREP_ARGS=(
  --coco-json "${COCO_JSON}"
  --images-dir "${IMAGES_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --val-ratio "${VAL_RATIO}"
)
if [[ -n "${INCLUDE_CLASS}" ]]; then
  PREP_ARGS+=(--include-class "${INCLUDE_CLASS}")
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/python/prepare_yolo_dataset.py" \
  "${PREP_ARGS[@]}"

echo "Starting training..."
"${YOLO_BIN}" detect train \
  data="${OUTPUT_DIR}/dataset.yaml" \
  model="${MODEL}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  project="${ROOT_DIR}/runs/train" \
  name="${RUN_NAME}"
