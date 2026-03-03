#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Defaults
DATA_YAML="${ROOT_DIR}/data/SandwichPanel.v7i.yolov8-obb/data.yaml"
MODEL="yolo11n-obb.pt"
EPOCHS=100
IMGSZ=640
BATCH=16
DEVICE="0"          # AMD GPU via ROCm (first GPU)
WORKERS=8
PROJECT="${ROOT_DIR}/runs/obb"
NAME="sandwich_panel_obb_v1"
HSA_GFX_OVERRIDE="10.3.0"  # Compatibility for gfx1031 cards on some ROCm builds
VAL="False"  # Disable val by default to avoid torchvision ROCm NMS issues on some setups
AMP="False"  # Disable AMP checks on ROCm setups where torchvision NMS GPU kernel is unavailable

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --data <path>       Dataset YAML (default: ${DATA_YAML})
  --model <model>     Model/checkpoint (default: ${MODEL})
  --epochs <int>      Epochs (default: ${EPOCHS})
  --imgsz <int>       Image size (default: ${IMGSZ})
  --batch <int>       Batch size (default: ${BATCH})
  --device <id|cpu>   Device (default: ${DEVICE})
  --workers <int>     Data loader workers (default: ${WORKERS})
  --project <path>    Output project dir (default: ${PROJECT})
  --name <string>     Run name (default: ${NAME})
  --hsa-gfx <M.m.p>   HSA_OVERRIDE_GFX_VERSION value (default: ${HSA_GFX_OVERRIDE})
  --val <True|False>  Run validation during training (default: ${VAL})
  --amp <True|False>  Enable AMP mixed precision (default: ${AMP})
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_YAML="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --hsa-gfx) HSA_GFX_OVERRIDE="$2"; shift 2 ;;
    --val) VAL="$2"; shift 2 ;;
    --amp) AMP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -f "${DATA_YAML}" ]]; then
  echo "Dataset YAML not found: ${DATA_YAML}"
  exit 1
fi

# Prefer ROCm venv, fallback to current venv
if [[ -x "${ROOT_DIR}/.venv-rocm/bin/yolo" ]]; then
  YOLO_BIN="${ROOT_DIR}/.venv-rocm/bin/yolo"
elif [[ -x "${ROOT_DIR}/.venv/bin/yolo" ]]; then
  YOLO_BIN="${ROOT_DIR}/.venv/bin/yolo"
else
  echo "Could not find yolo binary in .venv-rocm or .venv"
  exit 1
fi

echo "Using: ${YOLO_BIN}"
echo "Training OBB model on device=${DEVICE}"

# AMD ROCm compatibility override for GPUs like RX 6700 XT (gfx1031).
if [[ "${DEVICE}" != "cpu" ]]; then
  export HSA_OVERRIDE_GFX_VERSION="${HSA_GFX_OVERRIDE}"
  echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
fi

# Validate GPU visibility before launching when device is not CPU.
if [[ "${DEVICE}" != "cpu" ]]; then
  if ! "${YOLO_BIN%/yolo}/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)"; then
    echo "Warning: torch precheck did not detect GPU in selected environment."
    echo "Current env: ${YOLO_BIN%/yolo}"
    echo "Continuing anyway. If training fails, rerun with --device cpu or fix ROCm setup."
  fi
fi

"${YOLO_BIN}" obb train \
  data="${DATA_YAML}" \
  model="${MODEL}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  val="${VAL}" \
  amp="${AMP}" \
  project="${PROJECT}" \
  name="${NAME}"
