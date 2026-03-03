#!/usr/bin/env python3
"""Pseudo-label unlabeled images with a YOLO model, then train on generated labels."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TRAINABLE_SUFFIXES = {".pt", ".yaml", ".yml"}
GPU_FRAGILE_INFERENCE_SUFFIXES = {".onnx", ".engine", ".xml", ".pb", ".tflite", ".edgetpu", ".mnn", ".ncnn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", type=str, required=True, help="Path to existing model (.pt/.onnx)")
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder with unlabeled images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pseudo_labeled_yolo"), help="Output dataset dir")
    parser.add_argument("--conf", type=float, default=0.7, help="Pseudo-label confidence threshold (0.0-1.0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference/training image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference/training (cpu, 0, 0,1,...)")
    parser.add_argument(
        "--platform",
        type=str,
        choices=["auto", "amd", "nvidia", "cpu"],
        default="auto",
        help="Optimization platform profile (auto, amd, nvidia, cpu).",
    )
    parser.add_argument(
        "--hsa-gfx",
        type=str,
        default="10.3.0",
        help="AMD HSA_OVERRIDE_GFX_VERSION value used when --platform=amd.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (0.0-0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-images", type=int, default=0, help="Limit processed images (0 = all)")

    parser.add_argument("--train-model", type=str, default="", help="Model for training init (default: teacher .pt or yolo11n.yaml)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--workers", type=int, default=4, help="Training workers")
    parser.add_argument(
        "--amp",
        type=str,
        default="",
        help="Override training AMP (true/false). Empty uses platform defaults.",
    )
    parser.add_argument(
        "--train-val",
        type=str,
        default="",
        help="Override validation during train (true/false). Empty uses platform defaults.",
    )
    parser.add_argument("--project", type=str, default="runs/train", help="Ultralytics training project dir")
    parser.add_argument("--name", type=str, default="pseudo_label_train", help="Ultralytics training run name")
    parser.add_argument("--skip-train", action="store_true", help="Only generate pseudo labels and dataset.yaml")
    parser.add_argument(
        "--export-onnx-name",
        type=str,
        default="",
        help="If set and training runs, export trained best.pt to ONNX with this output name.",
    )
    parser.add_argument(
        "--export-onnx-imgsz",
        type=int,
        default=0,
        help="ONNX export image size (0 uses --imgsz).",
    )
    return parser.parse_args()


def parse_optional_bool(raw: str) -> bool | None:
    value = (raw or "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Expected boolean value for flag: true/false")


def resolve_platform_train_settings(args: argparse.Namespace) -> tuple[str, bool | None, bool | None]:
    device = args.device.strip() or "cpu"
    amp_override = parse_optional_bool(args.amp)
    val_override = parse_optional_bool(args.train_val)

    if args.platform == "cpu":
        return "cpu", amp_override if amp_override is not None else False, val_override

    if args.platform == "amd":
        if device.lower() == "cpu":
            device = "0"
        if device.lower() != "cpu":
            os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", args.hsa_gfx)
            print(f"Using AMD ROCm with HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
        # Safer defaults for ROCm stacks where AMP/NMS validation may fail.
        if amp_override is None:
            amp_override = False
        if val_override is None:
            val_override = False
        return device, amp_override, val_override

    if args.platform == "nvidia":
        if device.lower() == "cpu":
            device = "0"
        return device, amp_override if amp_override is not None else True, val_override

    # auto
    return device, amp_override, val_override


def list_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def split_images(paths: list[Path], val_ratio: float, seed: int) -> tuple[set[Path], set[Path]]:
    items = list(paths)
    random.Random(seed).shuffle(items)

    if len(items) <= 1 or val_ratio <= 0:
        return set(items), set()

    n_val = max(1, int(len(items) * val_ratio))
    n_val = min(n_val, len(items) - 1)
    return set(items[n_val:]), set(items[:n_val])


def class_names_from_model(model: YOLO) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys())]
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    return ["object"]


def copy_with_parents(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_dataset_yaml(dataset_dir: Path, class_names: list[str], has_val: bool) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        f"val: {'images/val' if has_val else 'images/train'}",
        "names:",
    ]
    lines.extend([f"  {i}: {name}" for i, name in enumerate(class_names)])
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def default_train_model(teacher_model: str, label_mode: str) -> str:
    if label_mode == "obb":
        local_obb_pt = Path(__file__).resolve().parents[2] / "pt" / "yolo11n-obb.pt"
        if local_obb_pt.exists():
            return str(local_obb_pt)
        return "yolo11n-obb.yaml"
    if teacher_model.lower().endswith(".pt"):
        return teacher_model
    return "yolo11n.yaml"


def validate_train_model_path(train_model: str) -> None:
    suffix = Path(train_model).suffix.lower()
    if suffix and suffix not in TRAINABLE_SUFFIXES:
        raise ValueError(
            f"--train-model '{train_model}' is not trainable. "
            "Use a PyTorch model/config (.pt, .yaml, .yml). "
            "ONNX/TensorRT/OpenVINO/etc. are inference-only."
        )


def should_force_cpu_inference(teacher_model: str, platform: str) -> bool:
    suffix = Path(teacher_model).suffix.lower()
    if suffix not in GPU_FRAGILE_INFERENCE_SUFFIXES:
        return False
    return platform in {"amd", "cpu"}


def _obb_lines(result) -> list[str]:
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return []

    polys = None
    if hasattr(obb, "xyxyxyxyn"):
        polys = obb.xyxyxyxyn.tolist()
    elif hasattr(obb, "xyxyxyxy"):
        raw = obb.xyxyxyxy.tolist()
        h, w = result.orig_shape[:2]
        polys = [[p[0] / w, p[1] / h, p[2] / w, p[3] / h, p[4] / w, p[5] / h, p[6] / w, p[7] / h] for p in raw]
    if polys is None:
        return []

    cls = obb.cls.tolist()
    lines = []
    for c, p in zip(cls, polys):
        # Accept both flat [x1,y1,...,x4,y4] and nested [[x1,y1],...,[x4,y4]].
        if isinstance(p, list) and len(p) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in p):
            flat = [float(p[0][0]), float(p[0][1]), float(p[1][0]), float(p[1][1]), float(p[2][0]), float(p[2][1]), float(p[3][0]), float(p[3][1])]
        else:
            flat = [float(v) for v in p]
            if len(flat) != 8:
                continue
        lines.append(
            f"{int(c)} "
            f"{flat[0]:.6f} {flat[1]:.6f} {flat[2]:.6f} {flat[3]:.6f} "
            f"{flat[4]:.6f} {flat[5]:.6f} {flat[6]:.6f} {flat[7]:.6f}"
        )
    return lines


def _detect_lines(result) -> list[str]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn.tolist()
    cls = boxes.cls.tolist()
    return [f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in zip(cls, xywhn)]


def find_best_weights(project_dir: Path, run_name: str, before: set[Path]) -> Path | None:
    # Prefer exact expected location first.
    exact = project_dir / run_name / "weights" / "best.pt"
    if exact.exists():
        return exact

    # Fallback: find newest best.pt under project dir, prioritize newly created files.
    candidates = sorted(project_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None

    for path in candidates:
        if path not in before:
            return path
    return candidates[0]


def main() -> int:
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")
    if not (0.0 <= args.conf <= 1.0):
        raise ValueError("--conf must be between 0.0 and 1.0")
    if not (0.0 <= args.val_ratio <= 0.5):
        raise ValueError("--val-ratio must be between 0.0 and 0.5")

    image_paths = list_images(args.images_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError("No images found in --images-dir")

    train_device, amp_override, val_override = resolve_platform_train_settings(args)
    infer_device = train_device
    if should_force_cpu_inference(args.teacher_model, args.platform):
        infer_device = "cpu"
        print(
            "Teacher model backend is inference-only and may require CUDA-specific ORT providers. "
            "Forcing pseudo-label inference to CPU for compatibility."
        )
    print(
        f"Platform={args.platform} | train_device={train_device} | infer_device={infer_device} | amp="
        f"{amp_override if amp_override is not None else 'auto'} | train_val="
        f"{val_override if val_override is not None else 'auto'}"
    )

    teacher = YOLO(args.teacher_model)
    class_names = class_names_from_model(teacher)
    train_paths, val_paths = split_images(image_paths, args.val_ratio, args.seed)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    label_mode = "unknown"
    total_images = len(image_paths)

    for i, src_img in enumerate(image_paths, start=1):
        split = "train" if src_img in train_paths else "val"
        rel = src_img.relative_to(args.images_dir)
        dst_img = args.output_dir / "images" / split / rel
        dst_lbl = (args.output_dir / "labels" / split / rel).with_suffix(".txt")

        copy_with_parents(src_img, dst_img)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = teacher.predict(
                source=str(src_img),
                conf=args.conf,
                imgsz=args.imgsz,
                device=infer_device,
                verbose=False,
            )[0]
        except Exception as exc:
            if str(infer_device).lower() != "cpu":
                msg = str(exc).lower()
                onnx_provider_error = (
                    "cudaexecutionprovider" in msg
                    or "libcublaslt.so" in msg
                    or "no data transfer registered" in msg
                    or "onnxruntime" in msg
                )
                if onnx_provider_error:
                    infer_device = "cpu"
                    print(
                        "Inference backend failed on GPU provider (likely ONNXRuntime provider mismatch). "
                        "Falling back to CPU inference for pseudo-labeling."
                    )
                    result = teacher.predict(
                        source=str(src_img),
                        conf=args.conf,
                        imgsz=args.imgsz,
                        device=infer_device,
                        verbose=False,
                    )[0]
                else:
                    raise
            else:
                raise
        lines = _obb_lines(result)
        current_mode = "obb" if lines else "detect"
        if not lines:
            lines = _detect_lines(result)
            current_mode = "detect"
        if lines:
            total_boxes += len(lines)
            if label_mode == "unknown":
                label_mode = current_mode

        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if i % 50 == 0 or i == total_images:
            print(f"Pseudo-labeled {i}/{total_images} images...")

    dataset_yaml = write_dataset_yaml(args.output_dir, class_names, has_val=len(val_paths) > 0)
    print(f"Pseudo-label dataset saved to: {args.output_dir.resolve()}")
    print(f"Train images: {len(train_paths)} | Val images: {len(val_paths)}")
    print(f"Total pseudo boxes: {total_boxes}")
    print(f"Pseudo-label mode: {label_mode}")
    print(f"Dataset config: {dataset_yaml.resolve()}")

    if total_boxes == 0:
        raise RuntimeError("No pseudo-label boxes were generated. Lower --conf or check teacher model.")

    if args.skip_train:
        print("Skipping training as requested (--skip-train).")
        return 0

    project_dir = Path(args.project)
    before_best = set(project_dir.rglob("best.pt")) if project_dir.exists() else set()
    train_model = args.train_model if args.train_model else default_train_model(args.teacher_model, label_mode)
    validate_train_model_path(train_model)
    print(f"Starting training with model: {train_model}")
    trainer = YOLO(train_model)
    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=train_device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    if amp_override is not None:
        train_kwargs["amp"] = amp_override
    if val_override is not None:
        train_kwargs["val"] = val_override
    try:
        trainer.train(**train_kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        nms_backend_error = "torchvision::nms" in msg and "could not run" in msg
        if not nms_backend_error:
            raise

        best_after_fail = find_best_weights(project_dir, args.name, before_best)
        if best_after_fail is None:
            raise
        print(
            "Training hit torchvision NMS backend error during validation/final-eval on current platform. "
            f"Continuing because trained weights were produced: {best_after_fail}"
        )

    if args.export_onnx_name:
        best_pt = find_best_weights(project_dir, args.name, before_best)
        if best_pt is None:
            raise RuntimeError("Training completed but best.pt was not found for ONNX export.")

        export_imgsz = args.export_onnx_imgsz if args.export_onnx_imgsz > 0 else args.imgsz
        print(f"Exporting ONNX from: {best_pt}")
        exporter = YOLO(str(best_pt))
        onnx_out = exporter.export(format="onnx", imgsz=export_imgsz)
        if isinstance(onnx_out, (list, tuple)):
            onnx_path = Path(str(onnx_out[0]))
        else:
            onnx_path = Path(str(onnx_out))
        if not onnx_path.exists():
            raise RuntimeError("ONNX export reported success but output file was not found.")

        output_name = args.export_onnx_name
        if not output_name.lower().endswith(".onnx"):
            output_name += ".onnx"
        final_onnx = onnx_path.parent / output_name
        if final_onnx.resolve() != onnx_path.resolve():
            if final_onnx.exists():
                final_onnx.unlink()
            shutil.move(str(onnx_path), str(final_onnx))
        print(f"ONNX exported to: {final_onnx.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
