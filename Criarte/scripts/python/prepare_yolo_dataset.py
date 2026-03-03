#!/usr/bin/env python3
"""Convert COCO detection annotations to a YOLO dataset with train/val split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", type=Path, required=True, help="Path to COCO _annotations.coco.json")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output YOLO dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (0.0-0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument(
        "--include-class",
        action="append",
        default=[],
        help="Class name to include. Repeat for multiple classes. If omitted, all classes are used.",
    )
    return parser.parse_args()


def yolo_box_from_coco(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    return (
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(w / width, 0.0), 1.0),
        min(max(h / height, 0.0), 1.0),
    )


def split_ids(image_ids: list[int], val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    if not image_ids:
        return set(), set()
    image_ids = list(image_ids)
    random.Random(seed).shuffle(image_ids)

    if len(image_ids) == 1 or val_ratio <= 0.0:
        return set(image_ids), set()

    n_val = max(1, int(len(image_ids) * val_ratio))
    n_val = min(n_val, len(image_ids) - 1)
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])
    return train_ids, val_ids


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()

    if not args.coco_json.exists():
        raise FileNotFoundError(f"COCO json not found: {args.coco_json}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not (0.0 <= args.val_ratio <= 0.5):
        raise ValueError("--val-ratio must be between 0.0 and 0.5")

    data = json.loads(args.coco_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_by_id = {int(img["id"]): img for img in images}
    category_lookup = {int(cat["id"]): str(cat["name"]) for cat in categories}
    include_classes = {c.strip() for c in args.include_class if c.strip()}
    if include_classes:
        available = set(category_lookup.values())
        missing = include_classes - available
        if missing:
            raise ValueError(
                f"Requested classes not present in COCO categories: {sorted(missing)}. "
                f"Available: {sorted(available)}"
            )

    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    used_cat_ids: set[int] = set()

    for ann in annotations:
        image_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        cat_name = category_lookup.get(cat_id, f"class_{cat_id}")
        if include_classes and cat_name not in include_classes:
            continue
        if image_id not in image_by_id:
            continue
        ann_by_image[image_id].append(ann)
        used_cat_ids.add(cat_id)

    if not used_cat_ids:
        raise RuntimeError("No annotations found in COCO file.")

    sorted_cat_ids = sorted(used_cat_ids)
    class_map = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    class_names = [category_lookup.get(cat_id, f"class_{cat_id}") for cat_id in sorted_cat_ids]

    existing_image_ids: list[int] = []
    missing_images: list[str] = []
    for image_id, img in image_by_id.items():
        src = args.images_dir / str(img["file_name"])
        if src.exists():
            existing_image_ids.append(image_id)
        else:
            missing_images.append(str(src))

    if not existing_image_ids:
        raise RuntimeError("No source images found for COCO entries.")

    train_ids, val_ids = split_ids(existing_image_ids, args.val_ratio, args.seed)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_to_ids = {"train": train_ids, "val": val_ids}
    for split, ids in split_to_ids.items():
        for image_id in sorted(ids):
            img = image_by_id[image_id]
            src = args.images_dir / str(img["file_name"])
            rel_img = Path(str(img["file_name"]))
            dst_img = args.output_dir / "images" / split / rel_img
            copy_image(src, dst_img)

            label_path = (args.output_dir / "labels" / split / rel_img).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)

            lines: list[str] = []
            width = int(img["width"])
            height = int(img["height"])
            for ann in ann_by_image.get(image_id, []):
                if ann.get("iscrowd", 0):
                    continue
                x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                if w <= 0 or h <= 0:
                    continue
                cls = class_map[int(ann["category_id"])]
                xc, yc, wn, hn = yolo_box_from_coco([x, y, w, h], width, height)
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    val_ref = "images/val" if val_ids else "images/train"
    yaml_lines = [
        f"path: {args.output_dir.resolve()}",
        "train: images/train",
        f"val: {val_ref}",
        "names:",
    ]
    yaml_lines.extend([f"  {i}: {name}" for i, name in enumerate(class_names)])
    (args.output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Prepared dataset at: {args.output_dir.resolve()}")
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    print(f"Classes: {len(class_names)} -> {class_names}")
    if missing_images:
        print(f"Warning: {len(missing_images)} COCO image entries not found on disk.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
