"""
Dataset preparation utilities for Songpan Ancient City heritage element fine-tuning.

Supports converting from common annotation formats to COCO JSON, which is required
by the GroundingDINO fine-tuning pipeline.

Supported input formats:
  - LabelMe JSON  (polygons / rectangles)
  - VOC XML       (Pascal VOC bounding-box annotations)
  - COCO JSON     (pass-through / validation)

Usage:
    # LabelMe -> COCO
    python finetune/prepare_dataset.py \
        --input_format labelme \
        --input_dir  data/songpan/labelme_annotations \
        --image_dir  data/songpan/images \
        --output     data/songpan/coco_annotations.json \
        --split      0.8

    # VOC -> COCO
    python finetune/prepare_dataset.py \
        --input_format voc \
        --input_dir  data/songpan/voc_annotations \
        --image_dir  data/songpan/images \
        --output     data/songpan/coco_annotations.json \
        --split      0.8

    # Validate an existing COCO file
    python finetune/prepare_dataset.py \
        --input_format coco \
        --input_dir  data/songpan/coco_annotations.json \
        --image_dir  data/songpan/images
"""

import argparse
import glob
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# Default Songpan heritage categories
# ---------------------------------------------------------------------------
SONGPAN_CATEGORIES = [
    {"id": 1, "name": "temple",            "supercategory": "heritage"},
    {"id": 2, "name": "ancient city wall", "supercategory": "heritage"},
    {"id": 3, "name": "city gate",         "supercategory": "heritage"},
    {"id": 4, "name": "sculpture",         "supercategory": "heritage"},
    {"id": 5, "name": "pagoda",            "supercategory": "heritage"},
    {"id": 6, "name": "ancient street",    "supercategory": "heritage"},
    {"id": 7, "name": "monastery",         "supercategory": "heritage"},
    {"id": 8, "name": "stone arch bridge", "supercategory": "heritage"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_size(image_path: str) -> Tuple[int, int]:
    """Return (width, height) of an image file."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def _polygon_to_bbox(points: List[List[float]]) -> List[float]:
    """Convert a polygon (list of [x, y]) to an axis-aligned bbox [x, y, w, h]."""
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _bbox_area(bbox: List[float]) -> float:
    return bbox[2] * bbox[3]


# ---------------------------------------------------------------------------
# LabelMe converter
# ---------------------------------------------------------------------------

def convert_labelme(
    annotation_dir: str,
    image_dir: str,
    categories: Optional[List[Dict]] = None,
) -> Dict:
    """Convert a directory of LabelMe JSON files to a COCO-format dict."""
    if categories is None:
        categories = SONGPAN_CATEGORIES
    name2id = {c["name"].lower(): c["id"] for c in categories}

    images, annotations = [], []
    img_id = 1
    ann_id = 1

    json_files = sorted(glob.glob(os.path.join(annotation_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(
            f"No LabelMe JSON files found in: {annotation_dir}"
        )

    for jf in json_files:
        with open(jf) as f:
            lm = json.load(f)

        img_filename = lm.get("imagePath", "")
        if not img_filename:
            # Try common extensions when imagePath is absent
            base = os.path.splitext(os.path.basename(jf))[0]
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                candidate = base + ext
                if os.path.exists(os.path.join(image_dir, candidate)):
                    img_filename = candidate
                    break
            if not img_filename:
                img_filename = base + ".jpg"  # last-resort fallback
        img_path = os.path.join(image_dir, img_filename)

        if os.path.exists(img_path):
            w, h = _image_size(img_path)
        else:
            # Fall back to LabelMe embedded size
            w = lm.get("imageWidth", 0)
            h = lm.get("imageHeight", 0)

        images.append({
            "id": img_id,
            "file_name": img_filename,
            "width": w,
            "height": h,
        })

        for shape in lm.get("shapes", []):
            label = shape.get("label", "").lower()
            cat_id = name2id.get(label)
            if cat_id is None:
                print(f"  [WARN] Unknown label '{label}' in {jf} — skipped.")
                continue

            pts = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon")

            if shape_type == "rectangle" and len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                bbox = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
            else:
                bbox = _polygon_to_bbox(pts)

            area = _bbox_area(bbox)
            if area <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "segmentation": [sum(pts, [])],  # flatten [[x,y],...] -> [x,y,...]
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


# ---------------------------------------------------------------------------
# VOC (Pascal VOC) converter
# ---------------------------------------------------------------------------

def convert_voc(
    annotation_dir: str,
    image_dir: str,
    categories: Optional[List[Dict]] = None,
) -> Dict:
    """Convert Pascal-VOC XML files to a COCO-format dict."""
    if categories is None:
        categories = SONGPAN_CATEGORIES
    name2id = {c["name"].lower(): c["id"] for c in categories}

    images, annotations = [], []
    img_id = 1
    ann_id = 1

    xml_files = sorted(glob.glob(os.path.join(annotation_dir, "*.xml")))
    if not xml_files:
        raise FileNotFoundError(
            f"No VOC XML files found in: {annotation_dir}"
        )

    for xf in xml_files:
        tree = ET.parse(xf)
        root = tree.getroot()

        img_filename = root.findtext("filename", default="")
        size_el = root.find("size")
        if size_el is not None:
            w = int(size_el.findtext("width", default="0"))
            h = int(size_el.findtext("height", default="0"))
        else:
            img_path = os.path.join(image_dir, img_filename)
            w, h = _image_size(img_path) if os.path.exists(img_path) else (0, 0)

        images.append({
            "id": img_id,
            "file_name": img_filename,
            "width": w,
            "height": h,
        })

        for obj in root.iter("object"):
            label = (obj.findtext("name") or "").lower()
            cat_id = name2id.get(label)
            if cat_id is None:
                print(f"  [WARN] Unknown label '{label}' in {xf} — skipped.")
                continue

            bndbox = obj.find("bndbox")
            x1 = float(bndbox.findtext("xmin", "0"))
            y1 = float(bndbox.findtext("ymin", "0"))
            x2 = float(bndbox.findtext("xmax", "0"))
            y2 = float(bndbox.findtext("ymax", "0"))
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = _bbox_area(bbox)
            if area <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "segmentation": [],
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


# ---------------------------------------------------------------------------
# COCO pass-through validator
# ---------------------------------------------------------------------------

def validate_coco(coco_json_path: str, image_dir: str) -> Dict:
    """Load and perform basic validation of an existing COCO JSON file."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    required_keys = {"images", "annotations", "categories"}
    missing = required_keys - set(coco.keys())
    if missing:
        raise ValueError(f"COCO JSON is missing keys: {missing}")

    n_images = len(coco["images"])
    n_anns = len(coco["annotations"])
    n_cats = len(coco["categories"])
    print(f"  images: {n_images}, annotations: {n_anns}, categories: {n_cats}")

    missing_files = []
    for img in coco["images"]:
        img_path = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(img_path):
            missing_files.append(img["file_name"])
    if missing_files:
        print(f"  [WARN] {len(missing_files)} image file(s) not found in '{image_dir}'.")
        if len(missing_files) <= 5:
            for f in missing_files:
                print(f"         {f}")

    return coco


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_coco(coco: Dict, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Dict, Dict]:
    """Split a COCO dict into train and val subsets."""
    images = coco["images"][:]
    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    val_anns = [a for a in coco["annotations"] if a["image_id"] in val_ids]

    train_coco = {
        "images": train_images,
        "annotations": train_anns,
        "categories": coco["categories"],
    }
    val_coco = {
        "images": val_images,
        "annotations": val_anns,
        "categories": coco["categories"],
    }
    return train_coco, val_coco


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Songpan heritage dataset for GroundingDINO fine-tuning."
    )
    parser.add_argument(
        "--input_format",
        choices=["labelme", "voc", "coco"],
        required=True,
        help="Source annotation format.",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Directory containing annotation files (labelme/voc) "
            "or path to an existing COCO JSON file."
        ),
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing the image files.",
    )
    parser.add_argument(
        "--output",
        default="data/songpan/coco_annotations.json",
        help="Output path for the combined COCO JSON file.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.0,
        help=(
            "Train/val split ratio (0 < split < 1). "
            "When provided, writes <output>_train.json and <output>_val.json "
            "instead of a single file."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split.",
    )
    args = parser.parse_args()

    print(f"[prepare_dataset] format={args.input_format}")

    if args.input_format == "labelme":
        coco = convert_labelme(args.input_dir, args.image_dir)
    elif args.input_format == "voc":
        coco = convert_voc(args.input_dir, args.image_dir)
    else:  # coco
        coco = validate_coco(args.input_dir, args.image_dir)

    print(
        f"  Total: {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, "
        f"{len(coco['categories'])} categories."
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if 0.0 < args.split < 1.0:
        train_coco, val_coco = split_coco(coco, train_ratio=args.split, seed=args.seed)
        train_out = output_path.parent / (output_path.stem + "_train.json")
        val_out = output_path.parent / (output_path.stem + "_val.json")
        with open(train_out, "w") as f:
            json.dump(train_coco, f)
        with open(val_out, "w") as f:
            json.dump(val_coco, f)
        print(
            f"  Train: {len(train_coco['images'])} images → {train_out}\n"
            f"  Val:   {len(val_coco['images'])}  images → {val_out}"
        )
    else:
        with open(output_path, "w") as f:
            json.dump(coco, f)
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
