"""
Annotation toolkit for the Songpan Ancient City heritage dataset.

Modes
-----
setup   Generate a LabelMe labels file and print step-by-step annotation
        instructions.  ONLY rectangle (bounding-box) shapes are listed so
        that annotators use the correct tool.

stats   Show how many images have been annotated vs. the total image count
        in an image directory.

check   Validate completed LabelMe JSON files.  Reports:
          - images with no annotations
          - annotations that used polygon instead of rectangle
          - annotations whose label is not in SONGPAN_CATEGORIES
        Optionally converts accidental polygon annotations to their
        axis-aligned bounding boxes (--fix flag).

export  Convert completed LabelMe per-image JSON files into a single COCO
        JSON file (the format required by the training pipeline).
        Optionally splits into train and val subsets.

        LabelMe saves ONE JSON file per image (LabelMe format).
        The training script requires ONE combined COCO JSON for all images.
        This command performs that conversion.

Format differences
------------------
  LabelMe (per-image JSON, one file per image)
  ┌──────────────────────────────────────────────┐
  │ { "imagePath": "IMG_001.jpg",                │
  │   "imageWidth": 1920, "imageHeight": 1080,   │
  │   "shapes": [                                │
  │     { "label": "temple",                     │
  │       "shape_type": "rectangle",             │
  │       "points": [[100,200],[400,350]] }       │
  │   ] }                                        │
  └──────────────────────────────────────────────┘

  COCO JSON (single file for all images)
  ┌──────────────────────────────────────────────┐
  │ { "images":      [ {id, file_name, w, h} ],  │
  │   "annotations": [ {id, image_id,           │
  │                      category_id,            │
  │                      bbox:[x,y,w,h],         │
  │                      area, segmentation,     │
  │                      iscrowd} ],             │
  │   "categories":  [ {id, name, ...} ] }       │
  └──────────────────────────────────────────────┘

Why bounding boxes, not polygons?
----------------------------------
Grounded-SAM (GroundingDINO + SAM) works as a two-stage pipeline:

  Stage 1 – GroundingDINO:  text prompt  →  bounding boxes
  Stage 2 – SAM:            bounding box →  precise pixel-level mask

GroundingDINO is a *detector*: it only predicts and trains on bounding
boxes, not segmentation polygons.  SAM then converts those boxes into
high-quality, polygon-equivalent masks automatically.

Therefore:
  ✓  Bounding-box (方框) annotation  ← CORRECT choice
  ✗  Free-polygon (自由锚点) annotation  ← unnecessary overhead
     (polygon points are converted to their enclosing bbox on export
      and the polygon coordinates are discarded)

Usage examples
--------------
  # Generate LabelMe labels file and print annotation instructions
  python finetune/annotate.py setup \
      --image_dir data/songpan/images \
      --output_dir data/songpan/annotations

  # Show annotation progress
  python finetune/annotate.py stats \
      --image_dir   data/songpan/images \
      --ann_dir     data/songpan/annotations

  # Validate completed annotations; optionally fix polygon→bbox issues
  python finetune/annotate.py check \
      --ann_dir  data/songpan/annotations \
      --fix

  # Convert LabelMe JSON files → COCO JSON (80/20 train/val split)
  python finetune/annotate.py export \
      --ann_dir   data/songpan/annotations \
      --image_dir data/songpan/images \
      --output    data/songpan/coco_annotations.json \
      --split     0.8
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default Songpan heritage categories (must stay in sync with prepare_dataset.py)
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

# LabelMe labels configuration file content (one label per line)
# This file is passed to LabelMe via --labels so the dropdown is pre-loaded.
_LABELS_CONTENT = "\n".join(c["name"] for c in SONGPAN_CATEGORIES) + "\n"

# Human-readable annotation tips, one per category
_ANNOTATION_TIPS: Dict[str, str] = {
    "temple":            "Draw the box tightly around the roof ridge and entrance gate.",
    "ancient city wall": "Draw the box around the visible wall section; include battlements.",
    "city gate":         "Include the full gate tower and archway opening.",
    "sculpture":         "Box the entire sculpture including its pedestal/base.",
    "pagoda":            "Include the full tower from base to spire.",
    "ancient street":    "Box the paved area; long streets may need multiple boxes.",
    "monastery":         "Include the main hall and courtyard walls.",
    "stone arch bridge": "Include both arch openings and the full span of the deck.",
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_images(image_dir: str) -> List[Path]:
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _iter_annotations(ann_dir: str) -> List[Path]:
    return sorted(Path(ann_dir).glob("*.json"))


def _polygon_to_bbox_points(points: List[List[float]]) -> List[List[float]]:
    """Return the two-point rectangle [top-left, bottom-right] enclosing a polygon."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [[min(xs), min(ys)], [max(xs), max(ys)]]


# ---------------------------------------------------------------------------
# Mode: setup
# ---------------------------------------------------------------------------

def cmd_setup(image_dir: str, output_dir: str) -> None:
    """
    Create output_dir, write the LabelMe labels file, and print annotation
    instructions including the exact LabelMe command to run.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels_file = out / "labels.txt"
    labels_file.write_text(_LABELS_CONTENT, encoding="utf-8")

    images = _iter_images(image_dir)
    n_images = len(images)

    print("=" * 70)
    print("  Songpan Heritage Dataset — Annotation Setup")
    print("=" * 70)
    print(f"\n  Image directory : {Path(image_dir).resolve()}")
    print(f"  Annotation dir  : {out.resolve()}")
    print(f"  Total images    : {n_images}")
    print(f"  Labels file     : {labels_file}")
    print()

    print("─" * 70)
    print("  标注类型选择 / Annotation type choice")
    print("─" * 70)
    print("""
  ✓  使用「方框标注」（Rectangle / Bounding Box）
  ✗  无需「自由锚点 Polygon」标注

  原因说明：
  Grounded-SAM 分为两个阶段——
    阶段 1  GroundingDINO：文本提示 → 预测边框（bounding box）
    阶段 2  SAM：          边框提示  → 自动生成像素级分割掩码

  GroundingDINO 的训练只需要边框坐标，不需要 polygon 轮廓。
  SAM 会在推理阶段根据边框自动生成高质量的 polygon 级别掩码。
  因此，仅标注矩形边框即可满足整个流水线的需求，
  多余的 polygon 标注会被 annotate.py export 自动转换为边框后丢弃。

  Reason:
  GroundingDINO is a detector: it predicts and trains on bounding boxes.
  SAM then auto-generates pixel-level masks from those boxes at runtime.
  Polygon annotations are converted to their enclosing bbox by
  'annotate.py export' and the polygon coordinates are discarded.
""")

    print("─" * 70)
    print("  标注步骤 / Annotation steps")
    print("─" * 70)
    print(f"""
  Step 1  Install LabelMe (if not already installed):
            pip install labelme

  Step 2  Open LabelMe with the pre-configured labels:
            labelme {Path(image_dir).resolve()} \\
                --output {out.resolve()} \\
                --labels {labels_file} \\
                --nodata \\
                --autosave

  Step 3  In LabelMe, use ONLY the "Create Rectangle" tool (shortcut: R)
          to draw bounding boxes.  DO NOT use "Create Polygon".

  Step 4  Select the category from the dropdown that appears after drawing
          each rectangle.

  Step 5  After finishing all images, run:
            python finetune/annotate.py check \\
                --ann_dir {out.resolve()}
          to verify there are no issues.

  Step 6  Convert LabelMe JSONs → COCO JSON (LabelMe format ≠ COCO format):
            python finetune/annotate.py export \\
                --ann_dir   {out.resolve()} \\
                --image_dir {Path(image_dir).resolve()} \\
                --output    data/songpan/coco_annotations.json \\
                --split     0.8
""")

    print("─" * 70)
    print("  类别标注提示 / Per-category annotation tips")
    print("─" * 70)
    for cat in SONGPAN_CATEGORIES:
        tip = _ANNOTATION_TIPS.get(cat["name"], "")
        print(f"  [{cat['id']:2d}] {cat['name']:<22}  {tip}")
    print()
    print("  标注完成后运行 'python finetune/annotate.py stats' 查看进度。")
    print("  Run 'python finetune/annotate.py stats' to check progress.\n")


# ---------------------------------------------------------------------------
# Mode: stats
# ---------------------------------------------------------------------------

def cmd_stats(image_dir: str, ann_dir: str) -> None:
    """Print annotation progress statistics."""
    images = _iter_images(image_dir)
    ann_files = {p.stem: p for p in _iter_annotations(ann_dir)}

    annotated = []
    unannotated = []
    for img in images:
        if img.stem in ann_files:
            annotated.append(img)
        else:
            unannotated.append(img)

    # Count boxes per category
    cat_counts: Dict[str, int] = {c["name"]: 0 for c in SONGPAN_CATEGORIES}
    total_boxes = 0
    polygon_warnings = 0
    valid_labels = {c["name"].lower() for c in SONGPAN_CATEGORIES}

    for img in annotated:
        jf = ann_files[img.stem]
        try:
            with open(jf) as f:
                lm = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for shape in lm.get("shapes", []):
            label = shape.get("label", "").lower()
            if label in valid_labels:
                cat_counts[label] += 1
                total_boxes += 1
            if shape.get("shape_type", "rectangle") != "rectangle":
                polygon_warnings += 1

    pct = 100.0 * len(annotated) / len(images) if images else 0.0

    print("=" * 60)
    print("  Annotation Progress")
    print("=" * 60)
    print(f"  Total images    : {len(images)}")
    print(f"  Annotated       : {len(annotated)}  ({pct:.1f}%)")
    print(f"  Remaining       : {len(unannotated)}")
    print(f"  Total boxes     : {total_boxes}")
    if polygon_warnings:
        print(f"  [WARN] Polygon shapes found: {polygon_warnings}  "
              f"(run 'check --fix' to convert to rectangles)")
    print()
    print("  Boxes per category:")
    for cat in SONGPAN_CATEGORIES:
        count = cat_counts.get(cat["name"], 0)
        bar = "█" * min(count, 40)
        print(f"    {cat['name']:<22}  {count:4d}  {bar}")
    print()

    if unannotated:
        print(f"  Next images to annotate ({min(5, len(unannotated))} shown):")
        for img in unannotated[:5]:
            print(f"    {img.name}")
        if len(unannotated) > 5:
            print(f"    ... and {len(unannotated) - 5} more")
        print()


# ---------------------------------------------------------------------------
# Mode: check
# ---------------------------------------------------------------------------

def cmd_check(ann_dir: str, fix: bool = False) -> int:
    """
    Validate LabelMe annotation files.

    Returns the number of issues found (0 = all clean).
    If fix=True, polygon shapes are converted to their enclosing rectangles
    and the JSON files are updated in-place.
    """
    ann_files = _iter_annotations(ann_dir)
    if not ann_files:
        print(f"[check] No JSON files found in: {ann_dir}")
        return 0

    valid_labels = {c["name"].lower() for c in SONGPAN_CATEGORIES}
    total_issues = 0
    total_fixed = 0

    for jf in ann_files:
        try:
            with open(jf) as f:
                lm = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [ERROR] Cannot read {jf.name}: {exc}")
            total_issues += 1
            continue

        shapes = lm.get("shapes", [])
        file_issues = 0
        file_fixed = 0

        if not shapes:
            print(f"  [WARN]  {jf.name}: no annotations (0 shapes)")
            total_issues += 1
            continue

        new_shapes = []
        for shape in shapes:
            label = shape.get("label", "").lower()
            stype = shape.get("shape_type", "rectangle")
            pts = shape.get("points", [])

            # Check unknown label
            if label not in valid_labels:
                print(
                    f"  [WARN]  {jf.name}: unknown label '{label}' "
                    "(not in SONGPAN_CATEGORIES)"
                )
                total_issues += 1
                file_issues += 1

            # Check / fix non-rectangle shapes
            if stype != "rectangle":
                msg = (
                    f"  [WARN]  {jf.name}: shape '{label}' uses "
                    f"'{stype}' instead of 'rectangle'"
                )
                if fix and len(pts) >= 2:
                    shape = dict(shape)  # shallow copy
                    shape["points"] = _polygon_to_bbox_points(pts)
                    shape["shape_type"] = "rectangle"
                    msg += "  → FIXED (converted to bounding box)"
                    file_fixed += 1
                    total_fixed += 1
                else:
                    total_issues += 1
                    file_issues += 1
                print(msg)

            new_shapes.append(shape)

        if fix and file_fixed > 0:
            lm["shapes"] = new_shapes
            with open(jf, "w") as f:
                json.dump(lm, f, indent=2, ensure_ascii=False)

        if file_issues == 0 and file_fixed == 0:
            pass  # clean file — no output needed

    print()
    print("─" * 60)
    if total_issues == 0 and total_fixed == 0:
        print(f"  ✓  All {len(ann_files)} annotation file(s) look correct.")
        print("     All shapes are rectangles with known category labels.")
    else:
        if total_fixed > 0:
            print(f"  Fixed {total_fixed} polygon shape(s) → rectangle in-place.")
        if total_issues > 0:
            print(f"  {total_issues} issue(s) remain. Review warnings above.")
    print()
    return total_issues


# ---------------------------------------------------------------------------
# Helpers shared by check and export
# ---------------------------------------------------------------------------

def _bbox_area(bbox: List[float]) -> float:
    return bbox[2] * bbox[3]


def _labelme_to_coco(
    ann_dir: str,
    image_dir: str,
    categories: Optional[List[Dict]] = None,
) -> Dict:
    """
    Convert a directory of LabelMe per-image JSON files into a single
    COCO-format dict.

    LabelMe JSON (one file per image) vs COCO JSON (one file for all):
    - LabelMe stores shapes with 'shape_type' + 'points' (rect or polygon)
    - COCO stores a flat list of annotations with 'bbox' [x, y, w, h]
    Rectangle points [[x1,y1],[x2,y2]] → bbox [min_x, min_y, w, h].
    Polygon points   [[x,y],...] are converted to their enclosing bbox.
    """
    if categories is None:
        categories = SONGPAN_CATEGORIES
    name2id = {c["name"].lower(): c["id"] for c in categories}

    images: List[Dict] = []
    annotations: List[Dict] = []
    img_id = 1
    ann_id = 1

    for jf in _iter_annotations(ann_dir):
        try:
            with open(jf) as f:
                lm = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Skipping unreadable file {jf.name}: {exc}")
            continue

        # Resolve image filename
        img_filename = lm.get("imagePath", "")
        if not img_filename:
            base = jf.stem
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                if os.path.exists(os.path.join(image_dir, base + ext)):
                    img_filename = base + ext
                    break
            if not img_filename:
                img_filename = jf.stem + ".jpg"

        # Image dimensions
        img_path = os.path.join(image_dir, img_filename)
        if os.path.exists(img_path):
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(img_path) as _img:
                    w, h = _img.size
            except Exception:
                w = lm.get("imageWidth", 0)
                h = lm.get("imageHeight", 0)
        else:
            w = lm.get("imageWidth", 0)
            h = lm.get("imageHeight", 0)

        images.append({"id": img_id, "file_name": img_filename,
                        "width": w, "height": h})

        for shape in lm.get("shapes", []):
            label = shape.get("label", "").lower()
            cat_id = name2id.get(label)
            if cat_id is None:
                print(f"  [WARN] Unknown label '{label}' in {jf.name} — skipped.")
                continue

            pts = shape.get("points", [])
            stype = shape.get("shape_type", "polygon")

            if stype == "rectangle" and len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                bbox = [min(x1, x2), min(y1, y2),
                        abs(x2 - x1), abs(y2 - y1)]
            elif len(pts) >= 2:
                # polygon or any other multi-point shape → enclosing bbox
                xs, ys = zip(*pts)
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            else:
                continue  # degenerate shape — skip

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

    return {"images": images, "annotations": annotations,
            "categories": categories}


def _split_coco(
    coco: Dict, train_ratio: float, seed: int = 42
) -> Tuple[Dict, Dict]:
    """Return (train_coco, val_coco) by image-level random split."""
    imgs = coco["images"][:]
    random.seed(seed)
    random.shuffle(imgs)
    cut = int(len(imgs) * train_ratio)
    train_imgs, val_imgs = imgs[:cut], imgs[cut:]
    train_ids = {i["id"] for i in train_imgs}
    val_ids   = {i["id"] for i in val_imgs}
    return (
        {"images": train_imgs,
         "annotations": [a for a in coco["annotations"] if a["image_id"] in train_ids],
         "categories": coco["categories"]},
        {"images": val_imgs,
         "annotations": [a for a in coco["annotations"] if a["image_id"] in val_ids],
         "categories": coco["categories"]},
    )


# ---------------------------------------------------------------------------
# Mode: export
# ---------------------------------------------------------------------------

def cmd_export(
    ann_dir: str,
    image_dir: str,
    output: str,
    split: float = 0.0,
    seed: int = 42,
) -> None:
    """
    Convert LabelMe per-image JSON files to a single COCO JSON file.

    LabelMe format (NOT COCO):
      - One .json file per image, stored in ann_dir
      - Each file contains 'shapes' with 'shape_type' and 'points'
      - bbox coordinates stored as two corner points [[x1,y1],[x2,y2]]

    COCO JSON format (required by training pipeline):
      - Single .json file for all images
      - Top-level keys: 'images', 'annotations', 'categories'
      - Each annotation has 'bbox': [x, y, width, height]

    When split > 0, writes <output>_train.json and <output>_val.json.
    """
    print(f"[export] Reading LabelMe annotations from: {ann_dir}")
    coco = _labelme_to_coco(ann_dir, image_dir)

    n_imgs = len(coco["images"])
    n_anns = len(coco["annotations"])
    print(f"  Converted: {n_imgs} images, {n_anns} annotations, "
          f"{len(coco['categories'])} categories.")

    if n_imgs == 0:
        print("  [WARN] No images found — check --ann_dir path.")
        return

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if 0.0 < split < 1.0:
        train_coco, val_coco = _split_coco(coco, train_ratio=split, seed=seed)
        train_out = out.parent / (out.stem + "_train.json")
        val_out   = out.parent / (out.stem + "_val.json")
        with open(train_out, "w") as f:
            json.dump(train_coco, f, ensure_ascii=False)
        with open(val_out, "w") as f:
            json.dump(val_coco, f, ensure_ascii=False)
        print(f"  Train ({len(train_coco['images'])} images) → {train_out}")
        print(f"  Val   ({len(val_coco['images'])} images)   → {val_out}")
        print()
        print("  Next step — fine-tune GroundingDINO:")
        print(f"    python finetune/train_grounding_dino.py \\")
        print(f"        --train_json {train_out} \\")
        print(f"        --val_json   {val_out} \\")
        print(f"        --image_dir  {Path(image_dir).resolve()}")
    else:
        with open(out, "w") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"  Saved: {out}")
        print()
        print("  Next step — fine-tune GroundingDINO:")
        print(f"    python finetune/train_grounding_dino.py \\")
        print(f"        --train_json {out} \\")
        print(f"        --image_dir  {Path(image_dir).resolve()}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="annotate.py",
        description=(
            "Annotation toolkit for the Songpan heritage dataset.\n\n"
            "Use 'setup' before you start annotating to get the correct LabelMe\n"
            "configuration and instructions.  Use 'stats' / 'check' afterwards."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- setup ---
    p_setup = sub.add_parser(
        "setup",
        help="Generate LabelMe labels file and print annotation instructions.",
    )
    p_setup.add_argument(
        "--image_dir",
        required=True,
        help="Directory that contains your JPG images.",
    )
    p_setup.add_argument(
        "--output_dir",
        default="data/songpan/annotations",
        help="Directory where LabelMe JSON files will be saved (default: data/songpan/annotations).",
    )

    # --- stats ---
    p_stats = sub.add_parser(
        "stats",
        help="Show annotation progress statistics.",
    )
    p_stats.add_argument(
        "--image_dir",
        required=True,
        help="Directory that contains your JPG images.",
    )
    p_stats.add_argument(
        "--ann_dir",
        required=True,
        help="Directory that contains LabelMe JSON annotation files.",
    )

    # --- check ---
    p_check = sub.add_parser(
        "check",
        help="Validate annotation files; optionally fix polygon→rectangle.",
    )
    p_check.add_argument(
        "--ann_dir",
        required=True,
        help="Directory that contains LabelMe JSON annotation files.",
    )
    p_check.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Automatically convert polygon shapes to their enclosing bounding\n"
            "boxes and update the JSON files in-place."
        ),
    )

    # --- export ---
    p_export = sub.add_parser(
        "export",
        help=(
            "Convert LabelMe per-image JSONs → single COCO JSON. "
            "LabelMe format ≠ COCO format; this step is required before training."
        ),
    )
    p_export.add_argument(
        "--ann_dir",
        required=True,
        help="Directory containing LabelMe JSON annotation files (one per image).",
    )
    p_export.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing the image files (used to read image dimensions).",
    )
    p_export.add_argument(
        "--output",
        default="data/songpan/coco_annotations.json",
        help=(
            "Output COCO JSON file path. "
            "When --split is provided, writes <output>_train.json and <output>_val.json "
            "(default: data/songpan/coco_annotations.json)."
        ),
    )
    p_export.add_argument(
        "--split",
        type=float,
        default=0.0,
        help=(
            "Train/val split ratio, e.g. 0.8 → 80%% train / 20%% val. "
            "When provided, writes two files instead of one (default: 0, no split)."
        ),
    )
    p_export.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split (default: 42).",
    )

    args = parser.parse_args()

    if args.mode == "setup":
        cmd_setup(args.image_dir, args.output_dir)
    elif args.mode == "stats":
        cmd_stats(args.image_dir, args.ann_dir)
    elif args.mode == "check":
        issues = cmd_check(args.ann_dir, fix=args.fix)
        sys.exit(0 if issues == 0 else 1)
    elif args.mode == "export":
        cmd_export(args.ann_dir, args.image_dir, args.output,
                   split=args.split, seed=args.seed)


if __name__ == "__main__":
    main()
