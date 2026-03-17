"""
Annotation toolkit for the Songpan Ancient City heritage dataset.

Three modes
-----------
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
     (polygons in LabelMe JSONs are silently converted to their
      enclosing bounding boxes by prepare_dataset.py anyway)

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
"""

import argparse
import glob
import json
import os
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
  多余的 polygon 标注会被 prepare_dataset.py 自动转换为边框后丢弃。

  Reason:
  GroundingDINO is a detector: it predicts and trains on bounding boxes.
  SAM then auto-generates pixel-level masks from those boxes at runtime.
  Polygon annotations are discarded (converted to their enclosing bbox)
  by prepare_dataset.py, so they provide no benefit while costing more
  annotation time.
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

  Step 6  Convert to COCO format and split train/val:
            python finetune/prepare_dataset.py \\
                --input_format labelme \\
                --input_dir    {out.resolve()} \\
                --image_dir    {Path(image_dir).resolve()} \\
                --output       data/songpan/coco_annotations.json \\
                --split        0.8
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

    args = parser.parse_args()

    if args.mode == "setup":
        cmd_setup(args.image_dir, args.output_dir)
    elif args.mode == "stats":
        cmd_stats(args.image_dir, args.ann_dir)
    elif args.mode == "check":
        issues = cmd_check(args.ann_dir, fix=args.fix)
        sys.exit(0 if issues == 0 else 1)


if __name__ == "__main__":
    main()
