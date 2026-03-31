"""
Inference script: fine-tuned GroundingDINO + SAM for Songpan Ancient City heritage
element detection and segmentation.

Usage:
    python finetune/inference_songpan.py \
        --config          GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_ckpt   outputs/songpan_finetune/best_model.pth \
        --sam_ckpt        weights/sam_vit_h_4b8939.pth \
        --sam_version     vit_h \
        --input           data/songpan/images/test_image.jpg \
        --output_dir      outputs/inference \
        --text_prompt     "temple . ancient city wall . sculpture" \
        --box_threshold   0.30 \
        --text_threshold  0.25

    # Run on an entire directory of images:
    python finetune/inference_songpan.py \
        --config          GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_ckpt   outputs/songpan_finetune/best_model.pth \
        --sam_ckpt        weights/sam_vit_h_4b8939.pth \
        --input           data/songpan/images/ \
        --output_dir      outputs/inference
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---- repo paths ----
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "GroundingDINO"))
sys.path.insert(0, str(_repo / "segment_anything"))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything import SamPredictor, sam_model_registry, sam_hq_model_registry


# ---------------------------------------------------------------------------
# Image loading / model loading
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> Tuple[Image.Image, torch.Tensor]:
    """Load an image and return (PIL image, normalised tensor)."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil = Image.open(image_path).convert("RGB")
    tensor, _ = transform(pil, None)
    return pil, tensor


def load_grounding_model(
    config_path: str,
    checkpoint_path: str,
    bert_base_uncased_path: Optional[str],
    device: str,
) -> torch.nn.Module:
    """Load a GroundingDINO model (original or fine-tuned checkpoint)."""
    args = SLConfig.fromfile(config_path)
    args.device = device
    if bert_base_uncased_path:
        args.bert_base_uncased_path = bert_base_uncased_path

    model = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    load_res = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(f"[load_grounding_model] {load_res}")
    model.eval()
    model.to(device)
    return model


_VALID_SAM_VERSIONS = {"vit_b", "vit_l", "vit_h"}


def load_sam_model(
    sam_version: str,
    sam_checkpoint: Optional[str],
    sam_hq_checkpoint: Optional[str],
    device: str,
) -> SamPredictor:
    """Load SAM or SAM-HQ predictor."""
    if sam_version not in _VALID_SAM_VERSIONS:
        raise ValueError(
            f"Invalid sam_version '{sam_version}'. "
            f"Must be one of: {sorted(_VALID_SAM_VERSIONS)}"
        )
    if sam_hq_checkpoint:
        if sam_version not in sam_hq_model_registry:
            raise ValueError(
                f"SAM-HQ does not support version '{sam_version}'. "
                f"Available: {list(sam_hq_model_registry.keys())}"
            )
        sam = sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint)
    else:
        sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
    sam.to(device)
    return SamPredictor(sam)


# ---------------------------------------------------------------------------
# Grounding DINO inference
# ---------------------------------------------------------------------------

def run_grounding(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> Tuple[torch.Tensor, List[str]]:
    """Run GroundingDINO and return filtered boxes (cxcywh, normalised) and phrases."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]   # (nq, 256)
    boxes  = outputs["pred_boxes"].cpu()[0]              # (nq, 4)

    mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[mask]
    boxes_filt  = boxes[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        + f"({logit.max().item():.2f})"
        for logit in logits_filt
    ]
    return boxes_filt, phrases


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def show_mask(mask: np.ndarray, ax, color: Optional[np.ndarray] = None):
    if color is None:
        color = np.concatenate([np.random.random(3), [0.6]])
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_box(box: np.ndarray, ax, label: str):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label, color="white", fontsize=8,
            bbox=dict(facecolor="green", alpha=0.5, pad=1, edgecolor="none"))


# ---------------------------------------------------------------------------
# Per-image inference pipeline
# ---------------------------------------------------------------------------

def process_image(
    image_path: str,
    grounding_model: torch.nn.Module,
    sam_predictor: SamPredictor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
    output_dir: str,
) -> dict:
    """
    Run GroundingDINO + SAM on one image and write outputs.

    Returns a dict with keys: image_file, boxes (xyxy), labels, output_image.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    # Load image
    image_pil, image_tensor = load_image(image_path)
    W, H = image_pil.size  # PIL: (width, height)

    # ----- GroundingDINO: detect boxes -----
    boxes_norm, phrases = run_grounding(
        grounding_model, image_tensor, caption, box_threshold, text_threshold, device
    )

    if boxes_norm.numel() == 0:
        print(f"  [INFO] No detections for {Path(image_path).name}")
        image_pil.save(os.path.join(output_dir, f"{stem}_grounded_sam.jpg"))
        return {
            "image_file": image_path,
            "caption": caption,
            "boxes_xyxy": [],
            "labels": [],
        }

    # Convert from normalised cxcywh -> pixel xyxy
    boxes_px = boxes_norm.clone()
    boxes_px[:, 0] *= W
    boxes_px[:, 1] *= H
    boxes_px[:, 2] *= W
    boxes_px[:, 3] *= H
    # cxcywh -> xyxy
    boxes_xyxy = boxes_px.clone()
    boxes_xyxy[:, 0] = boxes_px[:, 0] - boxes_px[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_px[:, 1] - boxes_px[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_px[:, 0] + boxes_px[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_px[:, 1] + boxes_px[:, 3] / 2

    # ----- SAM: segment from boxes -----
    image_cv2 = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_rgb.shape[:2]
    ).to(device)

    with torch.no_grad():
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    # ----- Visualise -----
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.axis("off")
    for mask in masks:
        show_mask(mask.cpu().numpy(), ax)
    for box, label in zip(boxes_xyxy.numpy(), phrases):
        show_box(box, ax, label)
    plt.title(caption, fontsize=10)
    plt.tight_layout()
    out_img_path = os.path.join(output_dir, f"{stem}_grounded_sam.jpg")
    plt.savefig(out_img_path, bbox_inches="tight", dpi=150, pad_inches=0.0)
    plt.close(fig)

    # ----- Save metadata JSON -----
    result = {
        "image_file": image_path,
        "caption": caption,
        "boxes_xyxy": boxes_xyxy.numpy().tolist(),
        "labels": phrases,
    }
    with open(os.path.join(output_dir, f"{stem}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  → {len(phrases)} object(s) detected: {phrases}")
    print(f"     saved: {out_img_path}")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference: fine-tuned GroundingDINO + SAM on Songpan heritage images."
    )
    # Model args
    parser.add_argument("--config",    required=True, help="GroundingDINO config .py")
    parser.add_argument("--grounded_ckpt", required=True,
                        help="Fine-tuned (or original) GroundingDINO checkpoint.")
    parser.add_argument("--sam_version",   default="vit_h",
                        help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_ckpt",  default=None, help="SAM checkpoint path.")
    parser.add_argument("--sam_hq_ckpt", default=None, help="SAM-HQ checkpoint path.")
    parser.add_argument("--bert_base_uncased_path", default=None)
    # Input / output
    parser.add_argument(
        "--input", required=True,
        help="Path to a single image file OR a directory of images."
    )
    parser.add_argument("--output_dir", default="outputs/inference")
    # Detection parameters
    parser.add_argument(
        "--text_prompt",
        default="temple . ancient city wall . city gate . sculpture . pagoda . "
                "ancient street . monastery . stone arch bridge",
        help="Text categories to detect, separated by ' . '",
    )
    parser.add_argument("--box_threshold",  type=float, default=0.30)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Gather image paths
    input_path = Path(args.input)
    if input_path.is_dir():
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_paths = sorted(
            p for ext in exts for p in input_path.glob(ext)
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in directory: {args.input}")
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"[inference_songpan] Processing {len(image_paths)} image(s)...")

    # Load models
    grounding_model = load_grounding_model(
        args.config, args.grounded_ckpt, args.bert_base_uncased_path, args.device
    )
    sam_predictor = load_sam_model(
        args.sam_version, args.sam_ckpt, args.sam_hq_ckpt, args.device
    )

    # Run inference
    all_results = []
    for img_path in image_paths:
        print(f"\n[{img_path.name}]")
        result = process_image(
            str(img_path),
            grounding_model,
            sam_predictor,
            caption=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
            output_dir=args.output_dir,
        )
        all_results.append(result)

    # Save combined summary
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
