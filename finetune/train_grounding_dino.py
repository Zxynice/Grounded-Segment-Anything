"""
Fine-tuning script for GroundingDINO on the Songpan Ancient City heritage dataset.

Usage:
    python finetune/train_grounding_dino.py \
        --config          GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --pretrained      weights/groundingdino_swint_ogc.pth \
        --train_json      data/songpan/coco_annotations_train.json \
        --val_json        data/songpan/coco_annotations_val.json \
        --image_dir       data/songpan/images \
        --output_dir      outputs/songpan_finetune \
        --epochs          20 \
        --batch_size      2 \
        --lr              1e-5 \
        --freeze_backbone

Training strategy:
    1. Load a GroundingDINO checkpoint pre-trained on large-scale grounding data.
    2. Fine-tune the full model (or backbone-frozen) on the Songpan dataset.
    3. Use:
         - Focal loss  for text-visual contrastive alignment (pred_logits vs positive_map)
         - L1 loss     for bounding-box regression
         - GIoU loss   for bounding-box quality
    4. Hungarian matching assigns each prediction to the nearest GT box/category.
    5. Periodic checkpoints are saved every ``--save_interval`` epochs (default 10).
       A checkpoint is always written at the final epoch.
    6. The model with the best validation F1-score (or best training loss when no
       validation split is provided) is saved as ``best_model.pth``.
    7. After training, performance metrics (Accuracy, Precision, Recall, F1-score,
       Class Pixel Accuracy) and loss curves are plotted via matplotlib and saved to
       ``--output_dir`` as ``training_curves.png`` and ``detection_metrics.png``.

Outputs (under --output_dir):
    checkpoint_epoch010.pth   – periodic checkpoint (every --save_interval epochs)
    best_model.pth            – best model by val F1 (or train loss if no val set)
    final_model.pth           – last-epoch weights
    training_curves.png       – train / val loss curves
    detection_metrics.png     – Accuracy / Precision / Recall / F1 / Class Pixel Acc
    metrics_history.json      – raw per-epoch numbers
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

# ---- paths ----
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "GroundingDINO"))

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from GroundingDINO.groundingdino.util.misc import nested_tensor_from_tensor_list

from finetune.songpan_dataset import SongpanHeritageDataset, collate_fn, SONGPAN_CATEGORIES


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for text-visual alignment."""
    if num_boxes == 0:
        return inputs.sum() * 0.0
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


# ---------------------------------------------------------------------------
# Box IoU helper (used for detection metrics)
# ---------------------------------------------------------------------------

def _box_iou_diag(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pair IoU between two equal-length sets of boxes in xyxy format.

    Args:
        boxes1 : (k, 4)  xyxy, values in [0, 1]
        boxes2 : (k, 4)  xyxy, values in [0, 1]

    Returns:
        iou : (k,)  true IoU in [0, 1]  (not GIoU)
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """
    Simplified Hungarian matcher for GroundingDINO outputs.

    Matching cost = w_cls * classification_cost
                  + w_bbox * L1_cost
                  + w_giou * GIoU_cost
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,   # (nq, max_text_len)
        pred_boxes: torch.Tensor,    # (nq, 4)  cxcywh normalised
        tgt_positive_map: torch.Tensor,  # (ng, max_text_len)
        tgt_boxes: torch.Tensor,     # (ng, 4)  cxcywh normalised
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_idx : LongTensor (k,)  – selected query indices
            tgt_idx  : LongTensor (k,)  – matched GT indices
        """
        nq = pred_logits.shape[0]
        ng = tgt_boxes.shape[0]

        if ng == 0:
            device = pred_logits.device
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        # Classification cost: negative log-likelihood proxy
        pred_prob = pred_logits.sigmoid()  # (nq, max_text_len)
        # For each GT object, use its positive_map as soft target
        # cost_class[i, j] = -mean( P(token | pred_i) for tokens in tgt_j )
        alpha, gamma = 0.25, 2.0
        neg_cost_class = (1 - alpha) * (pred_prob ** gamma) * (-(1 - pred_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - pred_prob) ** gamma) * (-(pred_prob + 1e-8).log())
        # (nq, max_text_len) x (max_text_len, ng) -> (nq, ng)
        cost_class = (pos_cost_class - neg_cost_class) @ tgt_positive_map.T

        # L1 box cost
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)  # (nq, ng)

        # GIoU cost
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # (nq, 4)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)    # (ng, 4)
        pred_xyxy_clamped = pred_xyxy.clamp(0.0, 1.0)
        tgt_xyxy_clamped = tgt_xyxy.clamp(0.0, 1.0)
        cost_giou = -generalized_box_iou(pred_xyxy_clamped, tgt_xyxy_clamped)  # (nq, ng)

        cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        cost_np = cost.cpu().numpy()
        pred_idx, tgt_idx = linear_sum_assignment(cost_np)
        device = pred_logits.device
        return (
            torch.as_tensor(pred_idx, dtype=torch.long, device=device),
            torch.as_tensor(tgt_idx, dtype=torch.long, device=device),
        )


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict],
    matcher: HungarianMatcher,
    device: torch.device,
    weight_loss_cls: float = 1.0,
    weight_loss_bbox: float = 5.0,
    weight_loss_giou: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the combined Grounding-DINO training loss for one batch.

    Args:
        outputs : dict with "pred_logits" (B, nq, max_text_len)
                             "pred_boxes"  (B, nq, 4)
        targets : list of target dicts (one per image in the batch)
        matcher : HungarianMatcher instance
        device  : torch device

    Returns:
        total_loss : scalar tensor
        loss_dict  : dict with scalar float values for logging
    """
    pred_logits_batch = outputs["pred_logits"]  # (B, nq, max_text_len)
    pred_boxes_batch = outputs["pred_boxes"]    # (B, nq, 4)

    total_cls_loss = torch.tensor(0.0, device=device)
    total_bbox_loss = torch.tensor(0.0, device=device)
    total_giou_loss = torch.tensor(0.0, device=device)
    total_boxes = 0

    for b_idx, target in enumerate(targets):
        pred_logits = pred_logits_batch[b_idx]   # (nq, max_text_len)
        pred_boxes = pred_boxes_batch[b_idx]     # (nq, 4)

        tgt_boxes = target["boxes"].to(device)             # (ng, 4)
        tgt_positive_map = target["positive_map"].to(device)  # (ng, max_text_len)
        ng = tgt_boxes.shape[0]

        if ng == 0:
            continue

        # Clamp predicted logits to max_text_len
        max_text_len = tgt_positive_map.shape[1]
        pred_logits_clamped = pred_logits[:, :max_text_len]

        # Hungarian matching
        pred_idx, tgt_idx = matcher(
            pred_logits_clamped,
            pred_boxes,
            tgt_positive_map,
            tgt_boxes,
        )

        if pred_idx.numel() == 0:
            continue

        matched_pred_logits = pred_logits_clamped[pred_idx]   # (k, max_text_len)
        matched_pred_boxes = pred_boxes[pred_idx]              # (k, 4)
        matched_tgt_pm = tgt_positive_map[tgt_idx]            # (k, max_text_len)
        matched_tgt_boxes = tgt_boxes[tgt_idx]                # (k, 4)
        k = pred_idx.numel()

        # Classification (focal) loss
        # Build full-logit targets: 0 everywhere except matched positive map entries
        cls_targets = torch.zeros_like(pred_logits_clamped)  # (nq, max_text_len)
        cls_targets[pred_idx] = matched_tgt_pm
        cls_loss = sigmoid_focal_loss(pred_logits_clamped, cls_targets, num_boxes=k)
        total_cls_loss += cls_loss

        # L1 box loss
        bbox_loss = F.l1_loss(matched_pred_boxes, matched_tgt_boxes, reduction="sum") / k
        total_bbox_loss += bbox_loss

        # GIoU loss
        pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes).clamp(0.0, 1.0)
        tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt_boxes).clamp(0.0, 1.0)
        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
        giou_loss = (1 - giou.diagonal()).sum() / k
        total_giou_loss += giou_loss

        total_boxes += k

    if total_boxes == 0:
        total_loss = (total_cls_loss + total_bbox_loss + total_giou_loss) * 0.0
        return total_loss, {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "total": 0.0}

    total_loss = (
        weight_loss_cls  * total_cls_loss
        + weight_loss_bbox * total_bbox_loss
        + weight_loss_giou * total_giou_loss
    )

    loss_dict = {
        "loss_cls":  total_cls_loss.item(),
        "loss_bbox": total_bbox_loss.item(),
        "loss_giou": total_giou_loss.item(),
        "total":     total_loss.item(),
    }
    return total_loss, loss_dict


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_finetuning(
    config_path: str,
    checkpoint_path: str,
    bert_base_uncased_path: Optional[str],
    device: str,
    freeze_backbone: bool = False,
) -> nn.Module:
    args = SLConfig.fromfile(config_path)
    args.device = device
    if bert_base_uncased_path:
        args.bert_base_uncased_path = bert_base_uncased_path

    model = build_model(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    load_res = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(f"[load_model] {load_res}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad_(False)
        print("[load_model] backbone frozen.")

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    matcher: HungarianMatcher,
    device: torch.device,
    epoch: int,
    weight_loss_cls: float,
    weight_loss_bbox: float,
    weight_loss_giou: float,
    grad_clip: float = 0.1,
) -> Dict[str, float]:
    model.train()
    running = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "total": 0.0}
    n_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Build NestedTensor for GroundingDINO
        image_tensors = [img.to(device) for img in images]
        samples = nested_tensor_from_tensor_list(image_tensors)

        # Move targets and attach captions for the model forward pass
        captions = [t["caption"] for t in targets]

        # Forward pass (model uses captions from targets via kw["captions"])
        outputs = model(samples, captions=captions)

        loss, loss_dict = compute_loss(
            outputs, targets, matcher, device,
            weight_loss_cls, weight_loss_bbox, weight_loss_giou,
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] += v

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            print(
                f"  epoch {epoch} [{batch_idx+1}/{n_batches}] "
                + "  ".join(f"{k}={v/(batch_idx+1):.4f}" for k, v in running.items())
            )

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    matcher: HungarianMatcher,
    device: torch.device,
    weight_loss_cls: float,
    weight_loss_bbox: float,
    weight_loss_giou: float,
    iou_threshold: float = 0.5,
    num_categories: int = 8,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate the model on *dataloader*.

    Returns:
        loss_dict : averaged losses over all batches
        metrics   : detection metrics dict with keys:
                    accuracy, precision, recall, f1, class_pixel_acc

    Detection metric definitions (at IoU threshold ``iou_threshold``):
        After Hungarian matching of nq predictions to ng GT boxes per image:
          TP = matched pairs whose box IoU >= iou_threshold
          FP = nq - TP   (unmatched predictions + low-IoU matches)
          FN = ng - TP   (unmatched GT boxes   + low-IoU matches)
        Accuracy        = TP / (TP + FP + FN)  [Jaccard / instance-level mIoU]
        Precision       = TP / (TP + FP)
        Recall          = TP / (TP + FN)
        F1              = 2 * P * R / (P + R)
        Class Pixel Acc = mean per-category recall (detection rate averaged over classes)
    """
    model.eval()
    running_loss = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "total": 0.0}
    n_batches = len(dataloader)

    # Metric accumulators
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    # per_class arrays indexed by category_id (1-based; index 0 unused)
    per_class_tp = [0.0] * (num_categories + 1)
    per_class_gt = [0.0] * (num_categories + 1)

    for images, targets in dataloader:
        image_tensors = [img.to(device) for img in images]
        samples = nested_tensor_from_tensor_list(image_tensors)
        captions = [t["caption"] for t in targets]

        outputs = model(samples, captions=captions)

        _, loss_dict = compute_loss(
            outputs, targets, matcher, device,
            weight_loss_cls, weight_loss_bbox, weight_loss_giou,
        )
        for k, v in loss_dict.items():
            running_loss[k] += v

        # ---------- per-image detection metrics ----------
        pred_logits_batch = outputs["pred_logits"]   # (B, nq, max_text_len)
        pred_boxes_batch = outputs["pred_boxes"]     # (B, nq, 4) cxcywh

        for b_idx, target in enumerate(targets):
            pred_logits = pred_logits_batch[b_idx]         # (nq, max_text_len)
            pred_boxes  = pred_boxes_batch[b_idx]          # (nq, 4)
            tgt_boxes        = target["boxes"].to(device)  # (ng, 4) cxcywh
            tgt_positive_map = target["positive_map"].to(device)
            tgt_labels       = target["labels"]            # (ng,) CPU tensor
            ng = tgt_boxes.shape[0]
            nq = pred_boxes.shape[0]

            if ng == 0:
                continue

            # Accumulate GT counts per category
            for cat_id in tgt_labels.tolist():
                if 0 <= cat_id <= num_categories:
                    per_class_gt[cat_id] += 1

            max_text_len = tgt_positive_map.shape[1]
            pred_logits_c = pred_logits[:, :max_text_len]

            pred_idx, tgt_idx = matcher(
                pred_logits_c, pred_boxes,
                tgt_positive_map, tgt_boxes,
            )

            if pred_idx.numel() == 0:
                # No matches: all GT are FN, all predictions are FP
                total_fn += ng
                total_fp += nq
                continue

            k = pred_idx.numel()
            matched_pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[pred_idx]).clamp(0.0, 1.0)
            matched_tgt_xyxy  = box_cxcywh_to_xyxy(tgt_boxes[tgt_idx]).clamp(0.0, 1.0)
            matched_tgt_labels = tgt_labels[tgt_idx.cpu()]  # CPU indexing

            pair_ious = _box_iou_diag(matched_pred_xyxy, matched_tgt_xyxy)
            tp_mask = pair_ious >= iou_threshold
            tp = int(tp_mask.sum().item())

            total_tp += tp
            total_fp += nq - tp   # unmatched predictions + low-IoU matches
            total_fn += ng - tp   # unmatched GT boxes   + low-IoU matches

            for is_tp, cat_id in zip(tp_mask.tolist(), matched_tgt_labels.tolist()):
                if is_tp and 0 <= cat_id <= num_categories:
                    per_class_tp[cat_id] += 1

    # ---- aggregate losses ----
    losses = {k: v / max(n_batches, 1) for k, v in running_loss.items()}

    # ---- aggregate detection metrics ----
    denom_prec = total_tp + total_fp
    denom_rec  = total_tp + total_fn
    denom_acc  = total_tp + total_fp + total_fn

    precision = total_tp / denom_prec if denom_prec > 0 else 0.0
    recall    = total_tp / denom_rec  if denom_rec  > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy  = total_tp / denom_acc if denom_acc > 0 else 0.0

    valid_recalls = [
        per_class_tp[i] / per_class_gt[i]
        for i in range(len(per_class_gt))
        if per_class_gt[i] > 0
    ]
    class_pixel_acc = (sum(valid_recalls) / len(valid_recalls)
                       if valid_recalls else 0.0)

    metrics = {
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "class_pixel_acc": class_pixel_acc,
    }
    return losses, metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List],
    output_dir: str,
) -> None:
    """
    Save training visualisation plots to *output_dir*.

    Generates two files:
        training_curves.png   – total / cls / bbox / GIoU loss per epoch
                                (train and val overlaid when both are available)
        detection_metrics.png – Accuracy / Precision / Recall / F1 / Class Pixel Acc
                                (validation only; skipped when no val set)
    """
    epochs = list(range(1, len(history.get("train_total", [])) + 1))
    if not epochs:
        return

    # ---- Figure 1: Loss curves (2×2 grid) ----
    loss_pairs = [
        ("train_total",     "val_total",     "Total Loss"),
        ("train_loss_cls",  "val_loss_cls",  "Classification Loss"),
        ("train_loss_bbox", "val_loss_bbox", "BBox L1 Loss"),
        ("train_loss_giou", "val_loss_giou", "GIoU Loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (train_key, val_key, title) in zip(axes.flat, loss_pairs):
        if history.get(train_key):
            ax.plot(epochs, history[train_key], label="train",
                    color="tab:blue", linewidth=1.5)
        if history.get(val_key):
            ax.plot(epochs, history[val_key], label="val",
                    color="tab:orange", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training / Validation Loss Curves", fontsize=13)
    fig.tight_layout()
    loss_png = os.path.join(output_dir, "training_curves.png")
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)
    print(f"  → {loss_png}")

    # ---- Figure 2: Detection metrics (validation only) ----
    metric_defs = [
        ("val_accuracy",        "Accuracy",         "tab:blue"),
        ("val_precision",       "Precision",        "tab:orange"),
        ("val_recall",          "Recall",           "tab:green"),
        ("val_f1",              "F1-score",         "tab:red"),
        ("val_class_pixel_acc", "Class Pixel Acc",  "tab:purple"),
    ]
    available = [
        (hk, label, color)
        for hk, label, color in metric_defs
        if history.get(hk)  # non-empty list with at least one value
    ]
    if not available:
        return  # no validation data → skip detection metrics plot

    n = len(available)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for i, (hk, label, color) in enumerate(available):
        ax = axes[i // cols][i % cols]
        vals = history[hk]
        ax.plot(epochs, vals, color=color, linewidth=1.5, marker="o", markersize=3)
        # Vertical dashed line at the best (highest) epoch
        best_idx = int(max(range(len(vals)), key=lambda j: vals[j]))
        ax.axvline(best_idx + 1, color="grey", linestyle="--",
                   linewidth=0.8, alpha=0.7, label=f"best (ep {best_idx + 1})")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    # Hide unused subplot cells
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)
    fig.suptitle(f"Validation Detection Metrics (IoU ≥ threshold)", fontsize=13)
    fig.tight_layout()
    metrics_png = os.path.join(output_dir, "detection_metrics.png")
    fig.savefig(metrics_png, dpi=150)
    plt.close(fig)
    print(f"  → {metrics_png}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GroundingDINO on the Songpan heritage dataset."
    )
    # Model
    parser.add_argument(
        "--config",
        required=True,
        help="Path to GroundingDINO config (e.g. GroundingDINO_SwinT_OGC.py).",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Path to pretrained GroundingDINO checkpoint (.pth).",
    )
    parser.add_argument(
        "--bert_base_uncased_path",
        default=None,
        help="(Optional) Local path to bert-base-uncased weights.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone parameters during fine-tuning.",
    )
    # Data
    parser.add_argument(
        "--train_json",
        required=True,
        help="Path to COCO-format training annotation JSON.",
    )
    parser.add_argument(
        "--val_json",
        default=None,
        help="Path to COCO-format validation annotation JSON.",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Root directory containing image files.",
    )
    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Base learning rate.")
    parser.add_argument("--lr_backbone", type=float, default=1e-6,
                        help="Learning rate for backbone (if not frozen).")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    # Loss weights
    parser.add_argument("--weight_loss_cls",  type=float, default=1.0)
    parser.add_argument("--weight_loss_bbox", type=float, default=5.0)
    parser.add_argument("--weight_loss_giou", type=float, default=2.0)
    # Checkpoint / metrics
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save a periodic checkpoint every N epochs (default: 10). "
             "The best-model and final checkpoints are always saved.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP/FN when computing detection metrics (default: 0.5).",
    )
    # Output
    parser.add_argument("--output_dir", default="outputs/songpan_finetune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    num_categories = len(SONGPAN_CATEGORIES)

    # ---- load model ----
    model = load_model_for_finetuning(
        args.config,
        args.pretrained,
        args.bert_base_uncased_path,
        args.device,
        freeze_backbone=args.freeze_backbone,
    )
    tokenizer = model.tokenizer

    # ---- datasets ----
    train_dataset = SongpanHeritageDataset(
        coco_json_path=args.train_json,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        categories=SONGPAN_CATEGORIES,
        image_set="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if args.val_json:
        val_dataset = SongpanHeritageDataset(
            coco_json_path=args.val_json,
            image_dir=args.image_dir,
            tokenizer=tokenizer,
            categories=SONGPAN_CATEGORIES,
            image_set="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    # ---- optimiser (differential LRs) ----
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    other_params    = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params,    "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    matcher = HungarianMatcher(
        cost_class=args.weight_loss_cls,
        cost_bbox=args.weight_loss_bbox,
        cost_giou=args.weight_loss_giou,
    )

    # ---- per-epoch history (used for plotting and metrics_history.json) ----
    history: Dict[str, List] = {
        "train_total":         [],
        "train_loss_cls":      [],
        "train_loss_bbox":     [],
        "train_loss_giou":     [],
        "val_total":           [],
        "val_loss_cls":        [],
        "val_loss_bbox":       [],
        "val_loss_giou":       [],
        "val_accuracy":        [],
        "val_precision":       [],
        "val_recall":          [],
        "val_f1":              [],
        "val_class_pixel_acc": [],
    }

    # best_score is higher-is-better:
    #   val F1  when a validation set is provided
    #   negative train loss  otherwise
    best_score = -math.inf
    best_epoch = 0

    # ---- training loop ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(
            model, train_loader, optimizer, matcher, device, epoch,
            args.weight_loss_cls, args.weight_loss_bbox, args.weight_loss_giou,
            grad_clip=args.grad_clip,
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch}/{args.epochs}] ({elapsed:.1f}s) "
            + "  ".join(f"train_{k}={v:.4f}" for k, v in train_losses.items())
        )

        # Update train history
        history["train_total"].append(train_losses["total"])
        history["train_loss_cls"].append(train_losses["loss_cls"])
        history["train_loss_bbox"].append(train_losses["loss_bbox"])
        history["train_loss_giou"].append(train_losses["loss_giou"])

        # ---------- validation ----------
        if val_loader is not None:
            val_losses, val_metrics = evaluate(
                model, val_loader, matcher, device,
                args.weight_loss_cls, args.weight_loss_bbox, args.weight_loss_giou,
                iou_threshold=args.iou_threshold,
                num_categories=num_categories,
            )
            history["val_total"].append(val_losses["total"])
            history["val_loss_cls"].append(val_losses["loss_cls"])
            history["val_loss_bbox"].append(val_losses["loss_bbox"])
            history["val_loss_giou"].append(val_losses["loss_giou"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_class_pixel_acc"].append(val_metrics["class_pixel_acc"])

            print(
                f"  val  loss={val_losses['total']:.4f}  "
                f"acc={val_metrics['accuracy']:.4f}  "
                f"prec={val_metrics['precision']:.4f}  "
                f"rec={val_metrics['recall']:.4f}  "
                f"f1={val_metrics['f1']:.4f}  "
                f"cls_px_acc={val_metrics['class_pixel_acc']:.4f}"
            )

            # Best model criterion: highest validation F1
            score = val_metrics["f1"]
        else:
            # No validation set: use negative training loss (lower loss → higher score)
            score = -train_losses["total"]

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
            criterion = ("val_f1" if val_loader is not None
                         else "train_loss(neg)")
            print(f"  → best_model.pth updated "
                  f"(epoch={epoch}, {criterion}={score:.4f})")

        # ---------- periodic checkpoint (every save_interval epochs) ----------
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch{epoch:03d}.pth"
            )
            torch.save(
                {
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "scheduler":    scheduler.state_dict(),
                    "train_losses": train_losses,
                },
                ckpt_path,
            )
            print(f"  checkpoint saved → {ckpt_path}")

    # ---- final model ----
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({"epoch": args.epochs, "model": model.state_dict()}, final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    if val_loader is not None:
        print(
            f"Best model: epoch {best_epoch}, val_f1={best_score:.4f} → "
            f"{os.path.join(args.output_dir, 'best_model.pth')}"
        )
    else:
        print(
            f"Best model: epoch {best_epoch}, train_loss={-best_score:.4f} → "
            f"{os.path.join(args.output_dir, 'best_model.pth')}"
        )

    # ---- save metrics history ----
    history_path = os.path.join(args.output_dir, "metrics_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Metrics history saved to: {history_path}")

    # ---- plot training curves and detection metrics ----
    print("Generating plots...")
    plot_training_curves(history, args.output_dir)


if __name__ == "__main__":
    main()
