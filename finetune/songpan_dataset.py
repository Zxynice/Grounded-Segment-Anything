"""
PyTorch Dataset for GroundingDINO fine-tuning on the Songpan Ancient City
heritage element dataset.

Each item returned by __getitem__ contains:
    image      : torch.Tensor  (3, H, W) – normalised image
    target     : dict
        "boxes"          : FloatTensor (N, 4)  – cx/cy/w/h in [0, 1]
        "labels"         : LongTensor  (N,)    – category ids
        "caption"        : str                 – e.g. "temple . ancient city wall ."
        "tokens_positive": list[list[[int,int]]]  – char spans per box
        "positive_map"   : FloatTensor (N, max_text_len) – token alignment
        "image_id"       : int
        "orig_size"      : tuple (H, W)
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as data
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "GroundingDINO"))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.vl_utils import (
    build_captions_and_token_span,
    create_positive_map_from_span,
)


# ---------------------------------------------------------------------------
# Default Songpan heritage categories (same as prepare_dataset.py)
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


def make_transforms(image_set: str = "train") -> T.Compose:
    """Return GroundingDINO-compatible image transforms."""
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                           max_size=1333),
            normalize,
        ])
    else:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])


class SongpanHeritageDataset(data.Dataset):
    """
    COCO-format dataset for Songpan Ancient City heritage elements.

    Args:
        coco_json_path  : Path to the COCO annotation JSON.
        image_dir       : Root directory that contains the image files.
        tokenizer       : HuggingFace tokenizer (e.g. from GroundingDINO model).
        categories      : List of category dicts; defaults to SONGPAN_CATEGORIES.
        image_set       : "train" or "val" – selects data augmentation strategy.
        max_text_len    : Maximum token length for caption (matches GroundingDINO config).
    """

    def __init__(
        self,
        coco_json_path: str,
        image_dir: str,
        tokenizer,
        categories: Optional[List[Dict]] = None,
        image_set: str = "train",
        max_text_len: int = 256,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.transforms = make_transforms(image_set)

        # Load COCO annotations
        with open(coco_json_path) as f:
            coco = json.load(f)

        # Override categories if provided
        if categories is not None:
            self.categories = categories
        else:
            self.categories = coco.get("categories", SONGPAN_CATEGORIES)

        self.id2name: Dict[int, str] = {
            c["id"]: c["name"].lower() for c in self.categories
        }

        # Build per-image annotation index
        self.images: List[Dict] = coco["images"]
        anns_by_image: Dict[int, List[Dict]] = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            anns_by_image.setdefault(img_id, []).append(ann)

        # Pre-build the shared caption and token spans for ALL categories so
        # that we can quickly look up the character spans per category id.
        all_cat_names = [c["name"].lower() for c in self.categories]
        self.caption, self.cat2tokenspan = build_captions_and_token_span(
            all_cat_names, force_lowercase=True
        )

        # Map category id -> character span list
        self.catid2tokenspan: Dict[int, List[List[int]]] = {}
        for cat in self.categories:
            name = cat["name"].lower()
            if name in self.cat2tokenspan:
                self.catid2tokenspan[cat["id"]] = self.cat2tokenspan[name]

        # Pre-compute the global positive map per category id using the shared caption
        tokenized_caption = self.tokenizer(
            self.caption, return_tensors="pt", padding="longest"
        )
        # shape: (num_categories, max_text_len)
        per_cat_positive = {}
        for cat in self.categories:
            cat_id = cat["id"]
            spans = self.catid2tokenspan.get(cat_id, [])
            pm = create_positive_map_from_span(
                tokenized_caption,
                token_span=[spans],
                max_text_len=self.max_text_len,
            )  # (1, max_text_len)
            per_cat_positive[cat_id] = pm[0]  # (max_text_len,)
        self.per_cat_positive = per_cat_positive

        # Filter out images with no annotations
        self.samples: List[Tuple[Dict, List[Dict]]] = [
            (img, anns_by_image[img["id"]])
            for img in self.images
            if img["id"] in anns_by_image and len(anns_by_image[img["id"]]) > 0
        ]

        print(
            f"[SongpanHeritageDataset] image_set={image_set}, "
            f"samples={len(self.samples)}, categories={len(self.categories)}, "
            f"caption='{self.caption}'"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_info, anns = self.samples[idx]

        # ----- load image -----
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image_pil.size  # PIL returns (width, height)

        # ----- build targets in DETR-format -----
        boxes_xywh = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)  # (N, 4)
        # convert [x, y, w, h] -> [cx, cy, w, h] normalised
        boxes_cxcywh = torch.zeros_like(boxes_xywh)
        boxes_cxcywh[:, 0] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) / orig_w
        boxes_cxcywh[:, 1] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) / orig_h
        boxes_cxcywh[:, 2] = boxes_xywh[:, 2] / orig_w
        boxes_cxcywh[:, 3] = boxes_xywh[:, 3] / orig_h
        boxes_cxcywh = boxes_cxcywh.clamp(0.0, 1.0)

        labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.long)

        # ----- positive map (N, max_text_len) -----
        positive_map = torch.stack(
            [self.per_cat_positive[cat_id.item()] for cat_id in labels], dim=0
        )  # (N, max_text_len)

        # ----- tokens_positive (char spans per box) -----
        tokens_positive = [
            self.catid2tokenspan.get(cat_id.item(), []) for cat_id in labels
        ]

        target = {
            "boxes": boxes_cxcywh,             # (N, 4) normalised cxcywh
            "labels": labels,                   # (N,)
            "caption": self.caption,            # shared caption string
            "tokens_positive": tokens_positive, # list of [[start, end], ...]
            "positive_map": positive_map,       # (N, max_text_len)
            "image_id": img_info["id"],
            "orig_size": (orig_h, orig_w),
        }

        # ----- apply image transforms -----
        # GroundingDINO transforms accept (PIL image, target_dict)
        # and expect target["boxes"] in cxcywh normalised, which they leave unchanged.
        image_tensor, target = self.transforms(image_pil, target)

        return image_tensor, target


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Collate a list of (image_tensor, target) pairs into a batch.
    Images may have different sizes after augmentation, so we keep them as a list.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)
