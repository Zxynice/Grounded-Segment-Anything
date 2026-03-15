# 松潘古城遗产要素识别 — 模型微调指南
# Songpan Ancient City Heritage Recognition — Fine-tuning Guide

本目录包含基于 **Grounded-SAM**（GroundingDINO + SAM）的微调流水线，
专为**松潘古城遗产要素**（寺庙、古城墙、城门、雕塑、佛塔、古街、寺院、石拱桥）的
"文本提示 + 图片分割"识别研究而设计。

> This directory contains a fine-tuning pipeline for **Grounded-SAM**
> (GroundingDINO + SAM) targeting heritage element detection and segmentation
> in **Songpan Ancient City** (temples, city walls, gates, sculptures, pagodas,
> ancient streets, monasteries, and stone arch bridges).

---

## 目录结构 / Directory structure

```
finetune/
├── prepare_dataset.py      # 数据集准备：格式转换 & 划分
├── songpan_dataset.py      # PyTorch Dataset（COCO 格式输入）
├── train_grounding_dino.py # GroundingDINO 微调脚本
├── inference_songpan.py    # 推理脚本（微调后模型 + SAM）
└── README.md               # 本文档
```

---

## 遗产要素类别 / Heritage categories

| ID | 英文名称            | 中文名称   |
|----|---------------------|----------|
| 1  | temple              | 寺庙      |
| 2  | ancient city wall   | 古城墙    |
| 3  | city gate           | 城门      |
| 4  | sculpture           | 雕塑      |
| 5  | pagoda              | 佛塔      |
| 6  | ancient street      | 古街      |
| 7  | monastery           | 寺院      |
| 8  | stone arch bridge   | 石拱桥    |

---

## 快速开始 / Quick Start

### 0. 环境准备 / Prerequisites

按照项目根目录的 `README.md` 安装主依赖。额外需要：

```bash
pip install scipy
```

下载预训练权重（如果尚未下载）：
```bash
# GroundingDINO (SwinT)
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O weights/groundingdino_swint_ogc.pth

# SAM ViT-H
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O weights/sam_vit_h_4b8939.pth
```

---

### 1. 准备数据集 / Prepare the dataset

将您已有的松潘古城数据集转换为 COCO JSON 格式。

#### 从 LabelMe 格式转换

```bash
python finetune/prepare_dataset.py \
    --input_format labelme \
    --input_dir    data/songpan/labelme_annotations \
    --image_dir    data/songpan/images \
    --output       data/songpan/coco_annotations.json \
    --split        0.8        # 80% 训练 / 20% 验证
```

输出：
- `data/songpan/coco_annotations_train.json`
- `data/songpan/coco_annotations_val.json`

#### 从 Pascal VOC (XML) 格式转换

```bash
python finetune/prepare_dataset.py \
    --input_format voc \
    --input_dir    data/songpan/voc_annotations \
    --image_dir    data/songpan/images \
    --output       data/songpan/coco_annotations.json \
    --split        0.8
```

#### 验证现有 COCO JSON

```bash
python finetune/prepare_dataset.py \
    --input_format coco \
    --input_dir    data/songpan/my_coco.json \
    --image_dir    data/songpan/images
```

**COCO JSON 格式说明 / COCO JSON format:**

```json
{
  "images": [
    {"id": 1, "file_name": "IMG_001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 300, 150],   // [x, y, width, height] 像素坐标
      "area": 45000,
      "segmentation": [],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "temple", "supercategory": "heritage"},
    {"id": 2, "name": "ancient city wall", "supercategory": "heritage"},
    ...
  ]
}
```

---

### 2. 微调 GroundingDINO / Fine-tune GroundingDINO

```bash
python finetune/train_grounding_dino.py \
    --config        GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --pretrained    weights/groundingdino_swint_ogc.pth \
    --train_json    data/songpan/coco_annotations_train.json \
    --val_json      data/songpan/coco_annotations_val.json \
    --image_dir     data/songpan/images \
    --output_dir    outputs/songpan_finetune \
    --epochs        20 \
    --batch_size    2 \
    --lr            1e-5 \
    --freeze_backbone
```

**主要参数说明 / Key arguments:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 20 |
| `--batch_size` | 批大小（显存不足时设为 1） | 2 |
| `--lr` | 学习率 | 1e-5 |
| `--lr_backbone` | 主干网络学习率 | 1e-6 |
| `--freeze_backbone` | 冻结主干，仅训练检测头 | False |
| `--weight_loss_cls` | 分类损失权重 | 1.0 |
| `--weight_loss_bbox` | 边框 L1 损失权重 | 5.0 |
| `--weight_loss_giou` | GIoU 损失权重 | 2.0 |

**训练策略建议 / Training strategy recommendations:**

1. **数据量较少（< 200 张）**：使用 `--freeze_backbone`，只微调检测头和文本-视觉融合模块。
2. **数据量中等（200–1000 张）**：全参数微调，适当降低学习率（`--lr 5e-6`）。
3. **数据量较大（> 1000 张）**：可使用 SwinB 版本（`GroundingDINO_SwinB.py`）。

输出文件：
- `outputs/songpan_finetune/checkpoint_epoch*.pth` — 每轮检查点
- `outputs/songpan_finetune/best_model.pth` — 验证损失最低的检查点
- `outputs/songpan_finetune/final_model.pth` — 训练结束后的最终模型

---

### 3. 推理 / Inference

使用微调后的 GroundingDINO + 原始 SAM 对新图像进行分割：

```bash
# 单张图片
python finetune/inference_songpan.py \
    --config        GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_ckpt outputs/songpan_finetune/best_model.pth \
    --sam_ckpt      weights/sam_vit_h_4b8939.pth \
    --sam_version   vit_h \
    --input         data/songpan/images/test_image.jpg \
    --output_dir    outputs/inference \
    --text_prompt   "temple . ancient city wall . city gate . sculpture"

# 批量处理目录
python finetune/inference_songpan.py \
    --config        GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_ckpt outputs/songpan_finetune/best_model.pth \
    --sam_ckpt      weights/sam_vit_h_4b8939.pth \
    --input         data/songpan/images/ \
    --output_dir    outputs/inference
```

每张图片输出：
- `*_grounded_sam.jpg` — 带分割掩码和检测框的可视化图
- `*_result.json` — 检测结果（框坐标、标签、置信度）
- `inference_summary.json` — 所有图片的汇总结果

---

## 系统架构 / System Architecture

```
输入图像 + 文本提示
       │
       ▼
  GroundingDINO (微调后)
  ┌──────────────────────────────────┐
  │  文本编码器 (BERT)               │
  │  视觉编码器 (Swin Transformer)   │
  │  跨模态融合 (双向注意力)         │
  │  检测头 → 边框 + 类别置信度      │
  └──────────────────────────────────┘
       │ 边框坐标
       ▼
  SAM (Segment Anything Model)
  ┌──────────────────────────────────┐
  │  图像编码器 (ViT)                │
  │  提示编码器 (边框提示)           │
  │  掩码解码器                      │
  └──────────────────────────────────┘
       │
       ▼
  分割掩码 + 类别标签
```

---

## 训练损失说明 / Loss Functions

微调过程使用三项损失，通过匈牙利匹配将预测框与真值框对齐：

| 损失项 | 公式 | 说明 |
|--------|------|------|
| 文本-视觉对齐损失 (Focal) | `FL(p, positive_map)` | 鼓励预测 logit 在对应文本 token 处激活 |
| 边框 L1 损失 | `L1(pred_box, gt_box)` | 监督框坐标回归 |
| GIoU 损失 | `1 - GIoU(pred_box, gt_box)` | 监督框质量，对不重叠情况更鲁棒 |

---

## 常见问题 / FAQ

**Q: 显存不足 (CUDA OOM)**  
A: 减小 `--batch_size 1`，或使用 `--freeze_backbone` 降低显存占用。

**Q: 检测不到目标**  
A: 降低 `--box_threshold`（如 0.20）和 `--text_threshold`（如 0.15）。

**Q: 想添加新类别（如"古井"、"牌坊"）**  
A: 修改 `finetune/prepare_dataset.py` 和 `finetune/songpan_dataset.py` 中的
`SONGPAN_CATEGORIES` 列表，并重新标注数据。

**Q: 如何使用 SAM-HQ 获得更精细的分割？**  
A: 下载 SAM-HQ 权重后，在推理时加上 `--sam_hq_ckpt weights/sam_hq_vit_h.pth`。
