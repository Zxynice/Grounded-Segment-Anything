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
├── annotate.py             # ★ 标注工具：配置 / 进度 / 验证 / 导出 COCO JSON
├── prepare_dataset.py      # 数据集准备：VOC/COCO 格式转换 & 划分
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

## ⚡ 必读：训练所需标注格式 / Required Annotation Format

> **结论：是的，训练脚本要求标注文件为 COCO JSON 格式。**  
> **Answer: YES — the training script requires annotations in COCO JSON format.**

训练脚本 `train_grounding_dino.py` 通过 `--train_json` / `--val_json` 参数接受标注文件，
这两个文件必须是 **COCO JSON 格式**。数据集类 `SongpanHeritageDataset` 直接解析该格式中的
三个顶层字段：

```
{ "images":      [ {"id", "file_name", "width", "height"} ],
  "annotations": [ {"id", "image_id", "category_id",
                    "bbox": [x, y, width, height], "area", "iscrowd"} ],
  "categories":  [ {"id", "name", "supercategory"} ] }
```

> 注意：`bbox` 格式是 **`[x, y,宽, 高]`**（COCO 标准），其中 `x, y` 是左上角像素坐标，
> 不是两个角点坐标。  
> Note: `bbox` uses **`[x, y, width, height]`** (COCO convention) with `x, y` at the top-left corner.

### 使用任意标注工具 / Any annotation tool works

**您不必使用 LabelMe。** 任何能导出 COCO JSON 或能转换为 COCO JSON 的标注工具均可使用：

| 标注工具 | 是否直接输出 COCO JSON | 说明 |
|---------|----------------------|------|
| **CVAT** (cvat.org) | ✓ 原生支持 | 导出时选择 "COCO 1.0" 格式 |
| **Roboflow** (roboflow.com) | ✓ 原生支持 | Export → COCO JSON |
| **makesense.ai** | ✓ 原生支持 | Export Annotations → COCO JSON |
| **COCO Annotator** | ✓ 原生支持 | 本身即 COCO 格式 |
| **LabelImg** | ✗ 输出 VOC XML | 用 `prepare_dataset.py --input_format voc` 转换 |
| **LabelMe** | ✗ 每图一个 JSON | 用 `annotate.py export` 或 `prepare_dataset.py --input_format labelme` 转换 |
| **VGG VIA** | △ 部分版本支持 | 导出 COCO JSON 后直接使用 |
| 其他任意工具 | 只需最终转换为 COCO JSON | 见下方格式说明 |

如果您使用的工具**直接导出 COCO JSON**，只需用 `prepare_dataset.py` 做格式验证和划分：

```bash
# 验证已有 COCO JSON，可选 train/val 划分
python finetune/prepare_dataset.py \
    --input_format coco \
    --input_dir    data/songpan/my_annotations.json \
    --image_dir    data/songpan/images \
    --output       data/songpan/coco_annotations.json \
    --split        0.8
```

如果您使用的工具**不直接支持 COCO JSON**，见 [Step 3](#3-准备数据集其他格式--prepare-dataset-other-formats) 中的转换命令。

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

### 1. 标注数据集 / Annotate your images

> **如果您还未对图片进行标注，请先阅读本节。**  
> **If your images are not yet annotated, read this section first.**
>
> 您可以使用任何标注工具，最终需要一个 COCO JSON 文件。  
> You can use any annotation tool — you only need a COCO JSON file in the end.

#### 用方框还是多边形？/ Bounding box or polygon?

> **结论：使用方框标注（Rectangle / Bounding Box）** ✓  
> **Conclusion: Use rectangle (bounding-box) annotation** ✓

Grounded-SAM 由两个阶段组成：

```
输入图像 + 文本提示
       │
       ▼  阶段 1：GroundingDINO（检测器）
          文本提示 ──→ 预测边框（bounding box）
          训练时仅需边框坐标，不需要多边形轮廓
       │
       ▼  阶段 2：SAM（分割模型）
          边框提示 ──→ 自动生成像素级分割掩码
          SAM 在推理阶段自动完成分割，无需人工标注掩码
```

| 标注方式 | 是否需要 | 原因 |
|---------|---------|------|
| ✓ 矩形边框（方框）| **是** | GroundingDINO 训练的唯一监督信号 |
| ✗ 自由多边形轮廓 | **否** | 在转换为 COCO JSON 时被转换为边框后丢弃，没有额外价值 |

The same two-stage logic explains it in English:  
- **GroundingDINO** is a *detector*: it trains on and predicts bounding boxes only.  
- **SAM** auto-generates polygon-quality masks at inference time from those boxes.  
Therefore, bounding-box annotation is sufficient for the entire pipeline.

#### 选项 A：使用可直接导出 COCO JSON 的工具（推荐）/ Option A: Tools with native COCO JSON export (recommended)

以下工具可以直接导出 COCO JSON，无需额外转换：

- **[CVAT](https://cvat.org)** — 在线/本地都可，导出时选 `COCO 1.0` 格式
- **[Roboflow](https://roboflow.com)** — 云端标注，Export → COCO JSON
- **[makesense.ai](https://www.makesense.ai)** — 免费在线工具，Export → COCO JSON format
- **[COCO Annotator](https://github.com/jsbroks/coco-annotator)** — 本地部署，原生 COCO 格式

导出后用 `prepare_dataset.py` 验证并划分 train/val（见 [Step 3](#3-准备数据集其他格式--prepare-dataset-other-formats)）。

#### 选项 B：使用 LabelMe（需转换）/ Option B: LabelMe (requires conversion)

若您选择 LabelMe，流程如下：

**Step 1** 安装 LabelMe / Install LabelMe:

```bash
pip install labelme
```

**Step 2** 生成配置并查看详细说明 / Generate config and see full instructions:

```bash
python finetune/annotate.py setup \
    --image_dir  data/songpan/images \
    --output_dir data/songpan/annotations
```

**Step 3** 启动 LabelMe / Start LabelMe:

```bash
labelme data/songpan/images \
    --output    data/songpan/annotations \
    --labels    data/songpan/annotations/labels.txt \
    --nodata \
    --autosave
```

在 LabelMe 中请使用 **"Create Rectangle"（快捷键 R）**，不要使用 "Create Polygon"。  
In LabelMe, use **"Create Rectangle" (shortcut: R)** only. Do NOT use "Create Polygon".

**Step 4** 查看标注进度 / Check annotation progress:

```bash
python finetune/annotate.py stats \
    --image_dir  data/songpan/images \
    --ann_dir    data/songpan/annotations
```

**Step 5** 导出为 COCO JSON / Export to COCO JSON:

```bash
python finetune/annotate.py export \
    --ann_dir   data/songpan/annotations \
    --image_dir data/songpan/images \
    --output    data/songpan/coco_annotations.json \
    --split     0.8
```

这一步将 LabelMe 每张图片一个 JSON 的格式合并转换为训练脚本所需的单一 COCO JSON。  
This converts the per-image LabelMe JSONs into the single COCO JSON required for training.
详见 [Step 2](#2-导出-coco-json--export-to-coco-json)。

#### 各类别标注建议 / Per-category annotation tips

| 类别 | 标注建议 |
|------|---------|
| temple 寺庙 | 紧贴屋脊和山门画框 |
| ancient city wall 古城墙 | 框住可见墙体段，含垛口 |
| city gate 城门 | 包含完整城楼和拱洞 |
| sculpture 雕塑 | 含基座/底座一并框入 |
| pagoda 佛塔 | 从塔基到塔尖完整框出 |
| ancient street 古街 | 框住铺装路面（长街可用多个框） |
| monastery 寺院 | 包含正殿和院墙 |
| stone arch bridge 石拱桥 | 含两侧桥拱和完整桥面 |

---

### 2. 导出 COCO JSON / Export to COCO JSON

> **LabelMe 保存的是每张图片一个独立 JSON 文件（LabelMe 格式），不是 COCO JSON。**  
> **LabelMe saves one JSON file per image (LabelMe format), NOT COCO JSON.**

#### 两种格式的区别 / Format differences

| 特性 | LabelMe JSON | COCO JSON |
|------|-------------|-----------|
| 文件数量 | 每张图片一个 `.json` | 所有图片合并为**一个** `.json` |
| 标注结构 | `shapes[].points` 二维坐标点列表 | `annotations[].bbox` `[x, y, w, h]` |
| 边框表示 | 两个角点 `[[x1,y1],[x2,y2]]` | `[左上角x, 左上角y, 宽, 高]` |
| 类别信息 | 仅文字 label 字符串 | 独立 `categories` 数组 + `category_id` |
| 训练支持 | ✗ 训练脚本不直接读取 | ✓ 训练脚本直接读取 |

运行以下命令将 LabelMe JSON 转换为训练所需的 COCO JSON：  
Run the following to convert LabelMe JSONs to the COCO JSON required for training:

```bash
python finetune/annotate.py export \
    --ann_dir   data/songpan/annotations \
    --image_dir data/songpan/images \
    --output    data/songpan/coco_annotations.json \
    --split     0.8          # 80% 训练 / 20% 验证
```

输出 / Output:
- `data/songpan/coco_annotations_train.json`
- `data/songpan/coco_annotations_val.json`

COCO JSON 结构（供参考）/ COCO JSON structure (for reference):

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
      "bbox": [100, 200, 300, 150],
      "area": 45000,
      "segmentation": [],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "temple", "supercategory": "heritage"},
    {"id": 2, "name": "ancient city wall", "supercategory": "heritage"}
  ]
}
```

> **注：** `bbox` 格式为 `[x, y, width, height]`（COCO 标准），
> 其中 `x, y` 是左上角像素坐标。  
> **Note:** `bbox` follows the COCO convention `[x, y, width, height]`
> where `x, y` is the top-left corner in pixel coordinates.

---

### 3. 准备数据集（其他格式）/ Prepare dataset (other formats)

如果您使用的标注工具**已直接导出 COCO JSON**，只需用 `prepare_dataset.py` 验证和划分；
如果使用的是 LabelMe 或 VOC 格式，则需先转换。

#### 工具已直接输出 COCO JSON（CVAT / Roboflow / makesense.ai 等）

```bash
python finetune/prepare_dataset.py \
    --input_format coco \
    --input_dir    data/songpan/my_annotations.json \
    --image_dir    data/songpan/images \
    --output       data/songpan/coco_annotations.json \
    --split        0.8
```

输出：
- `data/songpan/coco_annotations_train.json`
- `data/songpan/coco_annotations_val.json`

#### 从 LabelMe 格式转换（等同于 `annotate.py export`）

```bash
python finetune/prepare_dataset.py \
    --input_format labelme \
    --input_dir    data/songpan/annotations \
    --image_dir    data/songpan/images \
    --output       data/songpan/coco_annotations.json \
    --split        0.8
```

输出：
- `data/songpan/coco_annotations_train.json`
- `data/songpan/coco_annotations_val.json`

#### 从 Pascal VOC (XML) 格式转换（LabelImg 等工具）

```bash
python finetune/prepare_dataset.py \
    --input_format voc \
    --input_dir    data/songpan/voc_annotations \
    --image_dir    data/songpan/images \
    --output       data/songpan/coco_annotations.json \
    --split        0.8
```

---

### 4. 微调 GroundingDINO / Fine-tune GroundingDINO

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
| `--save_interval` | 每隔多少轮保存一次检查点（最终轮总会保存） | 10 |
| `--iou_threshold` | 计算检测指标时的 IoU 阈值 | 0.5 |
| `--weight_loss_cls` | 分类损失权重 | 1.0 |
| `--weight_loss_bbox` | 边框 L1 损失权重 | 5.0 |
| `--weight_loss_giou` | GIoU 损失权重 | 2.0 |

**训练策略建议 / Training strategy recommendations:**

1. **数据量较少（< 200 张）**：使用 `--freeze_backbone`，只微调检测头和文本-视觉融合模块。
2. **数据量中等（200–1000 张）**：全参数微调，适当降低学习率（`--lr 5e-6`）。
3. **数据量较大（> 1000 张）**：可使用 SwinB 版本（`GroundingDINO_SwinB.py`）。

输出文件：

| 文件 | 说明 |
|------|------|
| `checkpoint_epoch010.pth` | 周期性检查点（每 `--save_interval` 轮一次） |
| `best_model.pth` | 验证 F1 最高的模型（无验证集时为训练损失最低的模型） |
| `final_model.pth` | 最后一轮的权重 |
| `training_curves.png` | 训练 / 验证损失曲线（matplotlib） |
| `detection_metrics.png` | Accuracy / Precision / Recall / F1 / Class Pixel Acc 曲线 |
| `metrics_history.json` | 每轮的所有数值，可供后续分析 |

**性能指标定义 / Detection metrics:**

验证集上，对每张图像用匈牙利匹配将预测框与真值框配对，然后按 IoU 阈值统计：

| 指标 | 定义 |
|------|------|
| Accuracy | TP / (TP + FP + FN) — 实例级 Jaccard 系数 |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-score | 2 × Precision × Recall / (Precision + Recall) |
| Class Pixel Acc | 各类别 Recall 的均值（每类检测率的平均） |

其中：TP = 配对 IoU ≥ 阈值的对数，FP = 预测总数 − TP，FN = 真值总数 − TP。

---

### 5. 推理 / Inference

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

**Q: 训练脚本需要的标注格式是 COCO JSON 吗？/ Does training require COCO JSON format?**  
A: **是的。** `train_grounding_dino.py` 的 `--train_json` / `--val_json` 参数只接受
**COCO JSON 格式**的标注文件。您使用什么标注工具不重要，重要的是最终得到一个合法的
COCO JSON 文件（含 `images`、`annotations`、`categories` 三个顶层字段，
`bbox` 格式为 `[x, y, width, height]`）。
各种工具的转换方式见 [必读：训练所需标注格式](#-必读训练所需标注格式--required-annotation-format) 一节。  
A: **Yes.** The `--train_json` / `--val_json` arguments of `train_grounding_dino.py`
accept only **COCO JSON format** annotation files. The annotation tool does not matter —
what matters is producing a valid COCO JSON file with `images`, `annotations`, and
`categories` top-level keys and bboxes in `[x, y, width, height]` format.
See [Required Annotation Format](#-必读训练所需标注格式--required-annotation-format) for tool options.

**Q: LabelMe 保存的 JSON 就是 COCO JSON 格式吗？/ Is the LabelMe JSON the same as COCO JSON?**  
A: **不是**。LabelMe 为每张图片生成一个独立的 `.json` 文件（LabelMe 自有格式），
需要运行 `annotate.py export` 将多张图片的标注合并并转换为训练脚本所需的
单一 COCO JSON 文件。两者的主要区别见上方 [Step 2 格式对比表](#2-导出-coco-json--export-to-coco-json)。  
A: **No.** LabelMe writes one `.json` file per image in LabelMe's own format.
Run `annotate.py export` to merge all per-image files and convert them into a
single COCO JSON file that the training script can read.
See the [Step 2 format table](#2-导出-coco-json--export-to-coco-json) for differences.

**Q: 应该用方框标注还是 polygon 标注？/ Should I annotate with bounding boxes or polygons?**  
A: 用**方框（Rectangle）标注**。GroundingDINO 只用边框做训练监督；SAM 会在推理时自动生成
精细的分割掩码。即使你用 polygon 标注，`annotate.py export` 也会把它转换为边框后才用，
多边形坐标本身不会进入训练。  
A: Use **rectangle (bounding-box)** annotation. GroundingDINO trains on boxes only; SAM
auto-generates fine masks at inference time from those boxes. Any polygon annotations are
automatically converted to their enclosing bbox by `annotate.py export` and the polygon
coordinates are discarded.

**Q: 显存不足 (CUDA OOM)**  
A: 减小 `--batch_size 1`，或使用 `--freeze_backbone` 降低显存占用。

**Q: 检测不到目标**  
A: 降低 `--box_threshold`（如 0.20）和 `--text_threshold`（如 0.15）。

**Q: 想修改检查点保存频率 / How to change checkpoint saving frequency?**  
A: 使用 `--save_interval N`（默认 10），即每 N 轮保存一次。训练结束的最后一轮总会保存，
`best_model.pth`（性能最优）也始终自动更新。  
A: Use `--save_interval N` (default 10). The last epoch is always saved.
`best_model.pth` is always updated when the model improves.

**Q: best_model.pth 是如何选出的？/ How is the best model selected?**  
A: 提供 `--val_json` 时，每轮验证后计算 **val F1-score**，保存历轮 F1 最高的模型。
未提供验证集时，以**训练总损失最低**的轮次作为最优模型。  
A: When `--val_json` is given, the model with the highest **validation F1-score** across all
epochs is saved. Without a validation set, the model with the lowest training loss is saved.

**Q: 想添加新类别（如"古井"、"牌坊"）**  
A: 修改 `finetune/prepare_dataset.py` 和 `finetune/songpan_dataset.py` 中的
`SONGPAN_CATEGORIES` 列表，并重新标注数据。

**Q: 如何使用 SAM-HQ 获得更精细的分割？**  
A: 下载 SAM-HQ 权重后，在推理时加上 `--sam_hq_ckpt weights/sam_hq_vit_h.pth`。
