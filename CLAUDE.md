# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HistGen 是一个基于多重实例学习（MIL）的组织病理报告生成框架，MICCAI 2024 论文。输入 Whole Slide Image (WSI) 通过 DINOv2 ViT-L 预提取的特征，输出诊断报告文本。支持 7 种模型：HistGen（本论文方法）、R2Gen、R2GenCMN、M2Transformer、PlainTransformer、ShowTell、UpDownAttn。

## 环境设置

- **histgen 环境**: `conda activate histgen` — Python 3.10, PyTorch 2.2.1, CUDA 11.8
- **clam 环境**: `conda activate clam` — 仅用于 WSI 预处理和特征提取（CLAM 目录），训练/推理不需要
- 安装: `conda env create -f requirements.yml`

## 常用命令

**训练 HistGen：**
```bash
conda activate histgen
sh train_wsi_report.sh          # 训练 HistGen（主模型）
sh train_wsi_report_baselines.sh # 训练基线模型（r2gen, r2gen_cmn 等）
```

**测试/推理：**
```bash
conda activate histgen
sh test_wsi_report.sh
```

**WSI 预处理（CLAM 流水线，使用 clam 环境）：**
```bash
cd CLAM
conda activate clam
sh patching_scripts/tcga-wsi-report.sh    # WSI 切块
sh extract_scripts/tcga-wsi-report.sh     # 特征提取（支持 ResNet/CTranspath/PLIP/DINOv2）
```

**路径修复工具：**
```bash
python replace_pt_path.py   # 更新 annotation JSON 中的 image_path
```

## 核心架构

### 数据流

```
WSI 原始文件 (.svs)
  → CLAM 切块 + DINOv2 ViT-L 特征提取
    → 预计算的 .pt 特征文件 (dim=1024 或 768, N_patches × dim)
      → wsi_mapping (Linear: 768/1024 → 2048)
        → HATEncoder (层次化注意力编码器)
          → MultiThreadMemory (跨模态记忆交互)
            → Transformer Decoder
              → 生成诊断报告文本
```

### 关键模块

**`modules/histgen_module.py`** — HistGen 核心实现：
- `HATLayer`: 层次化注意力层，将大量 patch 划分为 region（`region_size` 控制），在每个 region 内做 self-attention（region encoder），再跨 region 做 WSI-level attention（WSI encoder）。每个 region 末尾插入 learnable global token。
- `HATEncoder`: 多层 HATLayer 堆叠，输出经过 Attentive Pooling 聚合为固定长度特征。
- `MultiThreadMemory`: 跨模态上下文模块，用 top-k 稀疏注意力实现视觉特征与可学习 memory matrix 之间的交互。
- `BaseHistGen(AttModel)`: 组装编码器-解码器，`encoder_layout` 字典定义每层 HATLayer 是否使用 region_encoder/WSI_encoder。

**`models/histgen_model.py`** — 模型入口：
- `HistGenModel`: 包装 `BaseHistGen`，包含 `wsi_mapping`（将 DINOv2 特征投影到统一维度）和 `forward_pathology`。注意：`visual_extractor` 在 WSI pipeline 中实际不使用（特征已预提取），仅用于兼容。

**`modules/tokenizers.py`** — 三个 tokenizer：
- `Tokenizer`: 基础版，构建词表时按词频 `threshold` 过滤
- `MedicalReportTokenizer`: **当前默认使用**，保留大小写，内置病理学领域词汇（解剖术语、病理学术语、测量值），对数字和测量单位做特殊处理
- `ModernTokenizer`: 基于 PubMedBERT，不常用

**`modules/datasets.py`** — `PathologySingleImageDataset`: 直接加载 `.pt` 特征文件，不需要原始图像

**`modules/trainer_AllinOne.py`** — 训练器：
- `BaseTrainer`: 通用训练循环，支持 checkpoint 恢复、early stopping（监控 `val_BLEU_4`，默认 patience=50）、best model 保存
- `Trainer(BaseTrainer)`: 每个 epoch 执行 train → val → test，生成报告用 beam search 采样

**`modules/metrics.py`** — 评估指标：BLEU-1~4, METEOR, ROUGE_L（基于 `pycocoevalcap`）
**`modules/loss.py`** — 损失函数：语言模型交叉熵（next-token prediction）

### 基线模型架构

基线模型与 HistGen 共用数据管道，但在编码器前增加了 **token selection**（`modules/wsi_token_select.py`）以减少 patch 数量到可承受范围：
- `uniform_sampling`: 均匀采样
- `cross_attention`: 交叉注意力选择
- `kmeans`: K-means 聚类选择

`modules_cmn/` 包含基线模型的原始实现版本（独立的 trainer、dataloader 等），AllinOne 版本的入口统一在 `modules/` 下。

### CLAM 目录

WSI 预处理流水线（来自 Mahmood Lab 的 CLAM 框架的裁剪版本）：
- `create_patches.py` / `create_patches_fp.py`: WSI 切块
- `extract_features.py` / `extract_features_fp.py`: 特征提取
- `models/`: 特征提取器（DINOv2 ViT-L, ResNet, CTranspath, PLIP）
- `patching_scripts/`, `extract_scripts/`: 各自的 shell 脚本
- DINOv2 特征提取器权重需从 HuggingFace 下载放入 `CLAM/models/ckpts/dinov2_cpath_v1.pth`

## 重要约定

- 没有 try-except 异常处理（项目惯例）
- 硬编码和超参数作为全局变量或写入 shell 脚本头部，通过 argparse 传递
- `--dataset_name` 固定为 `wsi_report`，`--image_dir` 指向 DINOv2 特征 `.pt` 文件目录，`--ann_path` 指向 `annotation712_update.json`
- 特征维度：DINOv2 ViT-L 输出 1024 维，`wsi_mapping` 将其映射到 `d_vf=2048`（HATEncoder 的输入维度）；如果特征来自 CTranspath（768 维），映射层会自动调整
- 数据划分：train:val:test = 7:1:2
- `--batch_size` 通常为 1（每个 WSI 有大量 patch，单卡显存限制）
- `--seed` 默认 9233，不同实验应设置不同 seed
- `--monitor_metric` 默认 `BLEU_4`，`--monitor_mode` 为 `max`
- checkpoint 保存到 `--save_dir`，实验记录（CSV）保存到 `--record_dir`（默认 `records/`）
