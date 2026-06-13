import json
import os

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# 特征归一化: "l2" — UNI 特征范数 (~38) 远大于 ResNet (~2)，需归一化
FEATURE_NORMALIZE = "l2"
CASE_ID_NUM_PARTS = 3


def _clean_data(data):
    """从 splits_0.csv 的 split 列中提取 case_id → full_name 映射"""
    cases = {}
    for idx in range(len(data)):
        case_name = data[idx]
        case_id = '-'.join(case_name.split('-')[:CASE_ID_NUM_PARTS])
        cases[case_id] = case_name
    return cases


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.split_path = args.split_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        # 从 splits_0.csv 读取 case_id 映射
        split_data = pd.read_csv(self.split_path).loc[:, self.split].dropna()
        cases = _clean_data(split_data)

        self.examples = []
        root = self.ann_path

        for dir in os.listdir(root):
            dir_path = os.path.join(root, dir)
            if not os.path.isdir(dir_path):
                continue
            if dir not in cases:
                continue

            # 检查特征文件是否存在
            image_path = os.path.join(self.image_dir, dir)
            if not os.path.exists(image_path + '.pt'):
                continue
            image_path = image_path + '.pt'

            # 读取 annotation 文件
            file_name = os.path.join(root, dir, 'annotation')
            raw = open(file_name, 'r').read()
            anno = raw.strip()
            if anno.startswith('"') and anno.endswith('"'):
                anno = anno[1:-1]

            report_ids = self.tokenizer(anno)[:self.max_seq_length]
            mask = [1] * len(report_ids)

            self.examples.append({
                'id': dir,
                'image_path': image_path,
                'report': anno,
                'split': self.split,
                'ids': report_ids,
                'mask': mask
            })

        print(f'The size of {self.split} dataset: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class PathologySingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = torch.load(image_path)

        # L2 归一化：UNI 特征范数 ~38，需归一化到单位范数
        if FEATURE_NORMALIZE == "l2":
            norms = image.norm(dim=1, keepdim=True).clamp(min=1e-8)
            image = image / norms

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
