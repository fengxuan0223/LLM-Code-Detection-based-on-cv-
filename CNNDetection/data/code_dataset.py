import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re
from transformers import RobertaTokenizer
import random

LABEL_MAP = {
    "fake": 1,  # AI生成 = 1
    "real": 0  # 人类写的 = 0
}


def read_code_file(path, max_len=2000):
    """安全读取代码文件"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        code = ""
    return code[:max_len]


class FakeCodeDataset(Dataset):
    """假数据集，用于快速测试"""

    def __init__(self, dataroot, split='train', num_samples=1000, emb_dim=128):
        super().__init__()
        self.split = split
        self.num_samples = num_samples
        self.emb_dim = emb_dim

        print(f">>> Using FakeCodeDataset ({split})")

        rng = np.random.RandomState(42 if split == 'train' else 123)
        self.embeddings = rng.randn(num_samples, emb_dim).astype(np.float32)
        self.labels = rng.randint(0, 2, size=(num_samples,)).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.embeddings[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


class RealCodeDataset(Dataset):
    def __init__(self, dataroot, split="train", max_length=512, max_samples=None):
        """
        dataroot: ./code_dataset
        split: train / val
        max_samples: 每个类别最多取多少样本（None=全部）
        """

        self.samples = []
        self.split = split
        self.max_length = max_length

        root = os.path.join(dataroot, split)
        print(f"\n{'=' * 60}")
        print(f"[DEBUG] 正在扫描文件夹: {root}")
        print(f"[DEBUG] 文件夹存在: {os.path.exists(root)}")
        print(f"[DEBUG] max_samples设置: {max_samples}")
        print(f"[DEBUG] split类型: {split}")  # ← 新增
        print(f"{'=' * 60}\n")

        # ✅ 检查根目录是否存在
        if not os.path.exists(root):
            raise FileNotFoundError(f"数据集路径不存在: {root}")

        # ✅ tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "./pretrained/codebert-base",
            local_files_only=True
        )

        # ✅ 统计信息
        total_files_found = 0
        total_files_loaded = 0

        # ✅ 遍历所有子文件夹
        for label_name in os.listdir(root):
            label_dir = os.path.join(root, label_name)

            # 跳过非文件夹
            if not os.path.isdir(label_dir):
                print(f"[SKIP] {label_name} 不是文件夹")
                continue

            # 跳过不在LABEL_MAP中的文件夹
            if label_name not in LABEL_MAP:
                print(f"[SKIP] {label_name} 不在标签映射中")
                continue

            label = LABEL_MAP[label_name]

            print(f"\n[扫描] {label_name}/ (标签={label})")
            print(f"  路径: {label_dir}")

            # ✅ 收集该类别的所有.py文件
            try:
                all_files = os.listdir(label_dir)
                py_files = [
                    os.path.join(label_dir, fname)
                    for fname in all_files
                    if fname.endswith(".py")
                ]
            except Exception as e:
                print(f"  [ERROR] 读取文件夹失败: {e}")
                continue

            # ⭐ 关键：对文件列表排序，保证顺序一致
            py_files = sorted(py_files)

            print(f"  找到 .py 文件: {len(py_files)} 个")
            total_files_found += len(py_files)

            # ✅ 如果指定了max_samples，随机采样
            if max_samples is not None and len(py_files) > max_samples:
                # ⭐⭐⭐ 核心：验证集使用固定种子
                if split == 'val':
                    random.seed(42)  # 固定验证集的随机采样
                    print(f"  [种子] 验证集使用固定种子42进行采样")

                py_files = random.sample(py_files, max_samples)

                # ⭐ 恢复随机状态（重要！）
                if split == 'val':
                    random.seed()  # 恢复随机性，避免影响其他随机操作

                print(f"  随机采样后: {len(py_files)} 个")

            # ✅ 添加到样本列表
            for fpath in py_files:
                self.samples.append((fpath, label))

            total_files_loaded += len(py_files)
            print(f"  已加载到内存: {len(py_files)} 个")

        # ✅ 打乱顺序
        # ⭐⭐⭐ 核心：验证集使用固定种子打乱
        if split == 'val':
            random.seed(42)  # 固定验证集的shuffle顺序
            print(f"\n[种子] 验证集使用固定种子42进行shuffle")

        random.shuffle(self.samples)

        # ⭐ 恢复随机状态
        if split == 'val':
            random.seed()

        print(f"\n{'=' * 60}")
        print(f"[总结] {split} 集:")
        print(f"  总共找到文件: {total_files_found}")
        print(f"  实际加载文件: {total_files_loaded}")
        print(f"  最终样本数量: {len(self.samples)}")
        print(f"{'=' * 60}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        code = read_code_file(path, max_len=2000)

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

