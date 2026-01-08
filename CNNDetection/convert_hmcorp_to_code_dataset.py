import json
import os
from tqdm import tqdm

# ======================
# 路径配置（关键）
# ======================

SRC_DIR = "dataset/python"     # HMCorp 解压后的目录
DST_DIR = "code_dataset"       # CNNDetection 下的目标目录

SPLITS = {
    "train_no_comment.jsonl": "train",
    "valid_no_comment.jsonl": "val",
    "test_no_comment.jsonl":  "test"
}

LABEL_MAP = {
    "human": "real",
    "llm": "fake",
    0: "real",
    1: "fake"
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


for jsonl_file, split in SPLITS.items():
    src_path = os.path.join(SRC_DIR, jsonl_file)

    print(f"\n>>> Processing {src_path}")

    # 创建目标目录
    for cls in ["real", "fake"]:
        ensure_dir(os.path.join(DST_DIR, split, cls))

    with open(src_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc=f"{split}")):
            item = json.loads(line)

            code = item["code"]

            # ⚠️ 兼容 label / source 两种格式
            if "label" in item:
                label = item["label"]
            else:
                label = item["source"]

            cls = LABEL_MAP[label]

            out_file = os.path.join(
                DST_DIR, split, cls, f"{split}_{idx:06d}.py"
            )

            with open(out_file, "w", encoding="utf-8") as wf:
                wf.write(code)
