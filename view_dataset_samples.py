"""预览 JSONL 的前几条记录，只使用 Python 标准库。"""

import argparse
import json
from itertools import islice
from pathlib import Path


def load_jsonl_samples(dataset_path, num_samples=5):
    with open(dataset_path, encoding="utf-8") as stream:
        samples = []
        for number, line in enumerate(islice(stream, num_samples), start=1):
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{dataset_path} 第 {number} 行不是有效的 JSON：{error.msg}") from error
        return samples


def print_samples(samples):
    for index, sample in enumerate(samples, start=1):
        print(f"\n=== 样本 {index} ===")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预览 dataset 目录中的 JSONL 文件，也支持绝对路径")
    parser.add_argument("--file", default="examples/pretrain.jsonl", help="JSONL 文件路径")
    parser.add_argument("--num", type=int, default=5, help="最多显示几条样本")
    args = parser.parse_args()
    if args.num < 0:
        parser.error("--num 需要大于或等于 0")
    path = Path(__file__).resolve().parent / "dataset" / args.file
    try:
        print_samples(load_jsonl_samples(path, args.num))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
