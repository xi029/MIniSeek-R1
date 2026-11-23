"""导出 MiniSeek 原生权重，并保留自定义注意力和 MoE 路由。"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.model_miniseek import MiniMindConfig, MiniMindForCausalLM


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)


def main():
    parser = argparse.ArgumentParser(description="将 MiniSeek 权重和分词器导出为 Transformers 目录")
    parser.add_argument("--input_model", type=Path, default=Path("out/full_sft_512_moe.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("exports/full_sft"))
    parser.add_argument("--tokenizer_path", type=Path, default=Path("model"))
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                            use_moe=bool(args.use_moe))
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(args.input_model, map_location="cpu", weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model.half().save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"已导出模型和分词器到 {args.output_dir}")


if __name__ == "__main__":
    main()
