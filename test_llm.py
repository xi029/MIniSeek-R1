"""使用本地 .pth 权重进行交互式问答。示例见 README。"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import torch
from transformers import AutoTokenizer


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _normalize_local_path(path: str) -> str:
    """把用户输入路径规范化成绝对路径，避免被误当成 HuggingFace repo_id。"""
    # 展开 ~ / 环境变量，并转为绝对路径
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _resolve_weight_path(save_dir: str, weight: str, hidden_size: int, use_moe: bool) -> str:
    moe_suffix = "_moe" if use_moe else ""
    # 与 trainer_utils.init_model / eval_llm.py 的命名规则保持一致
    cand = os.path.join(save_dir, f"{weight}_{hidden_size}{moe_suffix}.pth")
    if os.path.exists(cand):
        return cand

    raise FileNotFoundError(
        f"没找到权重文件。期望路径：{cand}。\n"
        f"请检查 --save_dir/--weight/--hidden_size/--use_moe 是否与训练一致。"
    )


def build_model(hidden_size: int, num_hidden_layers: int, use_moe: bool, inference_rope_scaling: bool):
    """按训练配置创建 MiniSeek 模型。"""

    from model.model_miniseek import MiniMindConfig, MiniMindForCausalLM


    cfg = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        inference_rope_scaling=inference_rope_scaling,
    )
    return MiniMindForCausalLM(cfg)


def load_pth_weights(model: torch.nn.Module, weight_path: str, device: str, strict: bool = False) -> None:
    sd = torch.load(weight_path, map_location="cpu")
    # 兼容 resume 包里可能是 {'model': state_dict}
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()


def build_inputs(tokenizer, prompt: str, messages: List[Dict[str, str]], enable_thinking: bool):
    # transformers 的 chat template 普遍使用 messages=[{role, content}, ...]
    messages = messages + [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        base_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Reason/GRPO(reasoning) 模型需要 enable_thinking
        if enable_thinking:
            base_kwargs["enable_thinking"] = True

        # transformers 不同版本/不同 tokenizer 的签名不完全一致：
        # - 新版多用 messages=
        # - 有些实现仍要求 conversation=
        try:
            text = tokenizer.apply_chat_template(messages=messages, **base_kwargs)
        except TypeError:
            text = tokenizer.apply_chat_template(conversation=messages, **base_kwargs)
    else:
        text = prompt

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    return enc, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniSeek-R1 GRPO .pth 模型问答测试")

    parser.add_argument("--tokenizer_path", type=str, default="./model", help="tokenizer目录（必须与训练一致）")
    parser.add_argument("--save_dir", type=str, default="./out", help=".pth 权重目录")
    parser.add_argument("--weight", type=str, default="grpo", help="权重前缀（grpo/reason/full_sft/...）")

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=1, choices=[0, 1])
    parser.add_argument("--inference_rope_scaling", action="store_true", help="推理时开启RoPE外推")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--do_sample", type=int, default=1, choices=[0, 1], help="是否采样（0=贪心，1=采样）")
    parser.add_argument("--enable_thinking", action="store_true", help="为推理模型开启 thinking 模板")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1], help="打印调试信息（tokenizer/template片段）")

    args = parser.parse_args()

    setup_seed(args.seed)

    # transformers 对一些形如 "../model" 的输入，在部分版本+异常路径下会走 repo_id 校验。
    # 这里强制把路径转为“存在的本地绝对路径”，并用 local_files_only 走纯本地加载。
    tok_dir = _normalize_local_path(args.tokenizer_path)
    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(
            f"tokenizer_path 不是有效目录：{tok_dir}\n"
            "请把 --tokenizer_path 指向训练时使用的 tokenizer 目录（应包含 tokenizer_config.json / tokenizer.json 或 vocab 等文件）。"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    if args.debug:
        print(f"[debug] tokenizer_path = {tok_dir}")
        print(f"[debug] tokenizer_class = {tokenizer.__class__.__name__}")
        print(f"[debug] has_chat_template = {hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None}")

    model = build_model(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling,
    )

    save_dir = _normalize_local_path(args.save_dir)
    weight_path = _resolve_weight_path(save_dir, args.weight, args.hidden_size, bool(args.use_moe))
    print(f"[info] load weight: {weight_path}")

    load_pth_weights(model, weight_path, device=args.device, strict=True)

    messages: List[Dict[str, str]] = []

    print("\n输入空行直接退出。\n")
    while True:
        prompt = input("👶: ").strip()
        if not prompt:
            break

        enc, next_messages = build_inputs(tokenizer, prompt, messages, enable_thinking=args.enable_thinking)
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)

        if args.debug:
            # 只打印前 300 字符，避免刷屏
            preview_text = tokenizer.decode(input_ids[0][: min(200, input_ids.shape[1])], skip_special_tokens=False)
            print(f"[debug] prompt_token_len = {input_ids.shape[1]}")
            print(f"[debug] input_preview = {preview_text[:300]!r}")

        print("🤖️: ", end="")
        with torch.no_grad():
            generated = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.0,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )

        new_tokens = generated[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(response)

        # 记录到对话历史（用于下一轮模板）
        messages = next_messages + [{"role": "assistant", "content": response}]
        print("\n")


if __name__ == "__main__":
    main()
