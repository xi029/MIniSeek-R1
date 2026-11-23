from model.model_miniseek import MiniMindConfig, MiniMindForCausalLM


if __name__ == "__main__":
    model = MiniMindForCausalLM(MiniMindConfig(use_moe=True))
    count = sum(p.numel() for p in model.parameters())
    print(f"MiniSeek-R1 · 512 维 / 8 层 / MoE · {count:,} 个参数")
