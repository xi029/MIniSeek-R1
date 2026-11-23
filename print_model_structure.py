"""查看默认训练配置的完整模型，以及第一层的路由和注意力模块。"""

from model.model_miniseek import MiniMindConfig, MiniMindForCausalLM


if __name__ == "__main__":
    model = MiniMindForCausalLM(MiniMindConfig(use_moe=True))
    print(model)
    first_layer = model.model.layers[0]
    print("\n第一层 · 混合专家")
    print(first_layer.mlp)
    print("\n第一层 · 注意力")
    print(first_layer.self_attn)
