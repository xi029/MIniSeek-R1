# Adapted from MiniMind for the MiniSeek-R1 experiments. See licenses/MiniMind-LICENSE.
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """MiniMind 模型配置，汇总注意力、MoE 与 RoPE 等超参数。"""

    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            # MLA attention parameters
            q_lora_rank: int = None,
            kv_lora_rank: int = None,
            qk_head_dim: int = None,
            qk_rope_head_dim: int = None,
            v_head_dim: int = None,
            attention_bias: bool = False,
            rope_interleave: bool = False,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            n_group: int = 1,
            topk_group: int = 1,
            routed_scaling_factor: float = 1.0,
            first_k_dense_replace: int = 0,
            router_bias_lr: float = 1e-3,
            router_bias_max: float = 1.0,
            select_k_per_group: int = 2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
         # 最大位置编码长度，32768适配YaRN外推后的长序列
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.qk_head_dim = qk_head_dim or (hidden_size // num_attention_heads)
        self.qk_rope_head_dim = qk_rope_head_dim or max(1, self.qk_head_dim // 2)
        self.qk_nope_head_dim = self.qk_head_dim - self.qk_rope_head_dim
        self.v_head_dim = v_head_dim or (hidden_size // num_attention_heads)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank or self.qk_head_dim
        self.rope_interleave = rope_interleave
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        self.n_group = max(1, n_group)
        self.topk_group = max(1, topk_group)
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = max(0, first_k_dense_replace)
        self.router_bias_lr = max(0.0, router_bias_lr)
        self.router_bias_max = max(0.0, router_bias_max)
        self.select_k_per_group = max(1, select_k_per_group)


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """Root Mean Square 标准化，保持张量尺度稳定。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """预计算 RoPE 频率表，支持 YaRN 外推与放缩。"""
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将特征后半部分旋转，配合 RoPE 完成坐标变换。"""
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """经典轴向 RoPE 赋值，适配不交错布局。"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """交错布局 RoPE，匹配 DeepSeek MLA 形态。"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头扩展到所有注意力头，等价于 repeat_interleave。"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class MultiHeadLatentAttention(nn.Module):
    """实现 DeepSeek 风格 MLA，将查询拆分为 pass/rot 两段。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        assert self.num_heads % self.num_key_value_heads == 0
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_head_dim = config.qk_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.qk_head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

    def _shape(self, tensor, bsz, seq_len, num_heads, head_dim):
        return tensor.view(bsz, seq_len, num_heads, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape
        cos, sin = position_embeddings
        cos = cos[:seq_len]
        sin = sin[:seq_len]

        q_states = self._shape(self.q_proj(x), bsz, seq_len, self.num_heads, self.qk_head_dim)
        k_states = self._shape(self.k_proj(x), bsz, seq_len, self.num_key_value_heads, self.qk_head_dim)
        v_states = self._shape(self.v_proj(x), bsz, seq_len, self.num_key_value_heads, self.v_head_dim)

        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pass, k_rot = torch.split(k_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.config.rope_interleave:
            # MLA 可选择交错式 RoPE，减少 reshape 成本
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        q_states = torch.cat((q_pass, q_rot), dim=-1)
        k_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_value is not None:
            # 拼接 KV Cache，实现自回归推理
            past_k, past_v = past_key_value
            k_states = torch.cat([past_k, k_states], dim=1)
            v_states = torch.cat([past_v, v_states], dim=1)
        past_kv = (k_states, v_states) if use_cache else None

        q_states = q_states.transpose(1, 2)
        k_states = repeat_kv(k_states, self.num_key_value_groups).transpose(1, 2)
        v_states = repeat_kv(v_states, self.num_key_value_groups).transpose(1, 2)

        if self.flash and past_key_value is None and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # PyTorch SDPA 快速路径
            attn_output = F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(q_states, k_states.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            causal_mask = torch.triu(
                torch.full((scores.size(-2), scores.size(-1)), float('-inf'), device=scores.device),
                diagonal=1 + scores.size(-1) - scores.size(-2),
            )
            scores = scores + causal_mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q_states)
            scores = self.attn_dropout(scores)
            attn_output = torch.matmul(scores, v_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_output = self.resid_dropout(self.o_proj(attn_output))
        return attn_output, past_kv


class FeedForward(nn.Module):
    """标准 SwiGLU 前馈网络，支持自动扩展中间维度。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # DeepSeek 默认按 hidden_size×8/3 放大，再向 64 对齐
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MiniSeekTopkRouter(nn.Module):
    """无辅助损失路由器，利用动态偏置确保专家负载均衡。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.top_k = min(config.num_experts_per_tok, self.n_routed_experts)
        self.n_group = max(1, min(config.n_group, self.n_routed_experts))
        if self.n_routed_experts % self.n_group != 0:
            self.n_group = 1
        self.group_size = self.n_routed_experts // self.n_group
        self.topk_group = max(1, min(config.topk_group, self.n_group))
        self.select_k_per_group = min(config.select_k_per_group, self.group_size)
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.bias_lr = config.router_bias_lr
        self.bias_max = config.router_bias_max
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def _update_bias(self, topk_idx: torch.Tensor) -> None:
        if not self.training or self.bias_lr == 0:
            return
        flat_idx = topk_idx.reshape(-1)
        if flat_idx.numel() == 0:
            return
        # 统计每个专家的命中次数，和平均值比较后微调偏置
        counts = torch.bincount(flat_idx, minlength=self.n_routed_experts).float()
        avg = counts.mean()
        delta = torch.zeros_like(self.e_score_correction_bias)
        delta[counts > avg] = -self.bias_lr
        delta[counts < avg] = self.bias_lr
        self.e_score_correction_bias.add_(delta)
        self.e_score_correction_bias.clamp_(-self.bias_max, self.bias_max)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, h = hidden_states.shape
        flat_states = hidden_states.view(-1, h)
        router_logits = F.linear(flat_states.float(), self.weight.float())
        router_scores = router_logits.sigmoid()
        routing_scores = router_scores + self.e_score_correction_bias  # 偏置仅影响路由排序

        if self.n_group > 1:
            reshaped = routing_scores.view(-1, self.n_group, self.group_size)
            select_k = max(1, min(self.select_k_per_group, self.group_size))
            group_scores = reshaped.topk(select_k, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1.0)
            score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.group_size)
            score_mask = score_mask.reshape(-1, self.n_routed_experts).bool()
            masked_logits = routing_scores.masked_fill(~score_mask, float('-inf'))
        else:
            masked_logits = routing_scores

        topk_idx = torch.topk(masked_logits, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weight = router_scores.gather(1, topk_idx)

        if self.norm_topk_prob:
            # 归一化门控权重，保持输出幅值稳定
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-9)

        topk_weight = topk_weight * self.routed_scaling_factor
        topk_idx = topk_idx.view(bsz, seq_len, self.top_k)
        topk_weight = topk_weight.view(bsz, seq_len, self.top_k)
        self._update_bias(topk_idx)
        return topk_idx, topk_weight


class MOEFeedForward(nn.Module):
    """混合专家 MLP，含路由专家与可选共享专家。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.router = MiniSeekTopkRouter(config)
        self.top_k = self.router.top_k
        self.shared_experts = None
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight = self.router(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练态：把 token 复制 top_k 份后分桶计算
            x = x.repeat_interleave(self.top_k, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=2)
            y = y.view(*orig_shape)
        else:
            # 推理态：按专家排序批量处理，避免重复计算
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.shared_experts is not None:
            shared_out = torch.zeros_like(identity)
            for expert in self.shared_experts:
                shared_out = shared_out + expert(identity)
            y = y + shared_out
        self.aux_loss = self._sequence_aux_loss(topk_idx, x.dtype, x.device)
        return y

    def _sequence_aux_loss(self, topk_idx: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not (self.config.seq_aux and self.config.aux_loss_alpha > 0 and self.training):
            return torch.zeros((), device=device, dtype=dtype)

        # topk_idx: (batch, seq_len, top_k)
        batch, seq_len, top_k = topk_idx.shape
        one_hot = F.one_hot(topk_idx, num_classes=self.config.n_routed_experts).float()
        per_seq_counts = one_hot.sum(dim=2).sum(dim=1)  # (batch, n_experts)，统计序列内调用频次
        seq_token_totals = per_seq_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        seq_probs = per_seq_counts / seq_token_totals
        indicator = (per_seq_counts > 0).float()
        uniform = 1.0 / self.config.n_routed_experts
        loss = (indicator * (seq_probs - uniform).abs()).sum(dim=-1).mean()
        return loss.to(dtype) * self.config.aux_loss_alpha

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.top_k
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """单层 Transformer Block：MLA 注意力 + (MoE) 前馈。"""

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = MultiHeadLatentAttention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """堆叠 MiniMindBlock 并管理 RoPE 缓存的主体模型。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        rope_dim = config.qk_rope_head_dim or (config.hidden_size // config.num_attention_heads)
        if rope_dim % 2 != 0:
            raise ValueError(f"qk_rope_head_dim must be even for RoPE, got {rope_dim}.")
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=rope_dim,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 逐层推进，并保存 KV 缓存供下次使用
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = hidden_states.new_tensor(0.0)
        for layer in self.layers:
            if isinstance(layer.mlp, MOEFeedForward):
                # 累加序列级均衡损失，供上层 Trainer 使用
                aux_loss = aux_loss + layer.mlp.aux_loss.to(hidden_states)

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """带语言建模头的 MiniMind，兼容 Transformers API。"""

    config_class = MiniMindConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
