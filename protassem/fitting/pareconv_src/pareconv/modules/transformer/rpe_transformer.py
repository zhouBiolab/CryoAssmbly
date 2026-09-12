r"""Transformer with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from IPython import embed

from pareconv.modules.layers import build_dropout_layer
from pareconv.modules.transformer.output_layer import AttentionOutput
class SRPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(SRPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_q1 = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        _init = nn.init.kaiming_normal_
        tensor2 = _init(torch.empty(1, self.num_heads, 1, self.d_model_per_head)).contiguous()
        self.f_shadow = nn.Parameter(tensor2, requires_grad=True)

        self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        q1 = rearrange(self.proj_q1(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

        attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q[:, :, :-1, :], p)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q1, k)
        attention_scores_p = torch.cat([attention_scores_p, torch.zeros_like(attention_scores_p[..., :1])], -1)
        attention_scores_p = torch.cat([attention_scores_p, torch.zeros_like(attention_scores_p[:, :, :1])], -2)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        print("attention_scores ",attention_scores.shape)
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        print("key_masks",key_masks.shape)
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores

class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_q1 = nn.Linear(self.d_model, self.d_model)
        # self.proj_q1 = nn.Linear(self.d_model, 64)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)
        # self.proj_p = nn.Linear(63, 64)

        self.dropout = build_dropout_layer(dropout)

    def compute_distance_mask(self, points, layer_idx, total_layers):
        """
        计算渐进式距离掩码 - 基于最近邻比例
        Args:
            points: torch.Tensor (B, N, 3) - 3D点坐标
            layer_idx: int - 当前层索引 (0-based)
            total_layers: int - 总层数
        Returns:
            mask: torch.Tensor (B, N, N) - 距离掩码
        """
        B, N, _ = points.shape

        # 计算点之间的欧氏距离
        # (B, N, 1, 3) - (B, 1, N, 3) = (B, N, N, 3)
        diff = points.unsqueeze(2) - points.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)  # (B, N, N)

        # 计算当前层应该保留的邻居数量
        # layer_idx + 1 因为我们想要 1/3, 2/3, 3/3 而不是 0/3, 1/3, 2/3
        keep_ratio = (layer_idx + 4) / (total_layers+3)
        #keep_ratio = 1#(layer_idx + 4) / (total_layers+3)
        #print("keep_ratio",keep_ratio)
        k = max(1, int(N * keep_ratio))  # 至少保留1个邻居（自己）
        #k = N
        # 对每个点，找到最近的k个邻居
        # 注意：distances矩阵的对角线是0（点到自己的距离）
        # topk返回的是最小的k个值（包括自己）
        _, indices = torch.topk(distances, k, dim=-1, largest=False)  # (B, N, k)

        # 创建掩码
        mask = torch.zeros(B, N, N, dtype=torch.bool, device=points.device)

        # 使用scatter_来设置mask
        # 为每个点(batch中的每个样本的每个点)设置其k个最近邻为True
        batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(B, N, k)
        row_indices = torch.arange(N, device=points.device).view(1, N, 1).expand(B, N, k)

        mask[batch_indices, row_indices, indices] = True

        return mask

    def forward(self, input_q, input_k, input_v, embed_qk, points=None,
                layer_idx=None, total_layers=None, key_weights=None,
                key_masks=None, attention_factors=None):
        """
        带渐进式注意力的前向传播

        额外参数:
            points: torch.Tensor (B, N, 3) - 用于计算距离掩码的3D点坐标
            layer_idx: int - 当前层索引 (0-based)
            total_layers: int - 总层数
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

        attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        #print("attention_scores",attention_scores.shape)
        # 应用渐进式距离掩码
        if points is not None and layer_idx is not None and total_layers is not None:
            distance_mask = self.compute_distance_mask(points, layer_idx, total_layers)  # (B, N, N)
            # 将False位置设为-inf，这样softmax后会变成0
            attention_scores = attention_scores.masked_fill(~distance_mask.unsqueeze(1), float('-inf'))

        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        points=None,
        layer_idx=None,
        total_layers=None,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            points=points,
            layer_idx=layer_idx,
            total_layers=total_layers,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        points=None,
        layer_idx=None,
        total_layers=None,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            points=points,
            layer_idx=layer_idx,
            total_layers=total_layers,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
