# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class SupportQueryAdapter(nn.Module):

    def __init__(self, dim=256, num_heads=8):
        super().__init__()

        self.query_proj = nn.Linear(dim, dim)
        self.support_proj = nn.Linear(dim, dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, queries, support_emb):
        """
        queries: [B, Nq, C]
        support_emb: [B, C] or [B, T, C] (e.g. [B, 4, C] for 4 support tokens)
        """
        if support_emb.dim() == 2:
            support = support_emb.unsqueeze(1)
        else:
            support = support_emb

        q = self.query_proj(queries)
        k = self.support_proj(support)

        delta_q, _ = self.cross_attn(q, k, k)

        if self.training and torch.rand(1, device=queries.device).item() < 0.01:
            print('delta_q_norm:', delta_q.norm(dim=-1).mean().item())

        queries = self.norm(queries + 0.1 * delta_q)

        return queries
