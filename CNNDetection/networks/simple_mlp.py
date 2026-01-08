# # simple_mlp.py
# import torch
# import torch.nn as nn
#
#
# class SimpleMLP(nn.Module):
#     """
#     基线模型：使用Embedding + 平均池化
#     """
#
#     def __init__(self, vocab_size=50265, embed_dim=128, hidden_dim=64):
#         super().__init__()
#
#         # Token IDs → Embeddings
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, input_ids, attention_mask=None):
#         # 1. 获取embeddings: [batch, seq_len, embed_dim]
#         x = self.embedding(input_ids)
#
#         # 2. 平均池化: [batch, seq_len, embed_dim] → [batch, embed_dim]
#         if attention_mask is not None:
#             # 只对有效token求平均
#             mask_expanded = attention_mask.unsqueeze(-1).float()
#             x = (x * mask_expanded).sum(1) / mask_expanded.sum(1)
#         else:
#             x = x.mean(dim=1)
#
#         # 3. 分类: [batch, embed_dim] → [batch, 1]
#         logits = self.classifier(x)
#         return logits.squeeze(-1)  # ✅ 压缩最后一维：[4, 1] → [4]

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, vocab_size=50265, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)      # [B, L, D]
        x = x.mean(dim=1)                  # 简单平均池化
        return torch.sigmoid(self.fc(x)).squeeze()
