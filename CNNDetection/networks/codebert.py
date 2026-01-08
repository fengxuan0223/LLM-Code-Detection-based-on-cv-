import torch
import torch.nn as nn
from transformers import RobertaModel


class CodeBERTClassifier(nn.Module):
    def __init__(
        self,
        model_path="./pretrained/codebert-base",
        hidden_dim=768,
        freeze_encoder=False   # ⭐ 新增
    ):
        super().__init__()

        self.encoder = RobertaModel.from_pretrained(
            model_path,
            local_files_only=True
        )

        # ⭐ 实验 3 核心：是否冻结 CodeBERT
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用 [CLS] token
        cls = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        logits = self.classifier(cls)

        return logits.squeeze(1)
