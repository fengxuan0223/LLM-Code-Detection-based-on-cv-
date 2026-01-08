import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size=50265, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return torch.sigmoid(self.fc(h)).squeeze()
