import torch
import torch.nn as nn
import torch.nn.functional as F

#self-attention, multi-head attention

#ett "attention-head"
class Head(nn.Module):

    def __init__(self, embed_dim, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)

        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)

        v = self.value(x)

        out = wei @ v

        return out

#flere "attention-heads"
class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()

        head_size = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size, block_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)
