import torch
import torch.nn as nn
import torch.nn.functional as F

#causual self attention
class SelfAttention(nn.Module):

    def __init__(self, embed_dim, block_size):
        super().__init__()

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # attention score
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)

        # causal mask
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # softmax
        wei = F.softmax(wei, dim=-1)

        # weighted values
        out = wei @ v

        return out
