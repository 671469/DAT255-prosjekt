import torch
import torch.nn as nn

#Transformer block, feed-forward evt layer norm-kobling

from src.model.attention import MultiHeadAttention

#feed-forward nettverk for blokk
class FeedForward(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

#Enkel blokk
class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, block_size):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, 4, block_size)

        self.ffwd = FeedForward(embed_dim)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.ffwd(self.ln2(x))

        return x
