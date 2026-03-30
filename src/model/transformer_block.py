import torch 
import torch.nn as nn  # Nevrale nettverkslag

# Transformer block, feed-forward og layer norm

from src.model.attention import MultiHeadAttention  # Importerer multi-head attention


# feed-forward nettverk for blokk
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_mult, dropout):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        self.net = nn.Sequential(  # Bygger et lite MLP-nettverk
            nn.Linear(embed_dim, ff_mult * embed_dim),  # Utvider dimensjonen (f.eks 128 -> 512 hvis ff_mult=4)
            nn.ReLU(),  # Ikke-linearitet
            nn.Linear(ff_mult * embed_dim, embed_dim),  # Prosjiserer tilbake til embed_dim
            nn.Dropout(dropout),  # Dropout (ingen effekt hvis dropout=0.0)
        )

    def forward(self, x):
        return self.net(x)  # (B, T, embed_dim)


# Transformer blokk
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, ff_mult, dropout):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        self.attn = MultiHeadAttention(
            embed_dim, num_heads, block_size, dropout
        )  # Multi-head attention med dropout

        self.ffwd = FeedForward(
            embed_dim, ff_mult, dropout
        )  # Feed-forward nettverk med ff_mult og dropout

        self.ln1 = nn.LayerNorm(embed_dim)  # LayerNorm før attention
        self.ln2 = nn.LayerNorm(embed_dim)  # LayerNorm før feed-forward

        self.dropout = nn.Dropout(dropout)  # Dropout etter attention-output

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))  # Residual + attention (pre-layernorm)

        x = x + self.ffwd(self.ln2(x))  # Residual + feed-forward (pre-layernorm)

        return x  # Returnerer oppdatert representasjon