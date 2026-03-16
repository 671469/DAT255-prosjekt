import torch
import torch.nn as nn

from src.model.embeddings import Embeddings
from src.model.transformer_block import TransformerBlock

#ShakespeareGPT in action
class ShakespeareModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, block_size, num_layers):
        super().__init__()

        self.embeddings = Embeddings(vocab_size, embed_dim, block_size)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, block_size) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        x = self.embeddings(x)

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits