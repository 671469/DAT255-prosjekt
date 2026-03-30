import torch  # PyTorch
import torch.nn as nn  # Nevrale nettverkslag

from src.model.embeddings import Embeddings  # Token- og positional embeddings
from src.model.transformer_block import TransformerBlock  # Transformer-blokk

# Hovedmodellen som setter sammen alt, forward pass og output head


# ShakespeareGPT in action
class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_layers, num_heads, ff_mult, dropout):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        self.embeddings = Embeddings(vocab_size, embed_dim, block_size)  # Lager token- og positional embeddings

        self.blocks = nn.Sequential(  # Lager flere transformer-blokker etter hverandre
            *[
                TransformerBlock(embed_dim, num_heads, block_size, ff_mult, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)  # Siste LayerNorm etter alle transformer-blokkene

        self.head = nn.Linear(embed_dim, vocab_size, bias=False)  # Output head (ingen bias for weight tying)

        # Weight tying: Bruk samme vekter for input embeddings og output head
        self.head.weight = self.embeddings.token_embedding.embedding.weight

    def forward(self, x):
        x = self.embeddings(x)  # Gjør token-IDer om til embeddings

        x = self.blocks(x)  # Sender gjennom transformer-blokkene

        x = self.ln_f(x)  # Siste layer norm

        logits = self.head(x)  # Predikerer neste token

        return logits  # Returnerer logits