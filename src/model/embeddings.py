import torch
import torch.nn as nn

#Token ID til embed vector
class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):

        return self.embedding(x)

#trenbar pos embed 
class PositionalEmbedding(nn.Module):

    def __init__(self, block_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, x):

        B, T = x.shape

        positions = torch.arange(T, device=x.device).unsqueeze(0)

        return self.embedding(positions)

#kombiner token og pos embed
class Embeddings(nn.Module):

    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(block_size, embed_dim)

    def forward(self, x):

        token_emb = self.token_embedding(x)

        pos_emb = self.position_embedding(x)

        return token_emb + pos_emb
