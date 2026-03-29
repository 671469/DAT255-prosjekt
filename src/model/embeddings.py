import torch 
import torch.nn as nn  # Inneholder .Embedding

# token- og positional embeddings

# Token-ID -> embedding-vektor
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Slår opp hver token-ID i en trenbar embedding-tabell

    def forward(self, x):
        return self.embedding(x)  # Input: (B, T) -> Output: (B, T, embed_dim)


# Trenbar positional embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, embed_dim):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        self.embedding = nn.Embedding(block_size, embed_dim)  # Lærer en egen embedding for hver posisjon fra 0 til block_size - 1

    def forward(self, x):
        B, T = x.shape  # B=batch size, T=sekvenslengde

        positions = torch.arange(T, device=x.device).unsqueeze(0)  # Lager posisjoner [0, 1, 2, ... T-1] på samme device som input

        return self.embedding(positions)  # Output: (1, T, embed_dim), broadcastes senere over batch-dimensjonen


# Kombiner token- og positional embeddings
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)  # Embedding for selve tokenene
        self.position_embedding = PositionalEmbedding(block_size, embed_dim)  # Embedding for posisjon i sekvensen

    def forward(self, x):
        token_emb = self.token_embedding(x)  # Token embeddings: (B, T, embed_dim)

        pos_emb = self.position_embedding(x)  # Positional embeddings: (1, T, embed_dim)

        return token_emb + pos_emb  # Summerer token og posisjon -> output: (B, T, embed_dim)