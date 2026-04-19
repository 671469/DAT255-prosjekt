import torch 
import torch.nn as nn  # Inneholder nevrale nettverkslag
import torch.nn.functional as F  # Funksjonelle operasjoner som softmax

# self-attention, multi-head attention


# ett "attention-head"
class Head(nn.Module):
    def __init__(self, embed_dim, head_size, block_size, dropout):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        self.key = nn.Linear(embed_dim, head_size, bias=False)  # Lineær projeksjon fra embed_dim -> head_size (keys)
        self.query = nn.Linear(embed_dim, head_size, bias=False)  # Lineær projeksjon fra embed_dim -> head_size (queries)
        self.value = nn.Linear(embed_dim, head_size, bias=False)  # Lineær projeksjon fra embed_dim -> head_size (values)

        self.head_size = head_size  # Lagrer head_size for bruk i skalering av attention
        self.dropout = nn.Dropout(dropout)  # Dropout på attention weights (ingen effekt hvis dropout=0.0)

        self.register_buffer(  # Registrerer en tensor som ikke er en trenbar parameter
            "mask",
            torch.tril(torch.ones(block_size, block_size))  # Nedre triangulær matrise for causal masking
        )

    def forward(self, x):
        B, T, C = x.shape  # B=batch size, T=sekvenslengde, C=embed_dim

        k = self.key(x)  # Lager key-matrise: (B, T, head_size)
        q = self.query(x)  # Lager query-matrise: (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # Attention scores skalert med head_size

        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # Maskerer fremtidige tokens (causal mask)

        wei = F.softmax(wei, dim=-1)  # Gjør scores om til sannsynligheter
        wei = self.dropout(wei)  # Dropout på attention weights

        v = self.value(x)  # Lager value-matrise: (B, T, head_size)

        out = wei @ v  # Multipliserer attention weights med values: (B, T, head_size)

        return out  # Returnerer output fra én attention-head


# flere "attention-heads"
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()  # Initialiserer parent-klassen nn.Module

        assert embed_dim % num_heads == 0, "embed_dim må være delelig på num_heads"  # Sjekker at embed_dim kan deles likt på heads

        head_size = embed_dim // num_heads  # Deler embedding-dimensjonen likt på antall heads

        self.heads = nn.ModuleList(  # Lager en liste med flere attention-heads
            [Head(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(embed_dim, embed_dim)  # Lineær projeksjon etter concatenation av heads
        self.dropout = nn.Dropout(dropout)  # Dropout etter output projection (ingen effekt hvis dropout=0.0)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Kjører hver head og concatenater langs feature-dimensjonen

        out = self.proj(out)  # Projiserer tilbake til embed_dim
        out = self.dropout(out)  # Dropout etter projeksjon

        return out  # Returnerer output fra multi-head attention