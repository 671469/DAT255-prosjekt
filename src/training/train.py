# Trainingsloop, optimizer, loss
import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.data.data import load_text
from src.data.tokenizer import build_tokenizer, encode_dataset
from src.training.data_utils import train_val_split, get_batch
from src.model.model import ShakespeareModel


# Hyperparametere
batch_size = 32
block_size = 64
embed_dim = 128
num_layers = 4
learning_rate = 3e-4
max_iters = 15000


def train():

    # Last inn datasettet
    text = load_text()

    # Bygg tokenizer (tegn → tall)
    stoi, itos = build_tokenizer(text)

    # Gjør datasettet til tensor
    data = encode_dataset(text, stoi)

    vocab_size = len(stoi)

    # Treningsdata | valideringsdata
    train_data, val_data = train_val_split(data)

    # Start modell
    net = ShakespeareModel(vocab_size, embed_dim, block_size, num_layers)

    # Optimizer
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

    # Loss-funksjon
    loss_fn = nn.CrossEntropyLoss()

    # Treningsloop
    for step in range(max_iters):

        # Hent batch
        x, y = get_batch(train_data, block_size, batch_size)

        # Forward pass
        logits = net(x)

        B, T, C = logits.shape

        # Beregn loss
        loss = loss_fn(
            logits.view(B * T, C),
            y.view(B * T)
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    # Lag models-mappe hvis den ikke finnes
    os.makedirs("models", exist_ok=True)

    # Lagre modellen
    torch.save(net.state_dict(), "models/shakespeare_model.pt")

    print("Modellen er lagret!")


if __name__ == "__main__":
    train()
