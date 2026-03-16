#Trainingsloop, optimizer, loss
import torch
import torch.nn as nn
import torch.optim as optim
from torchgen import model
from torchgen import model

from src.data.data import load_text
from src.data.tokenizer import build_tokenizer, encode_dataset
from src.training.data_utils import train_val_split, get_batch
from src.model.model import ShakespeareModel


# Hyperparameters
batch_size = 32
block_size = 64
embed_dim = 128
num_layers = 4
learning_rate = 3e-4
max_iters = 5000


def train():

    # Load dataset
    text = load_text()

    # Build tokenizer
    stoi, itos = build_tokenizer(text)

    # Encode
    data = encode_dataset(text, stoi)

    vocab_size = len(stoi)

    # tren | valider
    train_data, val_data = train_val_split(data)

    # model start
    model = ShakespeareModel(vocab_size, embed_dim, block_size, num_layers)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    for step in range(max_iters):

        # Get tbatch
        x, y = get_batch(train_data, block_size, batch_size)

        logits = model(x)

        B, T, C = logits.shape

        loss = loss_fn(
            logits.view(B * T, C),
            y.view(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

torch.save(model.state_dict(), "models/shakespeare_model.pt")

if __name__ == "__main__":
    train()