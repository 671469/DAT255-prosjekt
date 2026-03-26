import torch

#Train/val split og batching

from src.data.data import load_text  # Laster inn renset tekst
from src.data.tokenizer import CharTokenizer  # Importerer tokenizer


def prepare_data(split_ratio=0.9):  # Full datapipeline: last inn tekst, tokenize, tensor, split. Gjør alt klart til trening.
    text = load_text()  # Leser inn den rensede Shakespeare-teksten

    tokenizer = CharTokenizer(text)  # Bygger tokenizer basert på alle tegn i datasettet

    token_ids = tokenizer.encode(text)  # Gjør hele teksten om til en liste med token-IDer
    data = torch.tensor(token_ids, dtype=torch.long)  # Konverterer listen til en PyTorch-tensor

    train_data, val_data = train_val_split(data, split_ratio=split_ratio)  # Splitter data i train og val

    return train_data, val_data, tokenizer  # Returnerer treningsdata, valideringsdata og tokenizer


def train_val_split(data, split_ratio=0.9):  # Deler datasettet i treningsdel og valideringsdel
    n = int(split_ratio * len(data))  # Regner ut hvor splitten skal gå, f.eks. 90 % train

    train_data = data[:n]  # Tar første del som treningsdata
    val_data = data[n:]  # Tar resten som valideringsdata

    return train_data, val_data  # Returnerer begge delene


def get_batch(data, block_size, batch_size, device="cpu"):  # Lager en tilfeldig batch med input og target
    ix = torch.randint(0, len(data) - block_size, (batch_size,))  # Trekker tilfeldige startindekser i datasettet

    x = torch.stack([data[i:i + block_size] for i in ix])  # Input-sekvenser med lengde block_size
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # Target er samme sekvens forskjøvet én token

    return x.to(device), y.to(device)  # Flytter batchen til cpu/gpu og returnerer den. Safety greie for å unngå krasj