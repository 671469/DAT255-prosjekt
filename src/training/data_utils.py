import os  # Brukes for å sjekke om tokenizer-fil finnes
import torch  # PyTorch for tensor-operasjoner

# Train/val split og batching

from src.data.data import load_text  # Laster inn renset tekst
from src.data.tokenizer import CharTokenizer, BPETokenizer  # Importerer begge tokenizer-variantene


def prepare_data(
    split_ratio=0.9,  # Hvor stor andel av data som går til trening
    tokenizer_type="char",  # Velger tokenizer-type: "char" eller "bpe"
    tokenizer_path=None,  # Filsti for lagring / lasting av BPE-tokenizer
    vocab_size=2000,  # Ønsket vokabularstørrelse for BPE
    min_frequency=2,  # Minste frekvens for at et BPE-token skal bli med
):
    # Full datapipeline: last inn tekst, tokenize, tensor, split. Gjør alt klart til trening.

    text = load_text()  # Leser inn den rensede Shakespeare-teksten

    if tokenizer_type == "char":  # Hvis vi bruker gammel char-level tokenizer
        tokenizer = CharTokenizer(text)  # Bygger tokenizer basert på alle tegn i datasettet

    elif tokenizer_type == "bpe":  # Hvis vi bruker BPE-tokenizer
        if tokenizer_path is None:  # Sikkerhetssjekk hvis path mangler i config
            raise ValueError("tokenizer_path må settes når tokenizer_type='bpe'")  # Stopper med tydelig feilmelding

        if os.path.exists(tokenizer_path):  # Hvis tokenizer-fila allerede finnes
            tokenizer = BPETokenizer.load(tokenizer_path)  # Laster eksisterende BPE-tokenizer fra disk
        else:  # Hvis tokenizer-fila ikke finnes ennå
            tokenizer = BPETokenizer.train(  # Trener ny BPE-tokenizer på teksten
                text=text,  # Hele teksten brukes som treningsgrunnlag for tokenizer
                save_path=tokenizer_path,  # Hvor tokenizeren skal lagres
                vocab_size=vocab_size,  # Hvor mange subword tokens vi ønsker
                min_frequency=min_frequency,  # Hvor ofte et token minst må forekomme
            )

    else:  # Hvis tokenizer_type ikke er støttet
        raise ValueError(f"Ukjent tokenizer_type: {tokenizer_type}")  # Stopper med tydelig feilmelding

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