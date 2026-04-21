import os  # Brukes for å sjekke om tokenizer-fil finnes
import torch  # PyTorch for tensor-operasjoner

from src.data.data import load_text  # Laster inn renset tekst
from src.data.tokenizer import CharTokenizer, BPETokenizer  # Importerer begge tokenizer-variantene


def prepare_data(
    train_ratio=0.8,  # Hvor stor andel av råteksten som går til trening
    val_ratio=0.1,  # Hvor stor andel av råteksten som går til validering
    tokenizer_type="char",  # Velger tokenizer-type: "char" eller "bpe"
    tokenizer_path=None,  # Filsti for lagring / lasting av BPE-tokenizer
    vocab_size=2000,  # Ønsket vokabularstørrelse for BPE
    min_frequency=2,  # Minste frekvens for at et BPE-token skal bli med
):
    # Full datapipeline:
    # 1) last inn råtekst
    # 2) splitt råtekst i train / val / test
    # 3) tren eller last tokenizer
    # 4) encode hver split med samme tokenizer
    # 5) returner train_data, val_data, test_data og tokenizer

    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio og val_ratio må være > 0")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio må være < 1.0 slik at vi får en test-split også")

    text = load_text()  # Leser inn hele den rensede Shakespeare-teksten som én streng

    train_text, val_text, test_text = split_text_three_way(
        text,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )  # Splitter råteksten før tokenizer-trening for å unngå leakage fra val/test

    if tokenizer_type == "char":  # Hvis vi bruker char-level tokenizer
        tokenizer = CharTokenizer(train_text)  # Bygger tokenizer kun fra train-teksten
    elif tokenizer_type == "bpe":  # Hvis vi bruker BPE-tokenizer
        if tokenizer_path is None:
            raise ValueError("tokenizer_path må settes når tokenizer_type='bpe'")

        if os.path.exists(tokenizer_path):  # Hvis tokenizer-fila allerede finnes
            tokenizer = BPETokenizer.load(tokenizer_path)  # Laster eksisterende tokenizer fra disk
        else:  # Hvis tokenizer-fila ikke finnes
            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True) if os.path.dirname(tokenizer_path) else None
            tokenizer = BPETokenizer.train(
                text=train_text,  # Trener BPE kun på train-teksten
                save_path=tokenizer_path,  # Hvor tokenizeren skal lagres
                vocab_size=vocab_size,  # Hvor mange subword tokens vi ønsker
                min_frequency=min_frequency,  # Hvor ofte et token minst må forekomme
            )
    else:
        raise ValueError(f"Ukjent tokenizer_type: {tokenizer_type}")

    train_ids = tokenizer.encode(train_text)  # Encoder train-splittet med den valgte tokenizeren
    val_ids = tokenizer.encode(val_text)  # Encoder val-splittet med samme tokenizer
    test_ids = tokenizer.encode(test_text)  # Encoder test-splittet med samme tokenizer

    train_data = torch.tensor(train_ids, dtype=torch.long)  # Gjør train token-IDer om til tensor
    val_data = torch.tensor(val_ids, dtype=torch.long)  # Gjør val token-IDer om til tensor
    test_data = torch.tensor(test_ids, dtype=torch.long)  # Gjør test token-IDer om til tensor

    return train_data, val_data, test_data, tokenizer  # Returnerer alle tre splittene + tokenizer


def split_text_three_way(text, train_ratio=0.8, val_ratio=0.1):
    # Deler råteksten i tre deler basert på tegnposisjon:
    # train / val / test

    n = len(text)  # Total lengde på teksten i antall tegn
    train_end = int(train_ratio * n)  # Sluttindeks for train-delen
    val_end = int((train_ratio + val_ratio) * n)  # Sluttindeks for val-delen

    train_text = text[:train_end]  # Første del av teksten brukes til trening
    val_text = text[train_end:val_end]  # Midterste del brukes til validering
    test_text = text[val_end:]  # Siste del brukes til test

    return train_text, val_text, test_text  # Returnerer de tre tekstsplittene


def get_batch(data, block_size, batch_size, device="cpu"):  # Lager en tilfeldig batch med input og target
    if len(data) <= block_size:
        raise ValueError(
            f"Datasplitt er for liten for block_size={block_size}. "
            f"Lengde på splitten er {len(data)}."
        )

    ix = torch.randint(0, len(data) - block_size, (batch_size,))  # Trekker tilfeldige startindekser i datasettet

    x = torch.stack([data[i:i + block_size] for i in ix])  # Input-sekvenser med lengde block_size
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # Target er samme sekvens forskjøvet én token

    return x.to(device), y.to(device)  # Flytter batchen til cpu/gpu og returnerer den