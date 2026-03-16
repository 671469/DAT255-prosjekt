# Skal bygge vokabular, encode, decode
import torch


def build_tokenizer(text):
    """
    Build vocabulary from dataset.
    """
    chars = sorted(list(set(text)))  # Alle unike tegn i teksten

    stoi = {ch: i for i, ch in enumerate(chars)}  # string-to-index (tegn -> tall)
    itos = {i: ch for i, ch in enumerate(chars)}  # index-to-string (tall -> tegn)

    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    return stoi, itos


# encoder
def encode(text, stoi):
    """
    Gjør tekst om til tall
    """
    return [stoi[ch] for ch in text]


# decoder
def decode(ids, itos):
    """
    Gjør tall om til tekst
    """
    return "".join([itos[i] for i in ids])


# toTensor
def encode_dataset(text, stoi):
    """
    Encode entire dataset and convert to torch tensor
    """
    encoded = encode(text, stoi)
    return torch.tensor(encoded, dtype=torch.long)


if __name__ == "__main__":

    # Import here to avoid circular imports
    from src.data.data import load_text

    # Load dataset
    text = load_text()

    # Build vocabulary
    stoi, itos = build_tokenizer(text)

    # Encode dataset
    data = encode_dataset(text, stoi)
    print(f"Encoded tensor shape: {data.shape}")

    # Test tokenizer
    sample = "thou art fair"

    encoded = encode(sample, stoi)
    decoded = decode(encoded, itos)

    print("\nExample text:", sample)
    print("Encoded:", encoded)
    print("Decoded:", decoded)