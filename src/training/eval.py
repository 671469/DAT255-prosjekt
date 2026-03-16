# evaluering, val-loss og tekstgenerering
import torch

from src.data.data import load_text
from src.data.tokenizer import build_tokenizer, encode, decode
from src.model.model import ShakespeareModel


# Modell-hyperparametere (må matche train.py)
block_size = 64
embed_dim = 128
num_layers = 4

# Velg CPU eller GPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# Generer nye tokens fra modellen
def generate(model, idx, max_new_tokens):

    for _ in range(max_new_tokens):

        # bruk bare siste context-vindu
        idx_cond = idx[:, -block_size:]

        # forward pass
        logits = model(idx_cond)

        # ta kun siste token-prediksjon
        logits = logits[:, -1, :]

        # temperatur styrer hvor tilfeldig output er
        temperature = 0.8
        logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        # top-k sampling (hindrer rare symboler)
        top_k = 20
        v, ix = torch.topk(probs, top_k)

        probs = torch.zeros_like(probs).scatter_(1, ix, v)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # sample neste token
        next_token = torch.multinomial(probs, num_samples=1)

        # legg til token i sekvensen
        idx = torch.cat((idx, next_token), dim=1)

    return idx


def main():

    # last tekst for å bygge tokenizer
    text = load_text()

    stoi, itos = build_tokenizer(text)

    vocab_size = len(stoi)

    # initialiser modell
    model = ShakespeareModel(vocab_size, embed_dim, block_size, num_layers).to(device)

    # last trenede vekter
    model.load_state_dict(torch.load("models/shakespeare_model.pt", map_location=device))

    model.eval()

    # starttekst
    prompt = "Thou art"

    input_ids = torch.tensor([encode(prompt, stoi)], dtype=torch.long).to(device)

    # generer tekst
    output = generate(model, input_ids, max_new_tokens=200)

    result = decode(output[0].tolist(), itos)

    print("\nGenerated text:\n")
    print(result)


if __name__ == "__main__":
    main()
