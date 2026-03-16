#evaluering, val-loss og eventuelt tekstgenerering
import torch

from src.data.data import load_text
from src.data.tokenizer import build_tokenizer, encode, decode
from src.model.model import ShakespeareModel


# Model hyperparameters (must match training)
block_size = 64
embed_dim = 128
num_layers = 4


def generate(model, idx, max_new_tokens):
    """
    Generate new tokens from the model.
    """

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx


def main():

    # Load dataset to rebuild tokenizer
    text = load_text()

    stoi, itos = build_tokenizer(text)

    vocab_size = len(stoi)

    # Initialize model
    model = ShakespeareModel(vocab_size, embed_dim, block_size, num_layers)

    model.eval()

    prompt = "Thou art"

    input_ids = torch.tensor([encode(prompt, stoi)], dtype=torch.long)

    output = generate(model, input_ids, max_new_tokens=200)

    result = decode(output[0].tolist(), itos)

    print("\nGenerated text:\n")
    print(result)


if __name__ == "__main__":
    main()
