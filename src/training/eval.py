# evaluering, val-loss og tekstgenerering

import torch  # PyTorch for tensor-operasjoner og modellkjøring
import yaml  # Brukes til å lese YAML-config filer

from src.training.data_utils import prepare_data, get_batch  # Datapipeline (laster tekst, tokenizer osv.)
from src.model.model import ShakespeareModel  # Selve modellen


def load_config(config_path="configs/baseline.yaml"):  # Leser config-fil fra disk
    with open(config_path, "r", encoding="utf-8") as f:  # Åpner YAML-filen i lesemodus
        return yaml.safe_load(f)  # Parser YAML til Python-dictionary


@torch.no_grad()  # Skrur av gradientberegning (vi trener ikke, bare genererer tekst)
def generate(model, idx, max_new_tokens, block_size, temperature, top_k):
    model.eval()  # Setter modellen i eval-modus (deaktiverer dropout osv.)

    for _ in range(max_new_tokens):  # Genererer ett nytt token per iterasjon

        idx_cond = idx[:, -block_size:]  # Tar kun siste del av sekvensen (context window)

        logits = model(idx_cond)  # Forward pass gjennom modellen
        logits = logits[:, -1, :]  # Tar kun logits for siste token i sekvensen

        logits = logits / temperature  # Skalerer logits med temperature (styrer randomness)

        probs = torch.softmax(logits, dim=-1)  # Gjør logits om til sannsynligheter

        if top_k is not None:  # Hvis vi bruker top-k sampling
            v, ix = torch.topk(probs, top_k)  # Henter de k mest sannsynlige tokenene
            probs = torch.zeros_like(probs).scatter_(1, ix, v)  # Nuller ut alle andre tokens
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normaliserer sannsynlighetene

        next_token = torch.multinomial(probs, num_samples=1)  # Sampler ett token fra distribusjonen

        idx = torch.cat((idx, next_token), dim=1)  # Legger til det nye tokenet i sekvensen

    return idx  # Returnerer hele sekvensen (original + genererte tokens)


def main():
    config = load_config()  # Leser config fra YAML

    model_cfg = config["model"]  # Henter modellparametere
    train_cfg = config["training"]  # Henter treningsparametere
    tok_cfg = config["tokenizer"]  # Henter tokenizer-parametere
    gen_cfg = config["generation"]  # Henter genereringsparametere (temperature, top_k osv.)
    ckpt_cfg = config["checkpoint"]  # Henter info om lagret modell

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Velger GPU hvis tilgjengelig, ellers CPU

    _, _, _, tokenizer = prepare_data(  # Laster tokenizer via samme datapipeline som trening, men eval.py trenger ikke selve splittene
        train_ratio=train_cfg["train_ratio"],  # Leser train-ratio fra config
        val_ratio=train_cfg["val_ratio"],  # Leser val-ratio fra config
        tokenizer_type=tok_cfg["type"],  # Leser hvilken tokenizer-type som brukes
        tokenizer_path=tok_cfg.get("model_path"),  # Leser hvor tokenizeren lagres / lastes fra
        vocab_size=tok_cfg.get("vocab_size"),  # Leser ønsket vocab size for BPE
        min_frequency=tok_cfg.get("min_frequency", 2),  # Leser min_frequency for BPE hvis satt
    )

    vocab_size = tokenizer.vocab_size  # Henter størrelse på vokabularet fra tokenizeren

    model = ShakespeareModel(  # Initialiserer modellen med parametere fra config
        vocab_size=vocab_size,
        embed_dim=model_cfg["d_model"],
        block_size=model_cfg["context_length"],
        num_layers=model_cfg["n_layers"],
        num_heads=model_cfg["n_heads"],
        ff_mult=model_cfg["ff_mult"],
        dropout=model_cfg["dropout"],
    ).to(device)  # Flytter modellen til valgt device (CPU/GPU)

    model.load_state_dict(torch.load(ckpt_cfg["save_path"], map_location=device))  # Laster inn trenede vekter
    model.eval()  # Setter modellen i eval-modus

    prompt = gen_cfg["prompt"]  # Starttekst som modellen skal fortsette på

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)  # Gjør prompt om til token-IDer

    output = generate(  # Genererer tekst fra modellen
        model,
        input_ids,
        max_new_tokens=gen_cfg["max_new_tokens"],  # Hvor mange tokens som skal genereres
        block_size=model_cfg["context_length"],  # Hvor mye kontekst modellen bruker
        temperature=gen_cfg["temperature"],  # Leser temperature fra config
        top_k=gen_cfg["top_k"],  # Leser top_k fra config
    )

    result = tokenizer.decode(output[0].tolist())  # Gjør token-IDer tilbake til tekst

    print("\nGenerated text:\n")  # Formattering
    print(result)  # Skriver generert tekst


if __name__ == "__main__":  # Kjør kun hvis filen startes direkte
    main()

# Enkel top-k versjon only. Vil prøve på top-p senere, men å legge inn toggle var litt vanskelig.