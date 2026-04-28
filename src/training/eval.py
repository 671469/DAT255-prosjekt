# evaluering, val-loss og tekstgenerering

import torch  # PyTorch for tensor-operasjoner og modellkjøring
import yaml  # Brukes til å lese YAML-config filer

from src.training.data_utils import prepare_data, get_batch  # Datapipeline (laster tekst, tokenizer osv.)
from src.model.model import ShakespeareModel  # Selve modellen


def load_config(config_path="configs/baseline.yaml"):  # Leser config-fil fra disk
    with open(config_path, "r", encoding="utf-8") as f:  # Åpner YAML-filen i lesemodus
        return yaml.safe_load(f)  # Parser YAML til Python-dictionary


@torch.no_grad()  # Skrur av gradientberegning (vi trener ikke, bare genererer tekst)
def generate(model, idx, max_new_tokens, block_size, temperature, top_k, repetition_penalty=None, no_repeat_ngram_size=None):
    was_training = model.training  # Lagrer om modellen var i train-modus før generering

    model.eval()  # Setter modellen i eval-modus (deaktiverer dropout osv.)

    for _ in range(max_new_tokens):  # Genererer ett nytt token per iterasjon

        idx_cond = idx[:, -block_size:]  # Tar kun siste del av sekvensen (context window)

        logits = model(idx_cond)  # Forward pass gjennom modellen
        logits = logits[:, -1, :]  # Tar kun logits for siste token i sekvensen

        logits = logits / temperature  # Skalerer logits med temperature (styrer randomness)

        if repetition_penalty is not None and repetition_penalty > 1.0:  # Hvis repetition penalty er aktivert
            for b in range(idx.shape[0]):  # Går gjennom hver sekvens i batchen
                used_tokens = set(idx[b].tolist())  # Finner tokens som allerede er generert i sekvensen
                for token_id in used_tokens:  # Går gjennom tokens som allerede finnes
                    if logits[b, token_id] > 0:  # Hvis logit er positiv
                        logits[b, token_id] = logits[b, token_id] / repetition_penalty  # Senker sannsynlighet for gjentatte tokens
                    else:  # Hvis logit er negativ
                        logits[b, token_id] = logits[b, token_id] * repetition_penalty  # Gjør negativ logit enda mindre attraktiv

        probs = torch.softmax(logits, dim=-1)  # Gjør logits om til sannsynligheter

        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0 and idx.shape[1] >= no_repeat_ngram_size - 1:  # Hvis ngram-repetisjon skal blokkeres
            for b in range(idx.shape[0]):  # Går gjennom hver sekvens i batchen
                generated = idx[b].tolist()  # Gjør genererte token-IDer om til liste

                if len(generated) >= no_repeat_ngram_size - 1:  # Sjekker at sekvensen er lang nok
                    prefix = generated[-(no_repeat_ngram_size - 1):] if no_repeat_ngram_size > 1 else []  # Siste n-1 tokens brukes som prefix
                    banned_tokens = []  # Samler tokens som ville laget et repetert ngram

                    for i in range(len(generated) - no_repeat_ngram_size + 1):  # Går gjennom tidligere ngrams
                        previous_prefix = generated[i:i + no_repeat_ngram_size - 1] if no_repeat_ngram_size > 1 else []  # Tidligere prefix
                        next_token = generated[i + no_repeat_ngram_size - 1]  # Tokenet som fulgte etter tidligere prefix

                        if previous_prefix == prefix:  # Hvis samme prefix allerede har forekommet
                            banned_tokens.append(next_token)  # Blokker tokenet som ville gjentatt ngrammet

                    if banned_tokens:  # Hvis vi fant tokens som skal blokkeres
                        probs[b, banned_tokens] = 0.0  # Setter sannsynligheten til 0 for disse tokenene

            probs_sum = probs.sum(dim=-1, keepdim=True)  # Summerer sannsynlighetene etter blokkering
            probs = probs / probs_sum.clamp_min(1e-12)  # Re-normaliserer og unngår deling på null

        if top_k is not None:  # Hvis vi bruker top-k sampling
            v, ix = torch.topk(probs, top_k)  # Henter de k mest sannsynlige tokenene
            probs = torch.zeros_like(probs).scatter_(1, ix, v)  # Nuller ut alle andre tokens
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normaliserer sannsynlighetene

        next_token = torch.multinomial(probs, num_samples=1)  # Sampler ett token fra distribusjonen

        idx = torch.cat((idx, next_token), dim=1)  # Legger til det nye tokenet i sekvensen

    if was_training:  # Hvis modellen var i train-modus før generering
        model.train()  # Setter modellen tilbake til train-modus slik at dropout brukes videre under trening

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
        repetition_penalty=gen_cfg.get("repetition_penalty"),  # Leser repetition penalty fra config hvis satt
        no_repeat_ngram_size=gen_cfg.get("no_repeat_ngram_size"),  # Leser ngram-blokkering fra config hvis satt
    )

    result = tokenizer.decode(output[0].tolist())  # Gjør token-IDer tilbake til tekst

    print("\nGenerated text:\n")  # Formattering
    print(result)  # Skriver generert tekst


if __name__ == "__main__":  # Kjør kun hvis filen startes direkte
    main()

# Enkel top-k versjon only. Vil prøve på top-p senere, men å legge inn toggle var litt vanskelig.