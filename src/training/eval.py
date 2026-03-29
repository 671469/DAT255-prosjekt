# evaluering, val-loss og tekstgenerering

import torch 
import torch.nn as nn  # Inneholder loss-funksjoner som CrossEntropyLoss
import yaml  # Brukes til å lese config-filen (YAML)

from src.training.data_utils import prepare_data, get_batch  # Datapipeline og batching
from src.model.model import ShakespeareModel  # Selve transformer-modellen


def load_config(config_path="configs/baseline.yaml"):  # Leser config-fil fra disk
    with open(config_path, "r", encoding="utf-8") as f:  # Åpner YAML-filen
        return yaml.safe_load(f)  # Gjør YAML om til Python-dictionary


@torch.no_grad()  # Skrur av gradienter under evaluering (sparer minne og tid)
def estimate_loss(model, data, eval_iters, context_length, batch_size, device):
    model.eval()  # Setter modellen i eval-modus (deaktiverer dropout osv.)
    loss_fn = nn.CrossEntropyLoss()  # Samme loss som under trening
    losses = []  # Liste for å samle loss-verdier

    for _ in range(eval_iters):  # Kjører flere batches for mer stabilt estimat
        x, y = get_batch(  # Henter batch fra datasettet
            data,
            block_size=context_length,  # Lengde på input-sekvens
            batch_size=batch_size,  # Antall sekvenser i batch
            device=device,  # CPU eller GPU
        )

        logits = model(x)  # Forward pass gjennom modellen
        B, T, C = logits.shape  # B=batch, T=tidssteg, C=vocab size

        loss = loss_fn(logits.view(B * T, C), y.view(B * T))  # Flater ut før loss
        losses.append(loss.item())  # Lagrer loss som vanlig tall

    return sum(losses) / len(losses)  # Returnerer gjennomsnittlig loss


@torch.no_grad()  # Ingen gradienter under tekstgenerering
def generate(model, idx, max_new_tokens, block_size, temperature, top_k):
    model.eval()  # Setter modellen i eval-modus

    if temperature <= 0:  # Temperatur må være gyldig
        raise ValueError("temperature må være > 0")

    for _ in range(max_new_tokens):  # Genererer ett token per iterasjon
        idx_cond = idx[:, -block_size:]  # Bruk bare siste context-vindu

        logits = model(idx_cond)  # Forward pass
        logits = logits[:, -1, :]  # Tar kun siste token-prediksjon

        logits = logits / temperature  # Skalerer logits med temperature

        if top_k is not None:  # Bruk top-k sampling hvis satt i config
            k = min(top_k, logits.size(-1))  # Safety: k <= vocab size
            v, ix = torch.topk(logits, k)  # Henter topp k logits
            logits[logits < v[:, [-1]]] = float("-inf")  # Fjerner resten

        probs = torch.softmax(logits, dim=-1)  # Gjør logits om til sannsynligheter
        next_token = torch.multinomial(probs, num_samples=1)  # Sampler neste token

        idx = torch.cat((idx, next_token), dim=1)  # Legger til token i sekvensen

    return idx  # Returnerer hele sekvensen


def main(config_path="configs/baseline.yaml"):  # Hovedfunksjon for evaluering
    config = load_config(config_path)  # Leser config

    model_cfg = config["model"]  # Modellparametere
    train_cfg = config["training"]  # Treningsparametere
    ckpt_cfg = config["checkpoint"]  # Checkpoint-parametere
    gen_cfg = config["generation"]  # Parametere for tekstgenerering

    requested_device = train_cfg.get("device", "cpu")  # Leser ønsket device
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"  # Fallback til CPU

    train_data, val_data, tokenizer = prepare_data(split_ratio=train_cfg["train_split"])  # Laster og splitter data
    vocab_size = tokenizer.vocab_size  # Henter vokabularstørrelse

    model = ShakespeareModel(  # Lager modellen med samme config som trening
        vocab_size=vocab_size,
        embed_dim=model_cfg["d_model"],
        block_size=model_cfg["context_length"],
        num_layers=model_cfg["n_layers"],
    ).to(device)  # Flytter modellen til CPU/GPU

    model.load_state_dict(torch.load(ckpt_cfg["save_path"], map_location=device))  # Laster trenede vekter
    model.eval()  # Setter modellen i eval-modus

    train_loss = estimate_loss(  # Estimerer train loss
        model,
        train_data,
        eval_iters=train_cfg["eval_iters"],
        context_length=model_cfg["context_length"],
        batch_size=train_cfg["batch_size"],
        device=device,
    )

    val_loss = estimate_loss(  # Estimerer validation loss
        model,
        val_data,
        eval_iters=train_cfg["eval_iters"],
        context_length=model_cfg["context_length"],
        batch_size=train_cfg["batch_size"],
        device=device,
    )

    train_ppl = torch.exp(torch.tensor(train_loss)).item()  # Perplexity = e^loss
    val_ppl = torch.exp(torch.tensor(val_loss)).item()  # Perplexity = e^loss

    print(f"Train loss: {train_loss:.4f} | Train perplexity: {train_ppl:.2f}")  # Logger train metrics
    print(f"Val loss:   {val_loss:.4f} | Val perplexity:   {val_ppl:.2f}")  # Logger val metrics

    prompt = gen_cfg["prompt"]  # Leser starttekst fra config
    max_new_tokens = gen_cfg["max_new_tokens"]  # Leser hvor mange tokens som skal genereres
    temperature = gen_cfg["temperature"]  # Leser temperature fra config
    top_k = gen_cfg["top_k"]  # Leser top_k fra config

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)  # Gjør prompt om til token-IDer

    output_ids = generate(  # Genererer tekst fra modellen
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        block_size=model_cfg["context_length"],
        temperature=temperature,
        top_k=top_k,
    )

    result = tokenizer.decode(output_ids[0].tolist())  # Gjør token-IDer tilbake til tekst

    print("\nGenerated text:\n")
    print(result)  # Skriver generert tekst


if __name__ == "__main__":  # Kjør kun hvis filen startes direkte
    main()

#Enkel top-k versjon only. Vil prøve på top-p senere, men å legge inn toggle var litt vanskelig.