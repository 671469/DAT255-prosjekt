# Trainingsloop, optimizer, loss

import os  # Brukes for filstier og mapper
import yaml  # Leser YAML-config
import torch  
import torch.nn as nn  # Loss-funksjoner
import torch.optim as optim  # Optimizers
import wandb  # Weights & Biases logging
import math #til perplexity modellen 

from src.training.data_utils import prepare_data, get_batch  # Datapipeline og batching
from src.model.model import ShakespeareModel  # Selve modellen


def load_config(config_path="configs/baseline.yaml"):  # Leser config-fil fra disk
    with open(config_path, "r", encoding="utf-8") as f:  # Åpner YAML-filen
        return yaml.safe_load(f)  # Gjør YAML om til Python-dictionary


def estimate_loss(model, data, eval_iters, context_length, batch_size, device):  # Regner gjennomsnittlig loss over flere batches
    model.eval()  # Setter modellen i eval-modus
    loss_fn = nn.CrossEntropyLoss()  # Samme loss som under trening
    losses = []  # Samler loss-verdier

    with torch.no_grad():  # Skrur av gradienter under evaluering
        for _ in range(eval_iters):  # Kjører flere eval-batches
            x, y = get_batch(data, context_length, batch_size, device=device)  # Henter batch fra valgt datasplit
            logits = model(x)  # Forward pass

            B, T, C = logits.shape  # B=batch, T=tidssteg, C=vocab size
            loss = loss_fn(logits.view(B * T, C), y.view(B * T))  # Flater ut før loss
            losses.append(loss.item())  # Lagrer skalar loss

    model.train()  # Setter modellen tilbake i train-modus
    return sum(losses) / len(losses)  # Returnerer gjennomsnittlig loss


def train(config_path="configs/baseline.yaml"):  # Hovedfunksjon for trening
    config = load_config(config_path)  # Leser all config fra YAML

    model_cfg = config["model"]  # Henter modellparametere
    train_cfg = config["training"]  # Henter treningsparametere
    log_cfg = config["logging"]  # Henter logging-parametere
    ckpt_cfg = config["checkpoint"]  # Henter checkpoint-parametere

    requested_device = train_cfg.get("device", "cpu")  # Leser ønsket device fra config
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"  # Faller tilbake til CPU hvis GPU ikke finnes

    train_data, val_data, tokenizer = prepare_data(split_ratio=train_cfg["train_split"])  # Laster tekst, tokeniserer, tensoriserer og splitter data
    vocab_size = tokenizer.vocab_size  # Henter vokabularstørrelse fra tokenizeren
    
    run = None  # Placeholder for W&B-run
    if log_cfg.get("use_wandb", False):  # Logger bare hvis config sier ja
        run_name = f"{log_cfg['run_name']}_lr{train_cfg['lr']}_layers{model_cfg['n_layers']}"  # Lager automatisk run-navn basert på learning rate og transformer blocks

        run = wandb.init(  # Starter en W&B-run
            entity=log_cfg["entity"],  # W&B entity/team
            project=log_cfg["project"],  # Prosjektnavn i W&B
            name=run_name,  # Navn på run
            config=config,  # Logger hele YAML-configen til W&B
        )

    model = ShakespeareModel(  # Lager modellen
        vocab_size=vocab_size,  # Output-dimensjon må matche vocab
        embed_dim=model_cfg["d_model"],  # Embedding-dimensjon
        block_size=model_cfg["context_length"],  # Hvor lang kontekst modellen ser
        num_layers=model_cfg["n_layers"],  # Antall lag
        num_heads=model_cfg["n_heads"],  # Antall attention heads
        ff_mult=model_cfg["ff_mult"],  # Feed-forward multiplier
        dropout=model_cfg["dropout"],  # Dropout rate
    ).to(device)  # Flytter modellen til CPU/GPU

    model.train()  # Setter modellen i train-modus

    lr = float(train_cfg["lr"])
    optimizer = optim.AdamW(model.parameters(), lr=lr) # AdamW optimizer
    loss_fn = nn.CrossEntropyLoss()  # Loss for neste-token-prediksjon

    for step in range(train_cfg["max_iters"]):  # Hovedloop for trening
        x, y = get_batch(  # Henter tilfeldig treningsbatch
            train_data,
            model_cfg["context_length"],
            train_cfg["batch_size"],
            device=device,
        )

        logits = model(x)  # Forward pass
        B, T, C = logits.shape  # Leser ut shape

        loss = loss_fn(logits.view(B * T, C), y.view(B * T))  # Beregner loss

        optimizer.zero_grad()  # Nullstiller gamle gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Oppdaterer vekter

        if step % train_cfg["eval_interval"] == 0:  # Evaluerer med jevne mellomrom
            train_loss = estimate_loss(  # Estimerer train loss
                model,
                train_data,
                train_cfg["eval_iters"],
                model_cfg["context_length"],
                train_cfg["batch_size"],
                device,
            )
            val_loss = estimate_loss(  # Estimerer validation loss
                model,
                val_data,
                train_cfg["eval_iters"],
                model_cfg["context_length"],
                train_cfg["batch_size"],
                device,
            )
            #train_perplexity = math.exp(train_loss) #eventuelt endre til eks. (min(train_loss, 20))
            #val_perplexity = math.exp(val_loss) #eventuelt endre til eks. (min(val_loss, 20))
           
            #print(f"Step {step} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")  # Skriver status til terminal
            
            print(
                f"Step {step} | " f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | " 
                f"train ppl {train_perplexity:.2f} | val ppl {val_perplexity:.2f}") #Oppdatert skriver status til terminal så den inkluderer perplexity

            if run is not None:  # Logger til W&B hvis aktiv
                wandb.log({  # Sender metrics til dashboardet
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_perplexity": math.exp(train_loss),
                    "val_perplexity": math.exp(val_loss),
                })

    os.makedirs(ckpt_cfg["out_dir"], exist_ok=True)  # Lager mappe for checkpoints i colab
    torch.save(model.state_dict(), ckpt_cfg["save_path"])  # Lagrer modellvektene

    print(f"Modellen er lagret i {ckpt_cfg['save_path']}")  # Bekrefter lagring

    if run is not None:  # Avslutter W&B-run pent
        run.finish()


if __name__ == "__main__":  # Kjøres bare når filen startes direkte
    train()