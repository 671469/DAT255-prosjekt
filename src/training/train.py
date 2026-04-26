# Trainingsloop, optimizer, loss

import os  # Brukes for filstier og mapper
import yaml  # Leser YAML-config
import torch
import torch.nn as nn  # Loss-funksjoner
import torch.optim as optim  # Optimizers
import wandb  # Weights & Biases logging
import math  # til perplexity modellen

from src.training.data_utils import prepare_data, get_batch  # Datapipeline og batching
from src.model.model import ShakespeareModel  # Selve modellen
from src.training.eval import generate  # gjenbruker tekstgenerering fra eval.py


def load_config(config_path="configs/baseline.yaml"):  # Leser config-fil fra disk
    with open(config_path, "r", encoding="utf-8") as f:  # Åpner YAML-filen
        return yaml.safe_load(f)  # Gjør YAML om til Python-dictionary


def estimate_loss(model, data, eval_iters, context_length, batch_size, device):  # Regner gjennomsnittlig loss over flere batches
    model.eval()  # Setter modellen i eval-modus
    loss_fn = nn.CrossEntropyLoss()  # Samme loss som under trening
    losses = []  # Samler loss-verdier

    with torch.no_grad():  # Skrur av gradienter under evaluering
        for _ in range(eval_iters):  # Kjører flere eval-batches
            x, y = get_batch(data, context_length, batch_size, device=device)  # Henter batch fra valgt datasplitt
            logits = model(x)  # Forward pass

            B, T, C = logits.shape  # B=batch, T=tidssteg, C=vocab size
            loss = loss_fn(logits.view(B * T, C), y.view(B * T))  # Flater ut før loss
            losses.append(loss.item())  # Lagrer skalar loss

    model.train()  # Setter modellen tilbake i train-modus
    return sum(losses) / len(losses)  # Returnerer gjennomsnittlig loss


def get_lr(step, train_cfg):  # Beregner learning rate med lineær warmup + cosine decay
    base_lr = float(train_cfg["lr"])  # Maks learning rate etter warmup
    min_lr = float(train_cfg.get("min_lr", base_lr * 0.1))  # Laveste learning rate mot slutten, default er 10% av base_lr
    warmup_iters = int(train_cfg.get("warmup_iters", 0))  # Antall steg brukt på lineær warmup
    max_iters = int(train_cfg["max_iters"])  # Totalt antall treningssteg

    if warmup_iters > 0 and step < warmup_iters:  # Lineær warmup i starten av treningen
        return base_lr * (step + 1) / warmup_iters  # Øker learning rate gradvis fra nesten 0 til base_lr

    if step >= max_iters:  # Safety hvis step går forbi max_iters
        return min_lr  # Holder learning rate på minimum

    if max_iters == warmup_iters:  # Hindrer deling på null hvis warmup_iters tilfeldigvis er lik max_iters
        return min_lr  # Bruker min_lr som fallback

    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)  # Hvor langt vi er i cosine decay-fasen
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)  # Sørger for at ratio alltid er mellom 0 og 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay-koeffisient fra 1 til 0

    return min_lr + coeff * (base_lr - min_lr)  # Returnerer learning rate mellom base_lr og min_lr


def train(config_path="configs/baseline.yaml"):  # Hovedfunksjon for trening
    config = load_config(config_path)  # Leser all config fra YAML

    model_cfg = config["model"]  # Henter modellparametere
    tok_cfg = config["tokenizer"]  # Henter tokenizer-parametere
    train_cfg = config["training"]  # Henter treningsparametere
    log_cfg = config["logging"]  # Henter logging-parametere
    ckpt_cfg = config["checkpoint"]  # Henter checkpoint-parametere
    gen_cfg = config["generation"]  # Henter genereringsparametere fra YAML så prompt/temperature/top_k kan brukes under logging

    requested_device = train_cfg.get("device", "cpu")  # Leser ønsket device fra config
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"  # Faller tilbake til CPU hvis GPU ikke finnes

    train_data, val_data, test_data, tokenizer = prepare_data(  # Laster tekst, tokeniserer, tensoriserer og splitter data i train/val/test
        train_ratio=train_cfg["train_ratio"],  # Hvor mye som går til train
        val_ratio=train_cfg["val_ratio"],  # Hvor mye som går til val
        tokenizer_type=tok_cfg["type"],  # Velger tokenizer-type fra config
        tokenizer_path=tok_cfg.get("model_path"),  # Hvor tokenizer skal lagres / lastes fra
        vocab_size=tok_cfg.get("vocab_size"),  # Vokabularstørrelse for BPE
        min_frequency=tok_cfg.get("min_frequency", 2),  # Minste frekvens for BPE-token
    )

    vocab_size = tokenizer.vocab_size  # Henter vokabularstørrelse fra tokenizeren

    run = None  # Placeholder for W&B-run
    samples_table = None  # Tabell for å lagre historikk av genererte tekstsamples i W&B

    if log_cfg.get("use_wandb", False):  # Logger bare hvis config sier ja
        run_name = (
            f"{log_cfg['run_name']}_"
            f"{tok_cfg['type']}_"
            f"lr{train_cfg['lr']}_"
            f"layers{model_cfg['n_layers']}"
        )  # Lager run-navn med tokenizer-type, learning rate og antall lag

        run = wandb.init(  # Starter en W&B-run
            entity=log_cfg["entity"],  # W&B entity/team
            project=log_cfg["project"],  # Prosjektnavn i W&B
            name=run_name,  # Navn på run
            config=config,  # Logger hele YAML-configen til W&B
        )

        samples_table = wandb.Table(columns=["step", "train_loss", "val_loss", "val_perplexity", "text"])  # Oppretter W&B-tabell med step og generert tekst + valideringsmålinger

        wandb.define_metric("step")  # Definerer step som felles x-akse i W&B
        wandb.define_metric("train_loss", step_metric="step")  # Viser train_loss mot step
        wandb.define_metric("val_loss", step_metric="step")  # Viser val_loss mot step
        wandb.define_metric("train_perplexity", step_metric="step")  # Viser train_perplexity mot step
        wandb.define_metric("val_perplexity", step_metric="step")  # Viser val_perplexity mot step
        wandb.define_metric("learning_rate", step_metric="step")  # Viser learning rate mot step

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

    lr = float(train_cfg["lr"])  # Sørger for at learning rate er float
    weight_decay = float(train_cfg.get("weight_decay", 0.0))  # Henter weight decay fra config (bruker 0.0 hvis feltet mangler)
    grad_clip = float(train_cfg.get("grad_clip", 0.0))  # Henter gradient clipping fra config (bruker 0.0 hvis feltet mangler)

    use_early_stopping = train_cfg.get("early_stopping", False)  # Leser om early stopping er aktiv fra config
    patience = int(train_cfg.get("patience", 10))  # Hvor mange eval-runder vi tåler uten forbedring
    min_delta = float(train_cfg.get("min_delta", 0.0))  # Minste forbedring i val-loss som må til for å telle

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW optimizer med weight decay
    loss_fn = nn.CrossEntropyLoss()  # Loss for neste-token-prediksjon

    best_val_loss = float("inf")  # Lagrer beste validation loss så langt
    best_train_loss = None  # Lagrer train loss ved samme step som ga beste val loss
    patience_counter = 0  # Teller hvor mange eval-runder på rad uten forbedring
    best_step = -1  # Lagrer hvilket step som ga beste modell

    if run is not None:  # Lagrer ekstra info i W&B summary for denne run-en
        wandb.summary["resolved_device"] = device  # Viser faktisk device brukt
        wandb.summary["tokenizer_type"] = tok_cfg["type"]  # Viser om vi brukte char eller bpe
        wandb.summary["tokenizer_path"] = tok_cfg.get("model_path", "N/A")  # Viser hvor tokenizer ligger
        wandb.summary["vocab_size"] = vocab_size  # Viser vocab-størrelsen
        wandb.summary["parameter_count"] = sum(p.numel() for p in model.parameters())  # Viser antall parametere i modellen
        wandb.summary["checkpoint_path"] = ckpt_cfg["save_path"]  # Viser hvor modellen lagres
        wandb.summary["weight_decay"] = weight_decay  # Viser hvilken weight decay som ble brukt i denne run-en
        wandb.summary["grad_clip"] = grad_clip  # Viser hvilken gradient clipping-verdi som ble brukt i denne run-en
        wandb.summary["early_stopping"] = use_early_stopping  # Viser om early stopping var aktiv
        wandb.summary["patience"] = patience  # Viser patience-verdi brukt i denne run-en
        wandb.summary["min_delta"] = min_delta  # Viser min_delta brukt i denne run-en
        wandb.summary["train_ratio"] = train_cfg["train_ratio"]  # Viser train split
        wandb.summary["val_ratio"] = train_cfg["val_ratio"]  # Viser val split
        wandb.summary["test_ratio"] = 1.0 - train_cfg["train_ratio"] - train_cfg["val_ratio"]  # Viser hvor stor del som er igjen til test
        wandb.summary["min_lr"] = float(train_cfg.get("min_lr", lr * 0.1))  # Viser minimum learning rate brukt av scheduler
        wandb.summary["warmup_iters"] = int(train_cfg.get("warmup_iters", 0))  # Viser antall warmup-steg brukt
        wandb.summary["lr_schedule"] = "linear_warmup_cosine_decay"  # Viser hvilken learning rate schedule som ble brukt

    for step in range(train_cfg["max_iters"]):  # Hovedloop for trening
        current_lr = get_lr(step, train_cfg)  # Beregner learning rate for dette steget med warmup + cosine decay
        for param_group in optimizer.param_groups:  # Oppdaterer learning rate i optimizer
            param_group["lr"] = current_lr  # Setter ny learning rate før forward/backward-pass

        x, y = get_batch(
            train_data,
            model_cfg["context_length"],
            train_cfg["batch_size"],
            device=device,
        )

        logits = model(x)
        B, T, C = logits.shape

        loss = loss_fn(logits.view(B * T, C), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:  # Klipper gradientene hvis grad_clip er satt større enn 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # Hindrer eksploderende gradients ved å klippe total gradientnorm

        optimizer.step()

        if step % train_cfg["eval_interval"] == 0:  # Kjører evaluering med faste intervaller under trening
            train_loss = estimate_loss(
                model,
                train_data,
                train_cfg["eval_iters"],
                model_cfg["context_length"],
                train_cfg["batch_size"],
                device,
            )  # Måler gjennomsnittlig train loss på tilfeldig utvalg fra train-splittet

            val_loss = estimate_loss(
                model,
                val_data,
                train_cfg["eval_iters"],
                model_cfg["context_length"],
                train_cfg["batch_size"],
                device,
            )  # Måler gjennomsnittlig val loss på valideringssplittet

            train_perplexity = math.exp(min(train_loss, 20))  # Regner om train loss til perplexity
            val_perplexity = math.exp(min(val_loss, 20))  # Regner om val loss til perplexity

            print(
                f"Step {step} | "
                f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                f"train ppl {train_perplexity:.2f} | val ppl {val_perplexity:.2f}"
            )

            input_ids = torch.tensor([tokenizer.encode(gen_cfg["prompt"])], dtype=torch.long).to(device)  # Gjør prompt om til token-IDer

            output = generate(
                model,
                input_ids,
                max_new_tokens=gen_cfg["max_new_tokens"],
                block_size=model_cfg["context_length"],
                temperature=gen_cfg["temperature"],
                top_k=gen_cfg["top_k"],
            )  # Genererer tekst med modellen slik den er akkurat nå

            sample_text = tokenizer.decode(output[0].tolist())  # Gjør genererte token-IDer om til lesbar tekst

            if val_loss < best_val_loss - min_delta:  # Sjekker om validation loss er forbedret nok til å telle som ny beste modell
                best_val_loss = val_loss  # Oppdaterer beste validation loss
                best_train_loss = train_loss  # Lagrer train loss ved samme step som beste val loss
                patience_counter = 0  # Nullstiller teller fordi modellen forbedret seg
                best_step = step  # Lagrer hvilket step som ga beste modell

                os.makedirs(ckpt_cfg["out_dir"], exist_ok=True)  # Lager mappe for checkpoints hvis den ikke finnes
                torch.save(model.state_dict(), ckpt_cfg["save_path"])  # Lagrer beste modellvekter fortløpende
                print(f"Ny beste modell lagret i {ckpt_cfg['save_path']}")  # Skriver ut at vi har lagret ny beste modell
            else:
                patience_counter += 1  # Øker teller hvis val-loss ikke forbedret seg nok

            if run is not None:
                samples_table.add_data(step, train_loss, val_loss, val_perplexity, sample_text)  # Legger til tekstsample og val-målinger i W&B-tabellen

            if run is not None:
                wandb.log({
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_perplexity": train_perplexity,
                    "val_perplexity": val_perplexity,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "generated_text": sample_text,
                    "best_val_loss": best_val_loss,  # Logger beste validation loss så langt
                    "patience_counter": patience_counter,  # Logger hvor mange eval-runder uten forbedring vi er på
                })

            if use_early_stopping and patience_counter >= patience:  # Stopper tidlig hvis modellen ikke har forbedret seg på mange eval-runder
                print(
                    f"\n Early stopping triggered!\n"
                    f"Step: {step}\n"
                    f"Beste val_loss: {best_val_loss:.4f} (step {best_step})\n"
                    f"Ingen forbedring siste {patience} eval-runder\n"
                )  # Skriver tydelig ut hvorfor treningen stoppet
                break

    if best_step == -1:  # Safety check hvis ingen modell ble lagret underveis
        os.makedirs(ckpt_cfg["out_dir"], exist_ok=True)  # Lager mappe for checkpoints
        torch.save(model.state_dict(), ckpt_cfg["save_path"])  # Lagrer modellvektene likevel
        print(f"Ingen best-checkpoint ble lagret underveis, lagret siste modell i {ckpt_cfg['save_path']}")  # Bekrefter fallback-lagring
    else:
        print(f"Beste modell ble lagret fra step {best_step} i {ckpt_cfg['save_path']}")  # Bekrefter hvilket step som ga beste modell

    final_model = ShakespeareModel(  # Lager en ny modellinstans med samme arkitektur
        vocab_size=vocab_size,  # Output-dimensjon må matche vocab
        embed_dim=model_cfg["d_model"],  # Embedding-dimensjon
        block_size=model_cfg["context_length"],  # Hvor lang kontekst modellen ser
        num_layers=model_cfg["n_layers"],  # Antall lag
        num_heads=model_cfg["n_heads"],  # Antall attention heads
        ff_mult=model_cfg["ff_mult"],  # Feed-forward multiplier
        dropout=model_cfg["dropout"],  # Dropout rate
    ).to(device)  # Flytter modellen til CPU/GPU

    final_model.load_state_dict(torch.load(ckpt_cfg["save_path"], map_location=device))  # Laster inn beste lagrede modell fra disk
    final_model.eval()  # Setter modellen i eval-modus

    final_test_loss = estimate_loss(
        final_model,
        test_data,
        train_cfg["eval_iters"],
        model_cfg["context_length"],
        train_cfg["batch_size"],
        device,
    )  # Evaluerer beste modell én gang på test-splittet helt til slutt

    final_test_perplexity = math.exp(min(final_test_loss, 20))  # Regner om test loss til endelig test perplexity

    print(f"Final test loss: {final_test_loss:.4f}")  # Skriver ut endelig test loss i terminal
    print(f"Final test perplexity: {final_test_perplexity:.2f}")  # Skriver ut endelig test perplexity i terminal

    final_prompt = gen_cfg["prompt"]  # Bruker samme prompt som i config
    final_input_ids = torch.tensor([tokenizer.encode(final_prompt)], dtype=torch.long).to(device)  # Gjør prompt om til token-IDer

    final_output = generate(  # Genererer tekst med beste lagrede modell
        final_model,
        final_input_ids,
        max_new_tokens=gen_cfg["max_new_tokens"],  # Hvor mange tokens som skal genereres
        block_size=model_cfg["context_length"],  # Hvor mye kontekst modellen bruker
        temperature=gen_cfg["temperature"],  # Bruker temperature fra config
        top_k=gen_cfg["top_k"],  # Bruker top_k fra config
    )

    final_text = tokenizer.decode(final_output[0].tolist())  # Gjør genererte token-IDer om til lesbar tekst

    if run is not None:  # Logger sluttresultater i samme W&B-run
        final_samples_table = wandb.Table(columns=["best_step", "prompt", "text"])  # Lager egen tabell for final eval-sample
        final_samples_table.add_data(best_step, final_prompt, final_text)  # Legger inn sluttgenereringen i tabellen

        final_results_table = wandb.Table(columns=[
            "best_step",
            "train_loss",
            "val_loss",
            "test_loss",
            "test_perplexity",
            "prompt",
            "generated_text"
        ])  # Lager en egen ryddig tabell med de viktigste sluttresultatene for run-en, inkludert train loss ved beste checkpoint

        final_results_table.add_data(
            best_step,
            best_train_loss,
            best_val_loss,
            final_test_loss,
            final_test_perplexity,
            final_prompt,
            final_text
        )  # Legger inn beste step, train loss ved beste checkpoint, beste val-loss, endelig test-måling og generert tekst

        wandb.log({
            "final_samples": final_samples_table,  # Logger sluttgenereringen som egen tabell
            "final_results": final_results_table,  # Logger en samlet tabell med sluttresultater
            "final_test_loss": final_test_loss,  # Logger endelig test loss som egen metrikk
            "final_test_perplexity": final_test_perplexity,  # Logger endelig test perplexity som egen metrikk
        })

        wandb.summary["best_step"] = best_step  # Lagrer hvilket step som ga beste modell i W&B summary
        wandb.summary["best_train_loss"] = best_train_loss  # Lagrer train loss ved beste checkpoint i W&B summary
        wandb.summary["best_val_loss"] = best_val_loss  # Lagrer beste validation loss i W&B summary
        wandb.summary["final_prompt"] = final_prompt  # Lagrer prompt brukt til sluttgenerering i W&B summary
        wandb.summary["final_generated_text"] = final_text  # Lagrer sluttgenereringen også i W&B summary
        wandb.summary["final_test_loss"] = final_test_loss  # Lagrer endelig test loss tydelig i W&B summary
        wandb.summary["final_test_perplexity"] = final_test_perplexity  # Lagrer endelig test perplexity tydelig i W&B summary

        wandb.log({"samples": samples_table})  # Logger hele samples-tabellen én gang til slutt
        run.finish()


if __name__ == "__main__":
    train()