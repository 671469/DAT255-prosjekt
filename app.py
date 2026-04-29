import gradio as gr
import torch
from src.model.model import ShakespeareModel
from src.data.tokenizer import BPETokenizer
from src.training.eval import generate

device = "cpu"
tokenizer = BPETokenizer.load("models/bpe_tokenizer_train_only.json")

model = ShakespeareModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=512,      # d_model
    block_size=384,     # context_length
    num_layers=8,       # n_layers
    num_heads=8,        # n_heads
    ff_mult=4,
    dropout=0.0
).to(device)

model.load_state_dict(torch.load(
    "models/shakespeare_model.pt",
    map_location=device
))
model.eval()

def chat(message, history):
    input_ids = torch.tensor(
        [tokenizer.encode(f"Thou art {message}\n")],
        dtype=torch.long
    ).to(device)

    output = generate(
        model, input_ids, 
        max_new_tokens=300,
        block_size=384,
        temperature=0.65,
        top_k=30,
        repetition_penalty=1.25,
        no_repeat_ngram_size=4,
    )
    return tokenizer.decode(output[0].tolist())

demo = gr.ChatInterface(
    fn=chat,
    title="ShakespeareGPT",
    description="A miniture transformer trained on Shakespeare.",
    examples=[
        "What makes a good king?",
        "What is love?",
        "Tell me about death",
    ],
    theme=gr.themes.Soft(),
)

demo.launch()