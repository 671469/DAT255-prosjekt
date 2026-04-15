# Skal bygge vokabular, encode, decode

from pathlib import Path  # brukes for å håndtere filstier på en ryddig måte
from tokenizers import Tokenizer  # hovedklasse for tokenizer fra HF
from tokenizers.models import BPE  # BPE-modell (subword-tokenizer)
from tokenizers.trainers import BpeTrainer  # trainer for å lære BPE-vokabular
from tokenizers.pre_tokenizers import ByteLevel  # splitter tekst i byte-level tokens
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # decoder tilbake til tekst

# Char Tokenizer baseline
class CharTokenizer:

    def __init__(self, text):
        chars = sorted(set(text))  # finner alle unike tegn i teksten
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # map tegn -> index
        self.itos = {i: ch for i, ch in enumerate(chars)}  # map index -> tegn
        self.vocab_size = len(chars)  # antall unike tegn
        print(f"Character vocabulary size: {self.vocab_size}")  # debug-print

    def encode(self, text):
        try:
            return [self.stoi[ch] for ch in text]  # konverter hvert tegn til ID
        except KeyError as e:
            raise ValueError(f"Unknown character found during encoding: {e}")  # feil hvis ukjent tegn

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)  # konverter IDs tilbake til tekst

# BPE Tokenzer
class BPETokenizer:

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer  # lagrer selve HF-tokenizeren
        self.vocab_size = tokenizer.get_vocab_size()  # antall tokens i vokabular
        print(f"BPE vocabulary size: {self.vocab_size}")  # debug-print

    # Trener BPE tokenizer på teksten og lagrer
    @classmethod
    def train(cls, text: str, save_path: str, vocab_size: int = 2000, min_frequency: int = 2):

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # lager ny BPE-tokenizer med unknown-token

        tokenizer.pre_tokenizer = ByteLevel()  # splitter tekst i byte-level tokens før BPE
        tokenizer.decoder = ByteLevelDecoder()  # gjør decoding tilbake til tekst korrekt

        trainer = BpeTrainer(
            vocab_size=vocab_size,  # hvor mange tokens vi vil ha i vokabularet
            min_frequency=min_frequency,  # hvor ofte et token må forekomme for å bli med
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],  # spesielle tokens
        )

        save_path = Path(save_path)  # gjør om til Path-objekt
        save_path.parent.mkdir(parents=True, exist_ok=True)  # lager mappe hvis den ikke finnes

        temp_text_path = save_path.parent / "tokenizer_training_text.txt"  # midlertidig fil for trening
        temp_text_path.write_text(text, encoding="utf-8")  # skriver tekst til fil

        tokenizer.train([str(temp_text_path)], trainer)  # trener tokenizer på teksten
        tokenizer.save(str(save_path))  # lagrer tokenizer til disk

        print(f"Saved BPE tokenizer to {save_path}")  # debug-print
        return cls(tokenizer)  # returnerer ferdig tokenizer-objekt

    @classmethod
    def load(cls, path: str):
        tokenizer = Tokenizer.from_file(path)  # laster tokenizer fra fil
        return cls(tokenizer)  # pakker i vår klasse

    def encode(self, text: str):
        encoding = self.tokenizer.encode(text)  # encoder tekst til tokens
        return encoding.ids  # returnerer bare ID-listen

    def decode(self, ids):
        return self.tokenizer.decode(ids)  # gjør tokens tilbake til tekst


if __name__ == "__main__":
    from src.data.data import load_text  # importerer funksjon for å laste tekst

    text = load_text()  # laster hele datasettet som tekst

    print("Testing CharTokenizer...")  # test av char tokenizer
    char_tokenizer = CharTokenizer(text)  # lager tokenizer
    sample = text[:100]  # tar liten sample av tekst
    encoded = char_tokenizer.encode(sample)  # encoder sample
    decoded = char_tokenizer.decode(encoded)  # decoder tilbake

    print("\nChar example text:", repr(sample))  # viser original tekst
    print("Char encoded:", encoded[:30], "...")  # viser første tokens
    print("Char decoded:", repr(decoded))  # viser rekonstruert tekst

    print("\nTesting BPETokenizer...")  # test av BPE tokenizer
    bpe_tokenizer = BPETokenizer.train(
        text=text,  # tekst å trene på
        save_path="models/bpe_tokenizer.json",  # hvor tokenizer lagres
        vocab_size=2000,  # størrelse på vokabular
        min_frequency=2,  # min frekvens for tokens
    )

    bpe_encoded = bpe_tokenizer.encode(sample)  # encoder sample
    bpe_decoded = bpe_tokenizer.decode(bpe_encoded)  # decoder tilbake

    print("\nBPE example text:", repr(sample))  # original tekst
    print("BPE encoded:", bpe_encoded[:30], "...")  # første tokens
    print("BPE decoded:", repr(bpe_decoded))  # rekonstruert tekst
