# Skal bygge vokabular, encode, decode

class CharTokenizer:  # Klasse som håndterer tokenizering (tekst <-> tall)
    def __init__(self, text):  # Kjøres når vi lager tokenizer-objekt
        chars = sorted(set(text))  # Finn unike tegn og sorter de

        self.stoi = {ch: i for i, ch in enumerate(chars)}  # tegn -> indeks
        self.itos = {i: ch for i, ch in enumerate(chars)}  # indeks -> tegn

        self.vocab_size = len(chars)  # antall unike tegn (vocab size)

        print(f"Vocabulary size: {self.vocab_size}")  # debug/info

    def encode(self, text):  # tekst -> liste med tall
        try:
            return [self.stoi[ch] for ch in text]  # slå opp hvert tegn i stoi
        except KeyError as e:
            raise ValueError(f"Unknown character found during encoding: {e}")  # feil hvis tegn mangler

    def decode(self, ids):  # liste med tall -> tekst
        return "".join(self.itos[i] for i in ids)  # slå opp hvert tall og join til string


if __name__ == "__main__":  # kjører bare hvis filen startes direkte
    from src.data.data import load_text  # import her for å unngå circular imports

    text = load_text()  # last inn renset tekst

    tokenizer = CharTokenizer(text)  # bygg tokenizer fra teksten

    sample = text[:30]  # ta en liten bit av teksten for test

    encoded = tokenizer.encode(sample)  # tekst -> tall
    decoded = tokenizer.decode(encoded)  # tall -> tekst

    print("\nExample text:", repr(sample))  # vis original
    print("Encoded:", encoded)  # vis token IDs
    print("Decoded:", repr(decoded))  # skal være lik original