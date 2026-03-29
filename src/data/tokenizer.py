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
    
    
    # !!!!!!! Først stabil baseline char-level tokenizer ovenfor. Senere forbedre til SenterPiece/BPE (Byte-Pair Encoding) !!!!!!!!
    
    #Endring av tokenizer påvirker flere andre filer som må endres
    #Bytt til BPE etter 3-5 runs hvor alt funker, med config endringer. 
    
    
    #når alt kjører uten krasj, loss går ned, modellen lagres, eval kjører, får val loss + perplexity, får generert tekst, output gir mening,
    #samme config gir samme type resultat, pipeline er stabil. = baseline/referansepunkt.
    #så trene den et par ganger, med noen endringer i .YAML for sammenligningsgrunnlag. Feks 1 baseline-run for å se at alt fungerer, 
    #så 1-2 runs med endret modellstørrelse, så 1-2 runs med endret context length eller learning rate.
    
    #Så endre tokenizer. 
    
    #eksempel til rapporten feks:
    #“We first establish a baseline model using a simple character-level tokenizer. 
    # This allows us to validate the training pipeline and obtain a reference for later improvements. 
    # Once a stable baseline is achieved, we introduce a subword tokenizer (BPE) to investigate whether improved tokenization leads to 
    # better performance.”

