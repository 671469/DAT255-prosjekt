#Skal bygge vokabular, encode, decode

def build_tokenizer(text):
    chars = sorted(list(set(text))) #Alle unike tegn i teksten
    stoi = {ch: i for i, ch in enumerate(chars)} #string-to-index (tegn til tall)
    itos = {i: ch for i, ch in enumerate(chars)} #index-to-string (tall til tegn)
    return stoi, itos

def encode(text, stoi): #Gjør tekst om til tall
    return [stoi[ch] for ch in text]

def decode(ids, itos): #Gjør tall til tekst
    return "".join([itos[i] for i in ids])

