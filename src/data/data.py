import os #Importerer os-modul for å jobbbe med filer og mapper
import re #importerer re-modul for regulære uttrykk
import requests #importer requests, laste ned data via HTTP

#Download -> cleaning

URL = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt" #URL til datasettet

def download_shakespeare(path="data/shakespeare_raw.txt"): #Lager "data"-mappe
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(path): #Sjekker om filen allerede er lastet ned 
        r = requests.get(URL) #Sender en HTTP GET-request til URL'en
        r.raise_for_status() #kaster feil om nedlastningen ikke funket
        with open(path, "w", encoding="utf-8") as f: #Åpner filen der vi vil lagre datasettet
            f.write(r.text) #Skriver hele tekstinnholder fra responsen til filen
    return path #Returnerer filstien til datasettet

def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n") #Normaliserer linjeskift

 # fjerne start-delen hvis vi finner den, kutter vi bort alt før denne
    start = text.find("1609")
    if start != -1:
        text = text[start:]
 
    # fjern lisensblokker som dukker opp i teksten og erstatter med tom streng
    text = re.sub(
        r"<<.*?>>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE, #DOTALL lar .* matche linjeskift
    )

    # fjern all slutt-tekst etter "End of this Etext"
    text = re.sub(
        r"End of this Etext.*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # rydde litt i tomrom. Maks 2 tomme linjer etter hverandre
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() #Fjerner whitespace i starten og sluten av teksten

def load_text():
    raw_path = download_shakespeare() #Her kalles nedlasingsfunksjoner

    with open(raw_path, "r", encoding="utf-8") as f: #Åpner råfilen for lesing, "r" betyr read mode
        text = f.read() #Filen leses inn som 1 stor tekststreng og lagres i text

    text = clean_text(text) #Her sendes råteksten inn i rensefunksjonen

    with open("data/shakespeare_clean.txt", "w", encoding="utf-8") as f: #Åpner ny fil som skal inneholde den rensede teksten
        f.write(text) #Den rensede teksten lagres i "Data/shakespeare_clean.txt"

    return text #Returnerer den rensede teksten slik at den kan brukes videre i koden

if __name__ == "__main__": #Kjør kode om filen kjøres
    text = load_text() #Load datasett
