import os
import torch
import urllib.request

SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"
DATA_PATH = "data/shakespeare.txt"

def download_shakespeare():
    if os.path.exists(DATA_PATH):
        print("Data file exists, skipping the process")
        return
    os.makedirs(name="data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL,DATA_PATH)


class CharacterTokenizer:
    def __init__(self, text: str):
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        self.char_to_id = {}

        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index

        self.id_to_char = {}

        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char

        print(f"Vocab Size: {self.vocab_size}")
        print(f"Characters: {''.join(self.characters)}")

    def encode(self, text:str)->list:
        encoded_text = []
        for char in text:
            encoded_text.append(self.char_to_id[char])

        return encoded_text

    def decode(self, ids:list)->str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        decoded_text = []
        for id in ids:
            decoded_text.append(self.id_to_char[id])

        return ''.join(decoded_text)




if __name__ == "__main__":
    download_shakespeare()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharacterTokenizer(text)
    print(tokenizer.encode("Hasan Faruk Aktas"))

    # encoded "Hasan Faruk Aktas" str -> [20, 39, 57, 39, 52, 1, 18, 39, 56, 59, 49, 1, 13, 49, 58, 39, 57]

    print(tokenizer.decode([20, 39, 57, 39, 52, 1, 18, 39, 56, 59, 49, 1, 13, 49, 58, 39, 57]))

