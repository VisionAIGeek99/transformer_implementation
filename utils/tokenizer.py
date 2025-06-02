import json

class CharTokenizer:
    def __init__(self, text:str):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, ids: list[int]) -> list[str]:
        return ''.join([self.itos[i] for i in ids])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "stoi": self.stoi,
                "itos": self.itos
            }, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self.stoi)


def main():
    txt_path = "/home/kmw2622/transformer/data/tinyshakespeare.txt"
    tokenizer = CharTokenizer(open(txt_path).read())

    vocab_save_path = "/home/kmw2622/transformer/saved/tokenizer/vocab.json"
    # tokenizer.save(vocab_save_path)
    # tokenizer.load(vocab_save_path)

if __name__ == "__main__":
    main()
