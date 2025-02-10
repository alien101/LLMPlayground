import re
import tiktoken 

UNKOWN = "<|unk|>"
EOF = "<|endoftext|>"

class SimpleTokenizerV1:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = self._init_vocab()
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} # inverse mapping of vocab

    def _init_vocab(self):
        import urllib.request
        import os
        
        file_path = "the-verdict.txt"
        if not os.path.exists(file_path):
            url = (
                "https://raw.githubusercontent.com/rasbt/"
                "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
                "the-verdict.txt"
            )    
            urllib.request.urlretrieve(url, file_path)  

        with open ("the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Tokenizing the text
        tokens = re.split(r"\s|[\"()_:;,.!]|--", raw_text)
        tokens = [i.strip() for i in tokens if i.strip()] # remove whitespace
        words = sorted(set(tokens)) # disregard duplicas
        words.extend([UNKOWN, EOF]) # vocab size of 1,130
        # Generating token ID
        vocab = {token:integer for integer, token in enumerate(words)}

        return vocab

    def encode(self, text):
        preprocessed = re.split(r"\s|[\"()_:;,.!]|--", text)
        preprocessed = [i.strip() for i in preprocessed if i.strip()]
        ids = [self.str_to_int[s] if s in self.str_to_int else self.str_to_int[UNKOWN] for s in preprocessed]

        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # remove spaces before special characters
        text = re.sub(r"\s+([()_:;,.!])", r"\1", text) 
        return text


if __name__ == "__main__":
    tokenizer = SimpleTokenizerV1()
    text = """It's the last he painted, you know, " Mrs. Gisburn  said with pride nerissa"""
    ids = tokenizer.encode(text)
    back = tokenizer.decode(ids)
    # bpe tokenizer uses byte pair encoding algorithm to break down unknown word into known subwords to tokenize, 
    # so when decoder it is able to regenerate the text even if work not in its vocab list
    bpe_tokenizer = tiktoken.get_encoding("gpt2") # vocab size of 50,257
    text = "Akwirw ier"
    integers = bpe_tokenizer.encode(text) 
    bpe_tokenizer.decode(integers)