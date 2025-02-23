import torch 
import torch.nn as nn
from architecture.transformer.transformer import *

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.transform_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_length = x.shape
        x = self.tok_emb(x) + self.pos_emb(torch.arange(seq_length, device=x.device))
        x = self.dropout(x)
        x = self.transform_blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits

def inference(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        with torch.no_grad():
            input = idx[:, -context_size:]
            output = model(input)[:, -1, :] # only want the last time step

        next_token = torch.argmax(output, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=-1)   
    return idx

def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'|<endoftext>|'})
    encoded_tensor = torch.tensor(encoded).unsqueez(0)
    return encoded_tensor

def token_to_text(token, tokenizer):
    token = token.squeeze(0).tolist()
    text = tokenizer.decode(token)
    return text