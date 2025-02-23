import torch
import torch.nn as nn
from architecture.transformer.attention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        varirance = torch.var(x, dim=-1, keepdim=True)
        x_norm = (x - mean)/ torch.sqrt(varirance + self.eps)
        return x_norm * self.scale + self.shift        
    
class GELU(nn.Module):
    """
    Implenmentation of a heaper approximation of Gaussian cumulative distribution
    """
    def __init__(self):
        super().__init__()
        self.GELU_CONS = 0.044715
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2/ torch.pi)) * (x + self.GELU_CONS * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context"],
            num_heads=cfg["n_head"],
            dropout=cfg["drop_rate"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        res = x
        x = self.dropout(self.attn(self.norm1(x)))
        x += res
        
        res = x
        x = self.dropout(self.ff(self.norm2(x)))
        x += res

        return x

