from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0

        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)

        self.c_proj = nn.Linear(config.n_emb, config.n_emb)

        self.n_head = config.n_head
        self.n_embd = config.n_emb

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int  = 6
    n_head: int  = 6
    n_emb: int  = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb),
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

    