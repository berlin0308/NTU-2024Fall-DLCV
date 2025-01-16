import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora  # Make sure loralib is installed


class AttentionLoRA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=cfg.rank)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=cfg.rank)
        
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size()  # batch, context, embedding
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class BlockLoRA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = AttentionLoRA(cfg)
        
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=cfg.rank)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=cfg.rank))
        ]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderLoRA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte=lora.Embedding(cfg.vocab_size, cfg.n_embd, r=cfg.rank),
            wpe=lora.Embedding(cfg.block_size, cfg.n_embd, r=cfg.rank),
            h=nn.Sequential(*[BlockLoRA(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=cfg.rank)
        self.transformer.wte.weight = self.lm_head.weight

        if self.cfg.checkpoint is not None:
            print("Load decoder weights")
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = ['.c_attn.weight', '.c_fc.weight', '.c_proj.weight']
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, encoder_feature: Tensor):

        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size - encoder_feature.size(1)))
        pos = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)

        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        
        x = torch.cat((encoder_feature, x), dim=1)

        for block in self.transformer.h:
            x = block(x)

        x = self.lm_head(self.transformer.ln_f(x))
        return x