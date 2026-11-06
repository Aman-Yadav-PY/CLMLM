import torch
import torch.nn as nn



class RotatoryPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 50000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        ix = torch.arange(0, d_model//2).unsqueeze(0)
        omega = 1e4 ** (-2 * ix/d_model)
        angles = pos * omega
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, x):
        T= x.shape[-2]
        cos, sin = self.cos[:T], self.sin[:T]

        X, Y = x[..., 0::2], x[..., 1::2] # x-ords, y-ords for every token
     
        Xn =  X * cos[:T] + Y * sin[:T] # [ cosQ sinQ ]  [X]
        Yn = -X * sin[:T] + Y * cos[:T] # [-sinQ cosQ ] [Y]
        
        out = torch.empty_like(x)
        out[..., 0::2] = Xn
        out[..., 1::2] = Yn

        return out
        

class GroupedQueryAttention(nn.Module):

    def __init__(self, block_size, d_model, heads, groups, dropout):
        super().__init__()
        
        self.hs = d_model // heads
        self.rps = heads // groups
        
        self.Wqs = nn.ModuleList([nn.Linear(d_model, self.hs) for _ in range(heads)])
        self.Wks = nn.ModuleList([nn.Linear(d_model, self.hs) for _ in range(groups)])
        self.Wvs = nn.ModuleList([nn.Linear(d_model, self.hs) for _ in range(groups)])
        self.rope = RotatoryPositionalEncoding(self.hs, block_size)
        self.Wo = nn.Linear(heads * self.hs, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # Batch, block_size, d_models (B, T, C)

        Q = self.rope(torch.stack([Wq(x) for Wq in self.Wqs]))
        K = torch.stack([Wk(x) for Wk in self.Wks])
        V = torch.stack([Wv(x) for Wv in self.Wvs])

        K = self.rope(torch.repeat_interleave(K, self.rps, dim=0))
        V = torch.repeat_interleave(V, self.rps, dim=0)

        score = Q @ K.transpose(-2, -1) / self.hs ** 0.5
        score = score.masked_fill(self.mask[:T,  :T]==0, float("-inf"))
        probs = F.softmax(score, dim=-1)
        out = probs @ V
        out = out.permute(1, 2, 0, 3).reshape(B, T, -1)
        out  = self.dropout(self.Wo(out))

        return out
        


class FeedForward(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)
        


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, groups, block_size, dropout):
        super().__init__()
        head_size = d_model // n_heads
        self.gqa = GroupedQueryAttention(block_size, d_model, n_heads, groups, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.gqa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DecoderTransformerModel(nn.Module):

    def __init__(self, vocab_size, block_size, d_model, heads, groups, n_layers, dropout):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, heads, groups, block_size, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # print(isinstance(module,nn.Linear))
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.normal_(module.bias, mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, target = None):
        out = self.embeddings(x)
        out = self.blocks(out)
        logits = self.linear(self.ln(out))
        # B, T, C = logits.shape

        # loss = None
        # accuracy = None
        # if target:
        #     loss = F.cross_entropy(logits.view(B * T, -1), target.view(-1))
        #     accuracy = (logits.argmax(-1) == ys).float().mean()

            
        return logits #, loss, accuracy
