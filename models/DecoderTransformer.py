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
        

class SelfAttentionHead(nn.Module):

    def __init__(self, d_model, head_size, block_size, dropout):
        super().__init__()
        self.k = nn.Linear(d_model, head_size, bias=False)
        self.q = nn.Linear(d_model, head_size, bias=False)
        self.v = nn.Linear(d_model, head_size, bias=False)
        self.rope = RotatoryPositionalEncoding(head_size, block_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape # B, T, C

        query = self.rope(self.q(x)) #B, T, hs
        key = self.rope(self.k(x))
        value = self.v(x)
        
        out = query @ key.transpose(-2, -1) #B, T, hs @ B, hs, T --> B, T, T
        out = out * query.shape[-1] ** -0.5
        out = out.masked_fill(self.tril[:T, :T] == 0, float("-inf")) #masking
        out = F.softmax(out, dim=-1)
        out = self.dropout(out)
        out = out @ value #B, T, hs

        return out


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, heads, d_model, head_size, block_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionHead(d_model, head_size, block_size, dropout) for _ in range(heads)])
        self.projection = nn.Linear(heads * head_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([layer(x) for layer in self.layers], dim=-1)
        out = self.dropout(self.projection(out))
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

    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        head_size = d_model // n_heads
        self.mhsa = MultiHeadSelfAttention(n_heads, d_model, head_size, block_size, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DecoderTransformerModel(nn.Module):

    def __init__(self, vocab_size, block_size, d_model, heads, n_layers, dropout):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, heads, block_size, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # print(isinstance(module,nn.Linear))
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.normal_(module.bias) * 0.1

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, target = None):
        out = self.embeddings(x)
        out = self.blocks(out)
        logits = self.linear(self.ln(out))
        
        return logits
