# Character-Level Masked Language Model

A decoder-only Transformer trained at the **character level** with causal (autoregressive) attention over Tiny Shakespear dataset

## Architecture

| Component | Detail |
|-----------|--------|
| Attention | **Grouped Query Attention (GQA)** — fewer KV heads shared across query groups for reduced KV-cache memory and faster inference |
| Positional Encoding | **Rotary Position Embeddings (RoPE)** — encodes relative position via per-layer rotations of Q/K vectors |
| Masking | Causal (lower-triangular) mask — each position attends only to preceding tokens |

## Training

- Objective: next-character prediction (cross-entropy)
- Vocabulary: character-level (letters, digits, punctuation, whitespace)

## Inference

- **Context window:** 128 tokens (127 context tokens → predict the 128th)
- **Decoding:** Temperature-scaled logits → Top-K truncation → `torch.multinomial` sampling

```python
logits = logits / temperature
topk_logits, topk_indices = torch.topk(logits, k)
probs = F.softmax(topk_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)   
