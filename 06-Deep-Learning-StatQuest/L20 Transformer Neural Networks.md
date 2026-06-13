## Transformer Neural Networks

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Why Transformers?

### Motivation and Intuition

RNNs process tokens one at a time — inherently sequential and not parallelizable. For long sequences, information from early tokens gets diluted. **Transformers** (Vaswani et al., 2017) solve both problems with **self-attention**: every token directly attends to every other token in parallel.

| Problem | RNN | Transformer |
| :--- | :--- | :--- |
| Parallelism | Sequential (one token at a time) | Fully parallel over sequence |
| Long-range dependencies | Degrades with distance | Direct O(1) path between any tokens |
| Position info | Implicit (order of processing) | Explicit (positional encoding) |

---

## 2. Transformer Architecture (Encoder + Decoder)

The full Transformer has two stacks:

### Encoder
```
Input Embedding + Positional Encoding
  -> [Multi-Head Self-Attention -> Add & Norm -> FFN -> Add & Norm] x N
  -> Contextual Representations
```

### Decoder
```
Output Embedding + Positional Encoding
  -> [Masked Multi-Head Self-Attention -> Add & Norm
      -> Cross-Attention (encoder outputs) -> Add & Norm
      -> FFN -> Add & Norm] x N
  -> Linear -> Softmax -> Output Probabilities
```

| Component | Purpose |
| :--- | :--- |
| Self-attention | Each token attends to all tokens in the same sequence |
| Cross-attention | Decoder attends to encoder outputs (encoder-decoder attention) |
| Masked self-attention | Decoder can only attend to previous positions (causal mask) |
| Feed-forward network (FFN) | Two linear layers with ReLU/GELU: $FFN(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$ |
| Add & Norm | Residual connection + layer normalization (stabilizes training) |

---

## 3. Self-Attention

### Motivation and Intuition

Self-attention computes relationships between all pairs of positions in a sequence. Each token projects into **Query**, **Key**, and **Value** vectors. For each token, we compute how much to attend to every other token.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q$ | Query matrix — each token asks "what should I look for?" | Determines what each token finds relevant |
| $K$ | Key matrix — each token broadcasts "what I contain" | Used to match against queries |
| $V$ | Value matrix — each token's actual content | The information aggregated by attention |
| $Q K^T$ | Pairwise dot products between all queries and keys | Raw attention scores matrix |
| $\sqrt{d_k}$ | Scaling factor | Prevents softmax saturation for large dimensions |

---

## 4. Multi-Head Attention

### Motivation and Intuition

Instead of one attention function, run $h$ parallel heads with different learned projections. Each head learns different types of relationships (syntactic, semantic, positional).

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $W_i^Q, W_i^K, W_i^V$ | Learned projection matrices for head $i$ | Project input into different subspaces for each head |
| $W^O$ | Output projection matrix | Projects concatenated head outputs back to model dimension |
| $h$ | Number of heads (typically 8-16) | Each head captures different relationship types |

---

## 5. Positional Encoding

### Motivation and Intuition

Self-attention is permutation-invariant — without position information, "the cat sat" and "sat the cat" produce identical representations. Positional encoding injects order information.

Sinusoidal encodings (original paper):

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $pos$ | Token position in sequence (0, 1, 2, ...) | Gives model awareness of word order |
| $i$ | Dimension index within the embedding | Different dims oscillate at different frequencies |
| $d$ | Model dimension | Total width of positional encoding vector |
| $10000^{2i/d}$ | Frequency scaling | Lower dims = fast oscillation (local position), higher dims = slow (global position) |

---

## 6. Python Code: Multi-Head Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # batch, seq_len, embed_size

        # Project to Q, K, V and reshape for multi-head
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Concatenate heads and project
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Test
emb = nn.Embedding(100, 64)
pe = PositionalEncoding(64)
attn = MultiHeadSelfAttention(embed_size=64, num_heads=8)

tokens = torch.randint(0, 100, (2, 10))  # batch=2, seq_len=10
x = pe(emb(tokens))
out = attn(x)
print(f"Input: {x.shape}, Output: {out.shape}")
```

---

> **Check your intuition:** Self-attention has O(n^2) complexity in sequence length — every token attends to every other. For a 1000-token sequence, that's 1 million attention computations. Why is this acceptable for Transformers despite the quadratic cost? How do modern models (like those using Flash Attention) reduce the memory overhead?

---

## Prerequisites and Further Reading

- [StatQuest: Attention Mechanism](https://www.youtube.com/watch?v=PSs6nxngL6k)
- [StatQuest: Word Embeddings](https://www.youtube.com/watch?v=D-ekhWxubTs)
- Vaswani et al., "Attention Is All You Need" (2017)
- The Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html
