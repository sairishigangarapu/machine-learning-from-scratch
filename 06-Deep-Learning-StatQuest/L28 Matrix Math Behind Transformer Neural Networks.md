## L28: Matrix Math Behind Transformer Neural Networks

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. From Words to Matrices

### Motivation and Intuition

Transformers process sequences of tokens. Each token is mapped to a **dense vector** via an embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d_{\text{model}}}$. The entire input sequence becomes a matrix $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}}$ where $N$ is the sequence length.

Position information is added via positional encodings (sinusoidal or learned):

$$
\mathbf{X} = \text{TokenEmbeddings} + \text{PositionalEncodings}
$$

In PyTorch:

```python
import torch
import torch.nn as nn

vocab_size, d_model, seq_len = 1000, 512, 64
embed = nn.Embedding(vocab_size, d_model)
pos_enc = nn.Embedding(seq_len, d_model)

tokens = torch.randint(0, vocab_size, (1, seq_len))
positions = torch.arange(seq_len).unsqueeze(0)
X = embed(tokens) + pos_enc(positions)   # (1, 64, 512)
```

---

## 2. Query, Key, and Value Projections

### Motivation and Intuition

Each token produces three vectors: a **query** (what am I looking for?), a **key** (what do I contain?), and a **value** (what information do I pass along?). These are computed by multiplying the input $\mathbf{X}$ with learned weight matrices.

For each attention head $h$:

$$
\mathbf{Q}_h = \mathbf{X} \mathbf{W}_h^Q \quad\quad
\mathbf{K}_h = \mathbf{X} \mathbf{W}_h^K \quad\quad
\mathbf{V}_h = \mathbf{X} \mathbf{W}_h^V
$$

Where:
- $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}}$
- $\mathbf{W}_h^Q, \mathbf{W}_h^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $\mathbf{W}_h^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $\mathbf{Q}_h, \mathbf{K}_h \in \mathbb{R}^{N \times d_k}$, $\mathbf{V}_h \in \mathbb{R}^{N \times d_v}$

Typically $d_k = d_v = d_{\text{model}} / H$ where $H$ is the number of heads.

```python
d_model, num_heads, d_k = 512, 8, 64

W_Q = nn.Linear(d_model, d_k, bias=False)
W_K = nn.Linear(d_model, d_k, bias=False)
W_V = nn.Linear(d_model, d_k, bias=False)

Q = W_Q(X)   # (1, 64, 64)
K = W_K(X)   # (1, 64, 64)
V = W_V(X)   # (1, 64, 64)
```

---

## 3. Attention Score Matrix

### Motivation and Intuition

The attention score matrix measures compatibility between every query and every key. The dot product $\mathbf{Q}\mathbf{K}^\top$ gives a raw score matrix. Scaling by $\sqrt{d_k}$ prevents large values that push softmax into extreme gradients.

$$
\text{Scores} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}
\quad\quad
\mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}, \text{dim}=-1 \right)
$$

- $\text{Scores} \in \mathbb{R}^{N \times N}$ — each element $(i, j)$ is attention from token $i$ to token $j$
- Softmax is applied row-wise so each row sums to 1

```python
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)   # (1, 64, 64)
A = torch.softmax(scores, dim=-1)                  # (1, 64, 64)
```

For **causal (masked) self-attention** (decoder-only models like GPT), future positions are masked with $-\infty$ before softmax:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
A = torch.softmax(scores, dim=-1)
```

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{Q} \mathbf{K}^\top$ | Pairwise dot products between queries and keys | Captures compatibility/alignment |
| $\sqrt{d_k}$ | Scaling factor | Keeps variance stable, prevents saturating softmax |
| $\text{softmax}$ | Row-wise normalization | Converts scores into a probability distribution |
| $\mathbf{A}$ | Attention weight matrix | Each row sums to 1; how much each token attends to others |

---

## 4. Weighted Sum (Context Vectors)

### Motivation and Intuition

Each token's output is a weighted sum of all value vectors, weighted by the attention probabilities. Tokens gather information from the entire sequence.

$$
\text{head}_h = \mathbf{A}_h \mathbf{V}_h
$$

- $\mathbf{A}_h \in \mathbb{R}^{N \times N}$, $\mathbf{V}_h \in \mathbb{R}^{N \times d_v}$
- $\text{head}_h \in \mathbb{R}^{N \times d_v}$

```python
head = A @ V   # (1, 64, 64)
```

---

## 5. Multi-Head Concatenation and Projection

### Motivation and Intuition

Multiple attention heads run in parallel, each focusing on different relationships (e.g., syntax, semantics, position). Their outputs are concatenated and linearly projected back to $d_{\text{model}}$.

$$
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) \, \mathbf{W}^O
$$

Where:
- $\text{Concat} \in \mathbb{R}^{N \times (H \cdot d_v)}$
- $\mathbf{W}^O \in \mathbb{R}^{(H \cdot d_v) \times d_{\text{model}}}$
- Output: $\mathbb{R}^{N \times d_{\text{model}}}$

With $H = 8$ and $d_v = 64$, concatenation gives $8 \times 64 = 512 = d_{\text{model}}$.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        B, N, D = X.shape
        Q = self.W_Q(X).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(X).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(X).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        A = torch.softmax(scores, dim=-1)
        context = A @ V                           # (B, H, N, d_k)

        out = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)
```

---

## 6. Full Transformer Block

Each transformer block is: multi-head attention → residual add & layer norm → feed-forward → residual add & layer norm.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        X = X + self.attention(self.norm1(X))
        X = X + self.ff(self.norm2(X))
        return X
```

---

> **Check your intuition:** Why is the score matrix scaled by $\sqrt{d_k}$? What would happen to the softmax outputs if $d_k = 512$ and the scores were unscaled?

---

## Prerequisites and Further Reading

- **StatQuest:** Attention Mechanism (L19), Transformer Neural Networks (L20), Essential Matrix Algebra (L27)
- **Paper:** Vaswani et al., "Attention Is All You Need" (2017)
- **Concepts:** Softmax, matrix multiplication, residual connections, layer normalization
