# Transformers & Self-Attention: "Attention Is All You Need"

## 1. Why Transformers?

RNNs process tokens **one at a time** — this is inherently sequential and can't be parallelized. For long sequences, information from early tokens gets diluted by the time it reaches later tokens.

**Transformers** (Vaswani et al., 2017) solve both problems with **self-attention**: every token directly looks at every other token, in parallel, in a single operation.

> This is the architecture behind GPT, BERT, Claude, and virtually all modern LLMs.

---

## 2. The Core Idea: Self-Attention

Given a sequence of tokens, self-attention computes a **weighted sum** of all tokens for each token, where the weights are learned based on relevance.

### Query, Key, Value

Each token is projected into three vectors:

| Vector | Analogy | Role |
| :--- | :--- | :--- |
| **Query** ($\mathbf{q}$) | "What am I looking for?" | The current token's question |
| **Key** ($\mathbf{k}$) | "What do I contain?" | Every token's label |
| **Value** ($\mathbf{v}$) | "What do I actually provide?" | The information to aggregate |

### Scaled Dot-Product Attention

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{Q}$ | Query matrix (sequence of query vectors) | Each row is a token asking "what should I attend to?" |
| $\mathbf{K}$ | Key matrix (sequence of key vectors) | Each row is a token broadcasting "what I contain" |
| $\mathbf{V}$ | Value matrix (sequence of value vectors) | Each row is a token's actual information to be aggregated |
| $\mathbf{Q}\mathbf{K}^T$ | Raw attention scores matrix | Dot product of every query with every key — measures pairwise similarity |
| $d_k$ | Dimension of each key vector | Used for scaling — prevents dot products from growing too large with dimension |
| $\sqrt{d_k}$ | Scaling factor | Without this, for large $d_k$, the softmax would receive very large values, causing gradients to vanish |
| $\text{softmax}(\cdot)$ | Softmax normalization (row-wise) | Converts raw scores to probabilities that sum to 1 — each token distributes its "attention" across all others |
| $\times \mathbf{V}$ | Weighted sum of values | Each token's output is a weighted combination of all values, weighted by attention |

| Step | What Happens |
| :--- | :--- |
| $\mathbf{Q}\mathbf{K}^T$ | Compute similarity between every query and every key → attention scores |
| $/ \sqrt{d_k}$ | Scale to prevent softmax saturation (gradient vanishing) |
| $\text{softmax}(\cdot)$ | Normalize scores to probabilities (rows sum to 1) |
| $\times \mathbf{V}$ | Weighted sum of values → output |

**Intuition:** If $\mathbf{q}_i$ is similar to $\mathbf{k}_j$, token $i$ "pays attention" to token $j$'s value.

---

## 3. Multi-Head Attention

Instead of one attention function, we run **$h$ parallel attention heads**, each with different learned projections, then concatenate and project:

$$
\begin{aligned}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O \\
\text{head}_i &= \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{Q}$ | Query matrix | What each token is "looking for" — derived from input embeddings |
| $\mathbf{K}$ | Key matrix | What each token "offers" — used to compute compatibility with queries |
| $\mathbf{V}$ | Value matrix | What each token actually "contains" — the information to be aggregated |
| $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V$ | Learned projection matrices for head $i$ | Transform the input into query/key/value spaces specific to this head |
| $\mathbf{W}^O$ | Output projection matrix | Projects the concatenated head outputs back to model dimension |
| $\text{head}_i$ | Output of attention head $i$ | One head's perspective on the relationships between tokens |
| $h$ | Number of attention heads | More heads = more diverse relationship types captured |

> Each head learns different **types of relationships** — syntactic, semantic, positional, etc.

---

## 4. Transformer Architecture

### Encoder (used in BERT)
```
Input Tokens → Embedding + Positional Encoding
 ↓
[Multi-Head Self-Attention → Add & Norm → Feed Forward → Add & Norm] × N
 ↓
Contextual Representations
```

### Decoder (used in GPT)
```
Input Tokens → Embedding + Positional Encoding
 ↓
[Masked Multi-Head Self-Attention → Add & Norm
 → Cross-Attention → Add & Norm
 → Feed Forward → Add & Norm] × N
 ↓
Output Probabilities (next token)
```

### Key Components

| Component | Purpose |
| :--- | :--- |
| **Positional Encoding** | Injects token order (transformers have no inherent sense of position) |
| **Feed-Forward Network** | Two linear layers with GELU activation: $FFN(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ |
| **Add & Norm** | Residual connection + Layer Normalization (stabilizes training) |
| **Causal Mask** (decoder) | Prevents attending to future tokens (autoregressive generation) |

---

## 5. Positional Encoding

Since self-attention is permutation-invariant, we need to inject position information. The original paper used sinusoidal encodings:

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $pos$ | Position of the token in the sequence (0, 1, 2, ...) | Gives the model awareness of word order — without this, "the cat sat" and "sat the cat" would be identical |
| $i$ | Dimension index within the embedding | Different dimensions oscillate at different frequencies, encoding both fine and coarse position |
| $d$ | Model embedding dimension | The total width of the positional encoding vector |
| $10000^{2i/d}$ | Frequency scaling factor | Lower dimensions oscillate fast (capture local position), higher dimensions oscillate slow (capture global position) |
| $\sin / \cos$ | Sinusoidal functions | Continuous and bounded — the model can generalize to sequence lengths unseen during training |

Modern models often use **learned positional embeddings** instead.

---

## 6. GPT: Decoder-Only Transformer

**GPT** (Generative Pre-trained Transformer) uses only the decoder stack:
* **Pre-training:** Predict the next token on massive text corpora.
* **Fine-tuning:** Adapt to specific tasks (classification, QA, etc.).
* **Autoregressive generation:** $P(x_1, x_2, \dots, x_n) = \prod_i P(x_i \mid x_1, \dots, x_{i-1})$

### Scaling Laws
GPT performance follows power laws with:
* More parameters (175B → 1T+)
* More training data (trillions of tokens)
* More compute (thousands of GPUs)

---

## 7. Code Example: Self-Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
 def __init__(self, embed_size, num_heads):
 super().__init__()
 self.num_heads = num_heads
 self.head_dim = embed_size // num_heads

 self.W_q = nn.Linear(embed_size, embed_size)
 self.W_k = nn.Linear(embed_size, embed_size)
 self.W_v = nn.Linear(embed_size, embed_size)
 self.W_o = nn.Linear(embed_size, embed_size)

 def forward(self, x, mask=None):
 B, T, C = x.shape

 # Project to Q, K, V
 q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
 k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
 v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

 # Scaled dot-product attention
 scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
 if mask is not None:
 scores = scores.masked_fill(mask == 0, float('-inf'))
 attn = F.softmax(scores, dim=-1)

 # Weighted sum of values
 out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
 return self.W_o(out)

# Test
attn = SelfAttention(embed_size=64, num_heads=8)
x = torch.randn(2, 10, 64) # batch=2, seq_len=10, embed=64
out = attn(x)
print(f"Input: {x.shape}, Output: {out.shape}") # Same shape (residual-friendly)
```

---

## 8. Transformer vs RNN vs CNN

| Feature | RNN/LSTM | CNN | Transformer |
| :--- | :--- | :--- | :--- |
| Parallelism | Sequential | Parallel | Parallel |
| Long-range dependencies | Degrades | Limited receptive field | Direct (O(1) path length) |
| Position information | Implicit (order) | Local only | Explicit (positional encoding) |
| Computational cost | O(n) per step | O(k·n) | O(n²) attention (improved in variants) |
| Best for | Streaming, edge | Images | NLP, vision, multimodal |

---

## 9. Attention Variants (2026 Landscape)

| Variant | Key Idea | Used In |
| :--- | :--- | :--- |
| **Multi-Head Attention** | Parallel attention heads | All transformers |
| **Grouped Query Attention (GQA)** | Share KV heads across query heads | LLaMA 2/3, Mistral |
| **Flash Attention** | IO-aware exact attention (faster, less memory) | All modern LLMs |
| **Multi-Query Attention (MQA)** | Single KV head | Fast inference |
| **Sparse Attention** | Attend to subset of tokens | Longformer, BigBird |

---

## 10. Advantages & Disadvantages

### Pros
* Fully parallelizable — much faster to train than RNNs.
* Handles long-range dependencies via direct attention.
* Scales massive — powers all frontier AI models.
* Transfer learning king — pretrain once, fine-tune everywhere.

### Cons
* **O(n²) memory** for attention — quadratic in sequence length.
* Requires large datasets and compute to shine.
* Positional encoding is a workaround, not a solution.
* Less natural for streaming/incremental data than RNNs.

---

**Previous:** [RNN](../RNN/Theory.md) | **Next:** [Word Embeddings](../WORD%20EMBEDDINGS/Theory.md) | **Related:** [ANN](../../02-Supervised-Learning/ARTIFICIAL%20NEURAL%20NETWORKS/Theory.md)
