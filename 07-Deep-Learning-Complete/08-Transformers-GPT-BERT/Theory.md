# Transformers, GPT, and BERT: Attention Is All You Need

## 1. Why Transformers

### Motivation and Intuition

RNNs process tokens one at a time — this is inherently sequential and cannot be parallelized. For long sequences, information from early tokens gets diluted by the time it reaches later tokens. CNNs can process in parallel but have limited receptive fields, requiring many layers to connect distant tokens.

**Transformers** (Vaswani et al., 2017) solve both problems with **self-attention**: every token directly looks at every other token, in parallel, in a single operation. This is the architecture behind GPT, BERT, Claude, and virtually all modern LLMs.

| Problem | RNN/LSTM | CNN | Transformer |
| :--- | :--- | :--- | :--- |
| Parallelism | Sequential (one token at a time) | Parallel | Fully parallel over sequence |
| Long-range dependencies | Degrades with distance | Limited by kernel size (stack many layers) | Direct O(1) path between any tokens |
| Position information | Implicit (order of processing) | Local only (pixel neighborhoods) | Explicit (positional encoding) |
| Computational cost | O(n) sequential steps | O(kn) for kernel size k | O(n^2) attention (optimized in variants) |
| Best for | Streaming, edge, time series | Images, spatial data | NLP, vision, multimodal, everything |

**ML Connection:** Transformers replaced RNNs as the dominant architecture in NLP in ~2018 and now dominate vision, audio, and multimodal tasks. The "Attention Is All You Need" paper is the most cited ML paper of the 2010s.

---

## 2. Self-Attention — Query, Key, Value

### Motivation and Intuition

Self-attention computes a **weighted sum** of all tokens for each token, where the weights are learned based on relevance. Given a sequence of tokens, each token is projected into three vectors.

| Vector | Analogy | Role |
| :--- | :--- | :--- |
| **Query** ($Q$) | "What am I looking for?" | The current token's question |
| **Key** ($K$) | "What do I contain?" | Every token's label or content description |
| **Value** ($V$) | "What do I actually provide?" | The information to aggregate |

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q$ | Query matrix (sequence of query vectors) | Each row is a token asking "what should I attend to?" |
| $K$ | Key matrix (sequence of key vectors) | Each row is a token broadcasting "what I contain" |
| $V$ | Value matrix (sequence of value vectors) | Each row is a token's actual information to be aggregated |
| $Q K^T$ | Raw attention scores matrix | Dot product of every query with every key — measures pairwise similarity |
| $d_k$ | Dimension of each key vector | Used for scaling — prevents dot products from growing too large |
| $\sqrt{d_k}$ | Scaling factor | Without this, for large $d_k$, softmax saturates and gradients vanish |
| $\text{softmax}$ | Row-wise softmax normalization | Converts raw scores to probabilities summing to 1 |
| $\times V$ | Weighted sum of values | Output is weighted combination of all values, weighted by attention |

**Step-by-step computation:**

| Step | Operation | Shape |
| :--- | :--- | :--- |
| 1 | Compute $Q K^T$ (query-key dot products) | $(n, d_k) \times (d_k, n) \rightarrow (n, n)$ |
| 2 | Scale by $1 / \sqrt{d_k}$ | $(n, n)$ |
| 3 | Apply $\text{softmax}$ row-wise | $(n, n)$ — each row sums to 1 |
| 4 | Weighted sum: $\text{softmax}(\dots) \times V$ | $(n, n) \times (n, d_v) \rightarrow (n, d_v)$ |

**Intuition:** If $q_i$ is similar to $k_j$, token $i$ "pays attention" to token $j$'s value. For instance, in "the cat sat on the mat", "sat" might attend strongly to "cat" (subject-verb relationship).

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights
```

---

## 3. Multi-Head Attention

### Motivation and Intuition

Instead of one attention function, we run **$h$ parallel attention heads**, each with different learned projections into lower-dimensional subspaces. Different heads learn different types of relationships — syntactic (subject-verb), semantic (coreference), positional (adjacent words), etc. The outputs are concatenated and projected back to the model dimension.

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\
\text{head}_i &= \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q$ | Query matrix | What each token is "looking for" — derived from input embeddings |
| $K$ | Key matrix | What each token "offers" — used to compute compatibility |
| $V$ | Value matrix | What each token actually "contains" — information to be aggregated |
| $W_i^Q, W_i^K, W_i^V$ | Learned projection matrices for head $i$ | Transform input into query/key/value spaces specific to this head |
| $W^O$ | Output projection matrix | Projects concatenated head outputs back to model dimension |
| $\text{head}_i$ | Output of attention head $i$ | One head's perspective on relationships between tokens |
| $h$ | Number of attention heads (typically 8-16) | More heads = more diverse relationship types captured |

**Parallelization:** All $h$ heads compute attention simultaneously on GPU — no sequential dependency between heads.

---

## 4. Positional Encoding

### Motivation and Intuition

Self-attention is **permutation-invariant** — without position information, "the cat sat" and "sat the cat" produce identical representations. We inject position information by adding positional encodings to token embeddings.

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
| $i$ | Dimension index within the embedding (0, 1, ..., d/2 - 1) | Different dimensions oscillate at different frequencies |
| $d$ | Model embedding dimension | Total width of the positional encoding vector |
| $10000^{2i/d}$ | Frequency scaling factor | Lower dims oscillate fast (capture local position), higher dims oscillate slow (capture global position) |
| $\sin / \cos$ | Sinusoidal functions | Continuous and bounded; model can generalize to unseen sequence lengths |

**Why sinusoidal instead of learned?** Sinusoidal encodings can extrapolate to longer sequences than seen during training (because they are continuous functions of position). Modern models (GPT, BERT) use **learned positional embeddings** instead, which are simpler and work equally well for fixed-length contexts.

```python
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
```

---

## 5. Full Transformer Architecture (Encoder + Decoder)

### Motivation and Intuition

The original Transformer has two stacks: an **encoder** that reads the input and produces contextual representations, and a **decoder** that generates the output autoregressively, attending to the encoder's outputs via **cross-attention**.

### Encoder Stack

```
Input Tokens -> Embedding + Positional Encoding
  -> [Multi-Head Self-Attention -> Add & Norm -> FFN -> Add & Norm] x N
  -> Contextual Representations
```

### Decoder Stack

```
Output Tokens (shifted right) -> Embedding + Positional Encoding
  -> [Masked Multi-Head Self-Attention -> Add & Norm
      -> Cross-Attention (encoder outputs) -> Add & Norm
      -> FFN -> Add & Norm] x N
  -> Linear -> Softmax -> Output Probabilities
```

| Component | Purpose |
| :--- | :--- |
| **Self-attention** | Each token attends to all tokens in the same sequence |
| **Cross-attention** | Decoder attends to encoder outputs (encoder-decoder attention) |
| **Masked self-attention** | Decoder can only attend to previous positions (causal mask) |
| **Feed-Forward Network (FFN)** | Two linear layers with ReLU/GELU: $FFN(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ |
| **Add & Norm** | Residual connection + Layer Normalization (stabilizes training, enables deep stacks) |
| **Positional Encoding** | Injects token order information (permutation-invariance fix) |
| **Causal Mask** | Upper-triangular mask preventing attending to future tokens |

**Encoder architecture detail:** $N$ identical layers, each with multi-head self-attention followed by a position-wise FFN. Residual connections and layer normalization wrap each sub-layer: $\text{LayerNorm}(x + \text{Sublayer}(x))$.

**Cross-attention detail:** The decoder's second attention sub-layer uses queries from the decoder and key/value pairs from the encoder output. This is the mechanism that connects the input understanding (encoding) with output generation (decoding).

---

## 6. GPT — Decoder-Only, Causal Masking, Autoregressive Generation

### Motivation and Intuition

The original Transformer had an encoder (reads input) and a decoder (generates output). GPT (Generative Pre-trained Transformer) uses **only the decoder stack** — there is no encoder. This makes GPT a pure **language model**: it predicts the next token given all previous tokens.

| Aspect | Encoder-Decoder | Decoder-Only (GPT) |
| :--- | :--- | :--- |
| Components | Encoder + Decoder stacks | Decoder stack only |
| Attention types | Bidirectional (encoder), causal (decoder), cross-attention | Causal (masked) self-attention only |
| Best for | Translation, summarization (input->output) | Text generation, chat, code generation |
| Training | Seq2seq objective (encoder reads, decoder predicts) | Next token prediction (autoregressive) |

### Causal (Masked) Self-Attention

In generation, the model must not peek at future tokens. A **causal mask** zeros out attention to future positions, ensuring each token only attends to itself and previous tokens.

```
Token 1: [1, 0, 0, 0]  attends only to itself
Token 2: [1, 1, 0, 0]  attends to tokens 1 and 2
Token 3: [1, 1, 1, 0]
Token 4: [1, 1, 1, 1]
```

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
$$

where $M_{ij} = 0$ if $i \geq j$ and $M_{ij} = -\infty$ if $i < j$.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Causal mask | Upper-triangular matrix of $-\infty$ above diagonal | Prevents attending to future tokens |
| Autoregressive | Each output depends only on previous outputs | Enables sequential generation without cheating |

### Autoregressive Next-Token Prediction

$$
P(x_1, \ldots, x_n) = \prod_{i=1}^n P(x_i \mid x_{<i})
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $x_1, \dots, x_n$ | Full token sequence | Joint probability of the entire sequence |
| $\prod_{i=1}^n$ | Product over all positions | Chain rule of probability — factorized autoregressively |
| $P(x_i \mid x_{<i})$ | Conditional probability of token $i$ given all previous tokens | What GPT outputs at each position |

### Training Objective

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x_t$ | True token at position $t$ | The target the model must predict |
| $x_{<t}$ | All tokens before position $t$ | The model's input (causal context) |
| $P(x_t \mid x_{<t})$ | Predicted probability of correct token | Model output via softmax over vocabulary |

### Temperature Sampling

$$
P(x_t \mid x_{<t}) = \text{softmax}\left(\frac{z_t}{\tau}\right)
$$

| $\tau$ | Behavior | Use Case |
| :--- | :--- | :--- |
| $\tau \to 0$ | Greedy — always picks most likely token | Deterministic, factual tasks |
| $\tau = 1$ | Standard softmax — balanced | General purpose |
| $\tau > 1$ | More uniform — increases creativity | Creative writing, idea generation |

### In-Context Learning

Without any gradient update, GPT can learn to perform tasks by conditioning on examples in the prompt. This is **in-context learning** (few-shot prompting).

| Paradigm | Requires Fine-Tuning? | Example |
| :--- | :--- | :--- |
| Zero-shot | No | "Translate to Spanish: hello" -> "hola" |
| Few-shot (in-context) | No | Provide 3 examples in prompt, model follows pattern |
| Fine-tuning | Yes | Update weights on labeled task data |

In-context learning works because causal attention lets later tokens attend to exemplars in the prompt. The model's internal representations adapt to patterns present in provided examples.

---

## 7. BERT — Encoder-Only, Bidirectional Attention, MLM, NSP

### Motivation and Intuition

BERT (Bidirectional Encoder Representations from Transformers) uses **only the encoder stack**. Its goal is not to generate text but to **understand** it — creating rich, context-aware representations for each token.

| Aspect | Encoder-Only (BERT) | Decoder-Only (GPT) |
| :--- | :--- | :--- |
| Attention | Bidirectional (full context, no mask) | Causal (left-to-right only) |
| Training objective | Masked LM + Next Sentence Prediction | Next token prediction |
| Output | Contextual embeddings per token | Generated token sequence |
| Best for | Classification, extraction, QA | Generation, chat, code |

### Bidirectional Self-Attention

Every token attends to every other token, including future ones. No causal mask is applied.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Bidirectional attention | Each token attends to all tokens in the sequence | Captures full left and right context — "bank" in "river bank" vs "money bank" |
| Full attention mask | All-ones matrix, no causal restriction | Every token sees everything |

**Why this matters:** "I went to the bank to deposit money" — a causal model sees only "I went to the" before "bank". BERT sees the whole sentence and uses "deposit money" to disambiguate "bank" as financial.

### Input Representation

$$
\text{input\_embedding} = \text{token\_embedding} + \text{segment\_embedding} + \text{position\_embedding}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Token embedding | Learned embedding for each subword (WordPiece) | Converts discrete tokens to continuous vectors |
| Segment embedding | Indicates sentence A (0) or sentence B (1) | Tells BERT which sentence a token belongs to |
| Position embedding | Learned embedding for each position (0, 1, 2, ...) | Gives model awareness of word order |

### Special Tokens

| Token | Purpose |
| :--- | :--- |
| `[CLS]` | Prepended to every sequence. Final hidden state used for classification |
| `[SEP]` | Separator between sentences (and at end) |
| `[MASK]` | Replaces tokens during pre-training. Model must predict original token |
| `[PAD]` | Padding for batched sequences |

Input format:
```
[CLS] The cat sat [SEP] It was on the mat [SEP]
```

### Masked Language Model (MLM)

Randomly mask tokens and train the model to predict them using full bidirectional context.

**Masking strategy (15% of tokens):**

1. **80%** of selected tokens: replace with `[MASK]`
2. **10%** of selected tokens: replace with a random token
3. **10%** of selected tokens: keep unchanged

### MLM Loss

$$
\mathcal{L}_{MLM} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid x_{\backslash M})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $M$ | Set of masked positions | Only 15% of tokens contribute to loss |
| $x_i$ | Original token at position $i$ | The target the model must predict |
| $x_{\backslash M}$ | Sequence with some tokens masked | Input representing corrupted sequence |
| $P(x_i \mid x_{\backslash M})$ | Predicted probability of original token | Output of softmax over vocabulary at position $i$ |

### Next Sentence Prediction (NSP)

Trains BERT to predict whether sentence B follows sentence A.

| Sentence A | Sentence B | Label |
| :--- | :--- | :--- |
| The cat sat on the mat. | It was comfortable. | IsNext |
| The cat sat on the mat. | Penguins eat fish. | NotNext |

$$
\mathcal{L}_{NSP} = -\log P(\text{label} \mid [CLS])
$$

Combined pre-training loss: $\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$

### Fine-Tuning for Downstream Tasks

After pre-training, BERT can be fine-tuned for specific tasks by adding a small task-specific head:

- **Classification:** $P(y \mid \text{text}) = \text{softmax}(W_c h_{[CLS]} + b_c)$
- **Named Entity Recognition:** $P(y_i \mid \text{text}) = \text{softmax}(W_{ner} h_i + b_{ner})$
- **Question Answering:** Predict answer span start and end positions

### BERT Configurations

| Hyperparameter | BERT Base | BERT Large |
| :--- | :--- | :--- |
| Transformer layers | 12 | 24 |
| Hidden size | 768 | 1024 |
| Attention heads | 12 | 16 |
| Parameters | 110M | 340M |

---

## 8. Code Example — SelfAttention + PositionalEncoding + MiniGPT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert embed_size % num_heads == 0

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Project to Q, K, V and reshape for multi-head
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Weighted sum of values, reshape, project
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(embed_size, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


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


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4, block_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, embed_size * 4)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

---

## 9. Attention Variants (2026 Landscape)

| Variant | Key Idea | Computation | Used In |
| :--- | :--- | :--- | :--- |
| **Multi-Head Attention** | Parallel attention heads, each with Q, K, V projections | O(n^2) per head, h heads | All transformers |
| **Multi-Query Attention (MQA)** | Single KV head shared across all query heads | O(n^2) but much less memory | Fast inference models |
| **Grouped Query Attention (GQA)** | Share KV heads among groups of query heads | O(n^2), middle ground | LLaMA 2/3, Mistral |
| **Flash Attention** | IO-aware exact attention using tiling | O(n^2) compute, O(n) memory (no materialization) | All modern LLMs |
| **Sparse Attention** | Attend only to subset of tokens (local + global) | O(n log n) or O(n sqrt n) | Longformer, BigBird |
| **Sliding Window Attention** | Attend only to nearby tokens in a window | O(nk) for window k | Mistral, some efficient variants |
| **Linear Attention** | Replace softmax with kernel feature maps | O(n) | Efficient Transformers |

**Flash Attention detail:** Instead of materializing the full $n \times n$ attention matrix in GPU HBM, it computes attention in tiles using the online softmax algorithm. This reduces memory from O(n^2) to O(n) while being exact (not approximate). It is the default attention implementation in PyTorch 2.0+.

---

## 10. Transformer vs RNN vs CNN — Pros and Cons

### Comparison Table

| Feature | RNN/LSTM | CNN | Transformer |
| :--- | :--- | :--- | :--- |
| Parallelism | Sequential (O(n) steps) | Parallel over positions | Fully parallel over positions |
| Long-range dependencies | Degrades with distance | Requires O(n/k) layers | Direct O(1) path length |
| Position info | Implicit (sequential processing) | Local only (kernel neighborhoods) | Explicit (positional encoding) |
| Computational cost (training) | O(n) sequential, O(n) memory | O(kn), O(n) memory | O(n^2) attention, O(n^2) memory |
| Computational cost (inference) | O(n) sequential (fast per step) | O(n) parallel (fast) | O(n) sequential (key-value cache) |
| Parameter efficiency | Moderate | High (weight sharing) | Low (large embeddings, projections) |
| Data efficiency | Good for small data | Good for images | Requires large datasets |
| Long sequences (10k+) | Struggles (vanishing gradients) | Works with enough layers | O(n^2) memory becomes prohibitive |

### Pros of Transformers

- Fully parallelizable — much faster to train than RNNs
- Handles long-range dependencies via direct attention (O(1) path length)
- Scales massive — powers all frontier AI models (up to trillions of parameters)
- Transfer learning king — pretrain once, fine-tune everywhere
- Single architecture for text, vision, audio, multimodal

### Cons of Transformers

- **O(n^2) memory and compute** for full attention — quadratic in sequence length
- Requires large datasets and substantial compute to train from scratch
- Positional encoding is a workaround, not an inherent capability
- Less natural for streaming/incremental data than RNNs
- Loses local inductive bias that CNNs have for images
- Sensitive to training instability (warmup, learning rate scheduling critical)

### When to Choose What

| Scenario | Recommendation |
| :--- | :--- |
| Large text corpus (1B+ tokens) | Transformer (GPT/BERT variants) |
| Small dataset (10k samples) | RNN/LSTM with regularization |
| Image classification | CNN or Vision Transformer (ViT) with enough data |
| Streaming audio | RNN or transformer with causal masking |
| Very long documents (10k+ tokens) | Sparse/Longformer/BigBird |

> **Check your intuition:** GPT uses causal (masked) self-attention so each token only sees previous tokens. BERT uses bidirectional self-attention. Why can't we use bidirectional attention for language generation? If we did, what would happen when generating the second token of a sentence? — The model would peek at tokens it hasn't generated yet, breaking the autoregressive property.

---

## Prerequisites and Further Reading

- **StatQuest:** Transformer Neural Networks, Decoder-Only Transformers (GPT), Encoder-Only Transformers (BERT), Attention Mechanism
- **Original paper:** Vaswani et al., "Attention Is All You Need" (2017)
- **GPT papers:** Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018); "Language Models are Unsupervised Multitask Learners" (2019); Brown et al., "Language Models are Few-Shot Learners" (2020)
- **BERT:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- **Flash Attention:** Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022); "FlashAttention-2" (2023)
- **Concepts:** Self-attention, softmax, cross-entropy loss, layer normalization, residual connections
