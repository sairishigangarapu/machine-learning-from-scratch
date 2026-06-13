## Attention Mechanism

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. The Problem: Fixed Context Vector Bottleneck

### Motivation and Intuition

In a basic encoder-decoder model, the encoder compresses the entire input sequence into a single context vector. For long sequences, early words get forgotten. For example, translating "don't eat the delicious looking and smelling pizza" — if the model forgets "don't", it becomes "eat the delicious looking and smelling pizza", the opposite meaning.

Even LSTMs struggle because both long-term and short-term memory paths must carry increasing amounts of information. Words at the start of long sequences can still get lost.

**The core insight:** Instead of one context vector, give each decoder step direct access to every encoder hidden state. The decoder learns to "pay attention" to the relevant input words at each step.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Context vector | Single fixed-size vector summarizing the entire input | Creates a bottleneck — early words get lost in long sequences |
| Attention | Mechanism giving decoder direct access to all encoder states | Removes the bottleneck; each decoder step can "peek" at the input |

---

## 2. Attention Intuition: Query, Key, Value

### Motivation and Intuition

Think of attention as a retrieval system:

- **Query**: "What am I looking for right now?" — the decoder's current state.
- **Keys**: "What do I have?" — labels on each encoder state.
- **Values**: "What information do I retrieve?" — the actual content of each encoder state.

For each decoder step, we compute how well the query matches each key (similarity score), convert those scores to probabilities (attention weights), and take a weighted sum of the values.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Query ($Q$) | Decoder's current hidden state | Represents what the decoder needs at this step |
| Key ($K$) | Each encoder hidden state | Used to match against the query |
| Value ($V$) | Each encoder hidden state (usually same as keys) | The content retrieved, weighted by attention |

---

## 3. Alignment Scores and Attention Weights

### Motivation and Intuition

We measure similarity between the decoder state (query) and each encoder state (key) using a dot product. Then softmax converts raw scores into a probability distribution — the attention weights.

$$
\text{score}(s_{t-1}, h_i) = s_{t-1} \cdot h_i
$$

$$
\alpha_{ti} = \text{softmax}(\text{score}(s_{t-1}, h_i)) = \frac{e^{\text{score}(s_{t-1}, h_i)}}{\sum_{j} e^{\text{score}(s_{t-1}, h_j)}}
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $s_{t-1}$ | Decoder hidden state from previous step (query) | What the decoder is currently focused on |
| $h_i$ | Encoder hidden state at position $i$ (key) | Input word representation to match against |
| $\alpha_{ti}$ | Attention weight for encoder position $i$ at decoder step $t$ | How much "attention" to pay to input word $i$ when generating output $t$ |

### Context Vector (Weighted Sum)

Once we have attention weights, the context vector for this decoder step is a weighted sum of all encoder hidden states:

$$
c_t = \sum_{i=1}^{n} \alpha_{ti} h_i
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $c_t$ | Context vector for decoder step $t$ | Carries the relevant input information for predicting the next output word |
| Weighted sum | Each encoder state multiplied by its attention weight | The model blends all inputs proportionally to relevance |

---

## 4. Scaled Dot-Product Attention

### Motivation and Intuition

As vector dimensions grow, dot products become very large, pushing softmax into regions with tiny gradients. Scaling prevents this.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q$ | Query matrix (decoder states) | "What am I looking for?" |
| $K$ | Key matrix (encoder states) | "What do I have?" |
| $V$ | Value matrix (encoder states) | "What information do I retrieve?" |
| $d_k$ | Dimension of key vectors | Scaling factor — prevents softmax saturation |

---

## 5. Python Code: Attention from Scratch

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """Scaled dot-product attention for batched inputs."""
    # Q: (batch_size, n_queries, d_k)
    # K: (batch_size, n_keys, d_k)
    # V: (batch_size, n_keys, d_v)
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

# Demonstration
np.random.seed(42)
batch_size, n_queries, n_keys, d_k = 2, 3, 4, 8
Q = np.random.randn(batch_size, n_queries, d_k)
K = np.random.randn(batch_size, n_keys, d_k)
V = np.random.randn(batch_size, n_keys, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"Attention weights sum to 1 per query:\n{attn_weights.sum(axis=-1)}")

# Visualize a single attention matrix
source_words = ["don't", "eat", "the", "pizza"]
target_words = ["no", "comas", "la", "pizza"]
attn_example = np.array([
    [0.85, 0.10, 0.03, 0.02],
    [0.05, 0.80, 0.10, 0.05],
    [0.02, 0.05, 0.88, 0.05],
    [0.02, 0.03, 0.05, 0.90],
])

print("\nAttention Matrix (target rows x source columns):")
print(f"{'':>6}", " ".join(f"{s:>8}" for s in source_words))
for i, t in enumerate(target_words):
    print(f"{t:>6}", " ".join(f"{attn_example[i,j]:.4f}" for j in range(len(source_words))))
```

### Why Scaled?

For $d_k = 64$ (typical), dot products have variance $d_k = 64$, producing large values. Unscaled softmax saturates, giving near-one-hot attention with vanishing gradients. Dividing by $\sqrt{d_k}$ keeps variance near 1.

---

## 6. Benefits for Long Sequences

| Benefit | Explanation |
| :--- | :--- |
| Direct connections | Each decoder step directly accesses all encoder states — no information loss |
| O(1) path length | Any input word can influence any output word in a single step |
| Interpretability | Attention weights show exactly which input words the model used |
| Gradient flow | Gradients flow directly from output to input without passing through many RNN steps |

---

> **Check your intuition:** In the scaled dot-product attention formula $\text{softmax}(Q K^T / \sqrt{d_k}) V$, what happens if we remove the scaling factor $1/\sqrt{d_k}$ and $d_k = 1024$? The softmax saturates and gradients vanish. What would the attention weights look like in that case (nearly one-hot)?

---

## Prerequisites and Further Reading

- [StatQuest: Encoder-Decoder LSTMs (Seq2Seq)](https://www.youtube.com/watch?v=L8HKweZIOmg)
- [StatQuest: Cosine Similarity](https://www.youtube.com/watch?v=e9U4dEDdtIU)
- [StatQuest: Softmax Function](https://www.youtube.com/watch?v=8tE2v4f3K-s)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- Vaswani et al., "Attention Is All You Need" (2017)
