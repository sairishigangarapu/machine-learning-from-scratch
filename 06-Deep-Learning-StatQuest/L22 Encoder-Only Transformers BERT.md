## Encoder-Only Transformers (BERT)

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What Is an Encoder-Only Transformer?

### Motivation and Intuition

The original Transformer had an encoder (reads input) and a decoder (generates output). BERT (Bidirectional Encoder Representations from Transformers) uses **only the encoder stack**. Its goal is not to generate text, but to understand it — creating rich, context-aware representations for each token.

| Aspect | Encoder-Only (BERT) | Decoder-Only (GPT) |
| :--- | :--- | :--- |
| Attention | Bidirectional (full context) | Causal (left-to-right only) |
| Training objective | Masked LM + Next Sentence Prediction | Next token prediction |
| Output | Contextual embeddings per token | Generated token sequence |
| Best for | Classification, extraction, QA | Generation, chat, code |

> The encoder reads the whole input at once — like a student reading an entire exam question before answering. The decoder reads one token at a time — like someone speaking a sentence word by word.

---

## 2. Bidirectional Self-Attention

### Motivation and Intuition

The defining feature of encoder-only transformers: **every token attends to every other token**, including future ones. No causal mask is applied. This gives BERT full left and right context for every token.

**Bidirectional mask:**
```
Token 1: [1, 1, 1, 1]
Token 2: [1, 1, 1, 1]
Token 3: [1, 1, 1, 1]
Token 4: [1, 1, 1, 1]
```

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Bidirectional attention | Each token attends to all tokens in the sequence | Captures full left and right context — "bank" differs in "river bank" vs "money bank" |
| Self-attention (encoder) | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | Same formula as original transformer, no causal mask |

**Why this matters:** "I went to the bank to deposit money" — a causal model sees only "I went to the" before "bank". BERT sees the whole sentence and uses "deposit money" to disambiguate "bank" as financial.

---

## 3. Input Representation

BERT input is the sum of three embeddings:

$$
\text{input\_embedding} = \text{token\_embedding} + \text{segment\_embedding} + \text{position\_embedding}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Token embedding | Learned embedding for each subword token (WordPiece) | Converts discrete tokens to continuous vectors |
| Segment embedding | Indicates sentence A (0) or sentence B (1) | Tells BERT which sentence a token belongs to |
| Position embedding | Learned embedding for each position (0, 1, 2, ...) | Gives model awareness of word order |

### Special Tokens

| Token | Purpose |
| :--- | :--- |
| `[CLS]` | Prepended to every sequence. Final hidden state used for classification. |
| `[SEP]` | Separator between sentences (and at end). |
| `[MASK]` | Replaces tokens during pre-training. Model must predict original token. |
| `[PAD]` | Padding for batched sequences. |

```
Input: [CLS] The cat sat [SEP] It was on the mat [SEP]
```

---

## 4. Masked Language Model (MLM)

### Motivation and Intuition

BERT's primary pre-training objective: randomly mask tokens and train the model to predict them using full bidirectional context.

### Masking Strategy (15% of tokens):

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
| $x_{\backslash M}$ | Sequence with some tokens masked | Input — model must infer missing tokens from bidirectional context |
| $P(x_i \mid x_{\backslash M})$ | Predicted probability of original token | Output of softmax over vocabulary at position $i$ |

> **Why 80/10/10?** If BERT only saw `[MASK]` during pre-training, it would never learn robust representations for non-masked tokens at fine-tuning time. The random and unchanged tokens force the model to maintain representations of every token.

---

## 5. Next Sentence Prediction (NSP)

### Motivation and Intuition

Tasks like Question Answering require understanding relationships between sentences. NSP trains BERT to predict whether sentence B follows sentence A.

| Sentence A | Sentence B | Label |
| :--- | :--- | :--- |
| The cat sat on the mat. | It was comfortable. | IsNext |
| The cat sat on the mat. | Penguins eat fish. | NotNext |

$$
\mathcal{L}_{NSP} = -\log P(\text{label} \mid [CLS])
$$

Combined pre-training loss: $\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$

---

## 6. Fine-Tuning for Downstream Tasks

### Motivation and Intuition

After pre-training, BERT can be fine-tuned for specific tasks by adding a small task-specific head. The pre-trained encoder weights are updated with a low learning rate.

### Classification (using [CLS] token)

$$
P(y \mid \text{text}) = \text{softmax}(W_c h_{[CLS]} + b_c)
$$

The `[CLS]` token's final hidden state aggregates the entire sequence into a single vector for classification.

### Named Entity Recognition (using each token)

$$
P(y_i \mid \text{text}) = \text{softmax}(W_{ner} h_i + b_{ner})
$$

Each token is classified independently into entity types.

### Question Answering (predict answer span)

$$
\begin{aligned}
P_{start}(i) &= \text{softmax}(W_{start} h_i + b_{start})_i \\
P_{end}(j) &= \text{softmax}(W_{end} h_j + b_{end})_j
\end{aligned}
$$

Input format: `[CLS] Question [SEP] Context [SEP]`

---

### BERT Configurations

| Hyperparameter | BERT Base | BERT Large |
| :--- | :--- | :--- |
| Transformer layers | 12 | 24 |
| Hidden size | 768 | 1024 |
| Attention heads | 12 | 16 |
| Parameters | 110M | 340M |

---

> **Check your intuition:** Why would you use an encoder-only model instead of a decoder-only model for a sentiment analysis task? Consider the sentence: "This movie was not bad at all." BERT sees "at all" at the end and understands the negation scope correctly. A causal model only sees "This movie was not" when predicting and might predict negative sentiment incorrectly.

---

## Prerequisites and Further Reading

- [StatQuest: Transformer Neural Networks](https://www.youtube.com/watch?v=zxQyTK8quyY)
- [StatQuest: Word Embeddings](https://www.youtube.com/watch?v=D-ekhWxubTs)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
