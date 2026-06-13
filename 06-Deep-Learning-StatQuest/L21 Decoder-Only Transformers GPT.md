## Decoder-Only Transformers (GPT)

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What Is a Decoder-Only Transformer?

### Motivation and Intuition

The original Transformer had an encoder (reads input) and a decoder (generates output). GPT (Generative Pre-trained Transformer) uses **only the decoder stack** — there is no encoder. This makes GPT a pure **language model**: it predicts the next token given all previous tokens, enabling open-ended text generation.

Encoder-decoder transformers translate (input -> output). Decoder-only transformers generate: given a prompt, they continue the sequence one token at a time. ChatGPT, GPT-4, Claude, and LLaMA are all decoder-only.

| Aspect | Encoder-Decoder | Decoder-Only (GPT) |
| :--- | :--- | :--- |
| Components | Encoder + Decoder | Decoder only |
| Attention | Bidirectional (encoder), causal (decoder) | Causal (masked) only |
| Best for | Translation, summarization | Text generation, chat, code generation |
| Training | Sequence-to-sequence objective | Next token prediction |

---

## 2. Masked (Causal) Self-Attention

### Motivation and Intuition

In generation, the model must not peek at future tokens. A **causal mask** zeros out attention to future positions, ensuring each token only attends to itself and previous tokens. This preserves the autoregressive property: $P(x_1, x_2, \dots, x_n) = \prod_i P(x_i \mid x_1, \dots, x_{i-1})$.

The causal mask is an upper-triangular matrix:

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

---

## 3. Autoregressive Generation

### Motivation and Intuition

GPT generates text one token at a time. At each step, all previous tokens attend to each other (causally) through multiple transformer layers. The final layer's output for the last token is projected through a linear head to produce a distribution over the vocabulary.

$$
P(x_t \mid x_{<t}) = \text{softmax}(W_o h_t + b_o)
$$

### Temperature Sampling

The softmax temperature $\tau$ controls randomness:

$$
P(x_t \mid x_{<t}) = \text{softmax}\left(\frac{z_t}{\tau}\right)
$$

| $\tau$ | Behavior |
| :--- | :--- |
| $\tau \to 0$ | Greedy — always picks most likely token (deterministic) |
| $\tau = 1$ | Standard softmax — balanced |
| $\tau > 1$ | More uniform — increases creativity and diversity |

```python
import torch
import torch.nn.functional as F

def generate(model, prompt_tokens, max_new_tokens, temperature=1.0, top_k=50):
    model.eval()
    generated = prompt_tokens.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)  # forward pass
            logits_last = logits[:, -1, :] / temperature  # last token's logits

            # Top-k filtering: zero out all except top-k logits
            if top_k > 0:
                topk_vals, _ = torch.topk(logits_last, top_k, dim=-1)
                logits_last[logits_last < topk_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    return generated
```

---

## 4. Next-Token Prediction Training

### Motivation and Intuition

GPT is pre-trained on massive text corpora by predicting every token in the sequence given all previous tokens. The loss is cross-entropy between predicted and actual tokens.

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x_t$ | True token at position $t$ | The target the model must predict |
| $x_{<t}$ | All tokens before position $t$ | The model's input (causal context) |
| $P(x_t \mid x_{<t})$ | Predicted probability of correct token | What the model outputs via softmax over vocabulary |

```python
def train_step(model, tokens, optimizer):
    # tokens: (batch, seq_len)
    logits = model(tokens)  # (batch, seq_len, vocab_size)
    # Shift: predict token at position t from tokens up to t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

---

## 5. In-Context Learning

### Motivation and Intuition

A remarkable property of decoder-only transformers: without any gradient update, GPT can learn to perform tasks by conditioning on examples in the prompt. This is **in-context learning** (also called few-shot prompting).

| Learning Paradigm | Requires Fine-Tuning? | How It Works |
| :--- | :--- | :--- |
| Zero-shot | No | "Translate to Spanish: hello" -> "hola" |
| Few-shot (in-context) | No | "English: hello -> Spanish: hola\nEnglish: goodbye -> Spanish: adios\nEnglish: cat ->" |
| Fine-tuning | Yes | Update model weights on labeled task data |

In-context learning works because the causal attention mechanism lets later tokens attend to the exemplars in the prompt. The model's internal representations adapt to the pattern present in the provided examples.

---

> **Check your intuition:** GPT uses causal (masked) self-attention, so each token only sees previous tokens. BERT uses bidirectional self-attention. Why can't we use bidirectional attention for language generation? If we did, what would happen when generating the second token of a sentence?

---

## Prerequisites and Further Reading

- [StatQuest: Transformer Neural Networks](https://www.youtube.com/watch?v=zxQyTK8quyY)
- [StatQuest: Word Embeddings](https://www.youtube.com/watch?v=D-ekhWxubTs)
- Original GPT paper: Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018)
- GPT-2: Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- GPT-3: Brown et al., "Language Models are Few-Shot Learners" (2020)
