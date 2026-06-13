# Seq2Seq and Attention: From Fixed Context Vectors to Dynamic Alignment

## 1. Motivation — Sequence-to-Sequence Tasks

### Motivation and Intuition

Many problems in machine learning require transforming one variable-length sequence into another: machine translation ("let's go" -> "vamos"), text summarization (article -> headline), speech recognition (audio -> text), and question answering (question -> answer). These are **sequence-to-sequence (Seq2Seq)** tasks. The fundamental challenge is that both input and output lengths vary, and they may differ from each other.

Traditional neural networks (fixed-size input, fixed-size output) cannot handle this. RNNs can process variable-length input, but producing variable-length output requires a different architecture: the **encoder-decoder** paradigm.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Seq2Seq | Model mapping variable-length input to variable-length output | Enables translation, summarization, speech recognition, and dialogue |
| Encoder | Reads the entire input sequence and compresses it | Produces a fixed-size context vector summarizing the input |
| Decoder | Generates the output sequence from the context vector | Predicts one token at a time until an end-of-sequence token |
| Variable-length | Input and output sequences have different lengths | Requires specialized architecture — feedforward networks cannot handle this |
| End-of-sequence (EOS) | Special token marking generation stop | Decoder stops emitting tokens when EOS is predicted |

**ML Connection:** Seq2Seq models are the foundation of neural machine translation (Google Translate since 2016), speech recognition (DeepSpeech), and summarization systems. They were the dominant paradigm before Transformers.

---

## 2. Encoder-Decoder Architecture

### Motivation and Intuition

The encoder-decoder architecture splits the problem into two phases: understanding (encoding) and generation (decoding). The encoder reads the entire input sequence word by word, updating a hidden state that accumulates information. After processing all tokens, the final hidden state(s) form the **context vector** — a fixed-size summary of the entire input. The decoder then uses this context vector as its initial state and generates the output sequence token by token.

### Encoder: Reading the Input

The encoder converts each input token to a dense embedding vector, then feeds it into an RNN cell (LSTM or GRU). The hidden state at each step captures information from all tokens seen so far.

$$
h_t = \text{RNN}(x_t, h_{t-1})
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $x_t$ | Embedding of input token at time step $t$ | Dense vector representing the current word in continuous space |
| $h_t$ | Hidden state at time step $t$ | Accumulates information from all tokens seen so far |
| $\text{RNN}$ | Recurrent cell (LSTM or GRU) with learned weights | Same cell reused at every time step (weight sharing) |
| $h_0$ | Initial hidden state (usually zero vector) | Starting point for the encoding process |
| $c$ | Context vector: final hidden/cell states from encoder | Fixed-size summary of the entire input sequence |

### The Context Vector Bottleneck

The context vector is the only connection between encoder and decoder. It must encode the entire meaning of the input sequence into a fixed-size vector — this is both the strength and the weakness of basic Seq2Seq.

$$
c = \{(h_1^{(L)}, c_1^{(L)}), (h_2^{(L)}, c_2^{(L)}), \dots, (h_N^{(L)}, c_N^{(L)})\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Context vector | Final set of hidden/cell states from all encoder LSTM layers | Carries full input meaning to decoder |
| Bottleneck | Fixed-size vector must encode arbitrarily long input | Long sequences lose detail — this motivates attention |
| Information loss | Early input words may be forgotten by the time encoding finishes | Fundamental limitation of fixed-context Seq2Seq |

### Decoder: Generating Output

The decoder takes the context vector as its initial state and generates the output sequence one token at a time. At each step, it receives the previous token, updates its hidden state, and produces a probability distribution over the output vocabulary via a linear layer with softmax. Generation stops when the decoder outputs `<EOS>`.

$$
P(y_t \mid y_{<t}, c) = \text{softmax}(W_o h_t^{(dec)} + b_o)
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $y_t$ | Output token at time step $t$ | The predicted word at this position |
| $y_{<t}$ | All previously generated tokens | Condition for the next prediction |
| $h_t^{(dec)}$ | Decoder hidden state at time step $t$ | Conditioned on context vector and all previous outputs |
| $W_o, b_o$ | Output projection weights and bias | Map decoder hidden state to vocabulary-sized logits |
| $c$ | Context vector from encoder | Initial decoder state and conditioning information |

**Worked Example:** Suppose we have a vocabulary of 4 words: {`<SOS>`, `<EOS>`, "hello", "world"}. The decoder hidden state $h_t^{(dec)}$ has dimension 4. The output projection $W_o$ is a $4 \times 4$ matrix. At step $t$, if $h_t^{(dec)} = [1.2, -0.5, 0.8, -0.1]$, then logits are $W_o h_t^{(dec)} + b_o = [0.5, -2.1, 0.3, 1.4]$, and softmax gives probabilities $[0.30, 0.02, 0.25, 0.43]$. The model predicts "world" (index 3).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)           # (batch, seq_len, embedding_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell, outputs  # return all outputs for attention


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, x, hidden, cell):
        # x: (batch, 1) -- single token index
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell
```

---

## 3. Teacher Forcing During Training

### Motivation and Intuition

During training, the decoder is fed the **ground-truth** token at each step instead of its own prediction. This stabilizes training and speeds convergence. Without teacher forcing, early wrong predictions would cascade and compound, making learning extremely difficult — especially in early epochs when the model's predictions are nearly random.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Teacher forcing | Using true target token as next decoder input during training | Stabilizes training; prevents error propagation |
| Scheduled sampling | Gradually reducing teacher forcing ratio as training progresses | Bridges gap between training and inference behavior |
| Exposure bias | Model is only exposed to ground-truth inputs during training | At inference, errors compound — teacher forcing causes mismatch |

$$
\text{input}_{t+1} = \begin{cases} y_{t+1}^{\text{true}} & \text{if } \epsilon < p_{\text{tf}} \\ \hat{y}_t & \text{otherwise} \end{cases}
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $y_{t+1}^{\text{true}}$ | Ground-truth next token | Used during teacher forcing |
| $\hat{y}_t$ | Model's own prediction at step $t$ | Used during free-running mode |
| $p_{\text{tf}}$ | Teacher forcing probability | Controls how often ground truth is used |

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell, encoder_outputs = self.encoder(src)

        input_token = trg[:, 0].unsqueeze(1)  # <SOS> token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs
```

---

## 4. Attention Intuition — The Fix ed Context Vector Problem

### Motivation and Intuition

In a basic encoder-decoder model, the encoder compresses the entire input sequence into a single context vector. For long sequences, early words get forgotten. For example, translating "don't eat the delicious looking and smelling pizza" — if the model forgets "don't", it becomes "eat the delicious looking and smelling pizza", the opposite meaning.

Even LSTMs struggle because both long-term and short-term memory paths must carry increasing amounts of information. Words at the start of long sequences can still get lost. The core insight: instead of one context vector, give each decoder step direct access to every encoder hidden state. The decoder learns to "pay attention" to the relevant input words at each step.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Context vector | Single fixed-size vector summarizing the entire input | Creates a bottleneck — early words get lost in long sequences |
| Attention | Mechanism giving decoder direct access to all encoder states | Removes the bottleneck; each decoder step can "peek" at the input |
| Alignment | Mapping between output positions and relevant input positions | Attention learns which input words are relevant for each output word |
| Information bottleneck | Fixed capacity that limits information transmission | Fundamental problem that attention solves |

**ML Connection:** The attention mechanism was the key breakthrough that enabled neural machine translation to surpass statistical MT. It directly led to the Transformer architecture, which removes recurrence entirely and uses only attention — powering GPT, Claude, BERT, and all modern LLMs.

---

## 5. Attention Mechanism — Query, Key, Value

### Motivation and Intuition

Think of attention as a retrieval system:

- **Query**: "What am I looking for right now?" — the decoder's current hidden state.
- **Keys**: "What do I have?" — labels on each encoder hidden state.
- **Values**: "What information do I retrieve?" — the actual content of each encoder hidden state.

For each decoder step, we compute how well the query matches each key (similarity score), convert those scores to probabilities (attention weights), and take a weighted sum of the values. The result is a **dynamic context vector** that changes at each decoding step.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Query ($Q$) | Decoder's current hidden state | Represents what the decoder needs at this step |
| Key ($K$) | Each encoder hidden state | Used to match against the query |
| Value ($V$) | Each encoder hidden state (same as keys in basic attention) | The content retrieved, weighted by attention |
| Alignment score | Dot product between query and a key | Measures relevance of that input position to current output |

### Alignment Scores and Attention Weights

We measure similarity between the decoder state (query) and each encoder state (key). Then softmax converts raw scores into a probability distribution — the attention weights.

$$
\text{score}(s_{t-1}, h_i) = s_{t-1} \cdot h_i
$$

$$
\alpha_{ti} = \text{softmax}(\text{score}(s_{t-1}, h_i)) = \frac{e^{\text{score}(s_{t-1}, h_i)}}{\sum_{j} e^{\text{score}(s_{t-1}, h_j)}}
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $s_{t-1}$ | Decoder hidden state from previous step (query) | What the decoder is currently focused on |
| $h_i$ | Encoder hidden state at position $i$ (key/value) | Input word representation to match against |
| $\alpha_{ti}$ | Attention weight for encoder position $i$ at decoder step $t$ | How much "attention" to pay to input word $i$ when generating output $t$ |
| $e^{\text{score}}$ | Exponentiation of score | Converts scores to positive values for softmax normalization |
| $\sum_{j} e^{\text{score}(s_{t-1}, h_j)}$ | Normalization across all input positions | Ensures weights sum to 1 |

### Context Vector (Weighted Sum)

Once we have attention weights, the context vector for this decoder step is a weighted sum of all encoder hidden states:

$$
c_t = \sum_{i=1}^{n} \alpha_{ti} h_i
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $c_t$ | Context vector for decoder step $t$ | Carries the relevant input information for predicting the next output word |
| Weighted sum | Each encoder state multiplied by its attention weight | The model blends all inputs proportionally to relevance |
| $n$ | Length of input sequence | Number of encoder hidden states being attended over |

**Worked Example:** Consider translating "the cat" from English to Spanish ("el gato"). The encoder produces two hidden states: $h_0$ ("the") and $h_1$ ("cat"). At the first decoder step, the query $s_0$ might have scores: score($s_0$, $h_0$) = 2.0, score($s_0$, $h_1$) = 0.5. Softmax gives $\alpha_{00} = e^{2.0} / (e^{2.0} + e^{0.5}) = 7.39 / 9.17 = 0.81$ and $\alpha_{01} = 0.19$. The context vector $c_0 = 0.81 h_0 + 0.19 h_1$ is dominated by "the", appropriately for predicting "el" (masculine article for "the").

---

## 6. Scaled Dot-Product Attention

### Motivation and Intuition

As vector dimensions grow, dot products become very large, pushing softmax into regions with tiny gradients. Scaling prevents this by dividing by $\sqrt{d_k}$, keeping the variance near 1.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q$ | Query matrix (concatenated decoder states) | "What am I looking for?" — shape $(n_{\text{queries}}, d_k)$ |
| $K$ | Key matrix (concatenated encoder states) | "What do I have?" — shape $(n_{\text{keys}}, d_k)$ |
| $V$ | Value matrix (concatenated encoder/value states) | "What information do I retrieve?" — shape $(n_{\text{keys}}, d_v)$ |
| $Q K^T$ | Matrix of pairwise dot products | Raw attention scores — shape $(n_{\text{queries}}, n_{\text{keys}})$ |
| $d_k$ | Dimension of key vectors | Scaling factor — prevents softmax saturation |
| $\sqrt{d_k}$ | Square root of key dimension | Keeps variance of dot products near 1 |
| $\text{softmax}$ | Row-wise softmax normalization | Converts scores to probability distribution over keys |
| $V$ | Value matrix | Information to aggregate via weighted sum |

**Why scaling matters:** For $d_k = 64$ (typical), dot products have variance $d_k = 64$, producing values in range $\pm\sqrt{64} = \pm 8$ typically. The function $\text{softmax}(x)$ for $x$ values around $\pm 8$ produces near-one-hot distributions with gradients near zero. Dividing by $\sqrt{d_k} = 8$ keeps values in a range where softmax has meaningful gradients.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """Scaled dot-product attention for batched inputs."""
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
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
```

---

## 7. Attention Weights Visualization

### Motivation and Intuition

One of the most powerful features of attention is interpretability. By visualizing the attention weight matrix, we can see exactly which input words the model focuses on when generating each output word. This creates an **alignment** between input and output sequences that reveals the model's reasoning.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Attention matrix | Matrix of attention weights | Rows = output positions, columns = input positions |
| Alignment | Mapping between input and output tokens | Reveals which input words influence each output word |
| Diagonal structure | Strong alignment of corresponding positions | Indicates monotonic translation (common for related languages) |
| Off-diagonal attention | Non-monotonic alignment | Shows reordering for different syntactic structures |

**Worked Example:** For the translation "don't eat the pizza" -> "no comas la pizza", the attention matrix might show:
- Output "no" attends strongly to "don't" (the negation)
- Output "comas" attends strongly to "eat"
- Output "la" attends strongly to "the"
- Output "pizza" attends strongly to "pizza"

```python
# Visualize a sample attention matrix
source_words = ["don't", "eat", "the", "pizza"]
target_words = ["no", "comas", "la", "pizza"]

# Example attention weights (rows: target, columns: source)
attn_example = np.array([
    [0.85, 0.10, 0.03, 0.02],  # "no" attends to "don't"
    [0.05, 0.80, 0.10, 0.05],  # "comas" attends to "eat"
    [0.02, 0.05, 0.88, 0.05],  # "la" attends to "the"
    [0.02, 0.03, 0.05, 0.90],  # "pizza" attends to "pizza"
])

print("Attention Matrix (target rows x source columns):")
print(f"{'':>6}", " ".join(f"{s:>8}" for s in source_words))
for i, t in enumerate(target_words):
    print(f"{t:>6}", " ".join(f"{attn_example[i,j]:.4f}" for j in range(len(source_words))))
```

**Output format:**
```
          don't      eat      the    pizza
   no   0.8500   0.1000   0.0300   0.0200
comas   0.0500   0.8000   0.1000   0.0500
   la   0.0200   0.0500   0.8800   0.0500
 pizza  0.0200   0.0300   0.0500   0.9000
```

---

## 8. Additive Attention vs Dot-Product Attention

### Motivation and Intuition

Bahdanau et al. (2014) introduced **additive attention** (also called Bahdanau attention), which uses a small feedforward network to compute alignment scores. Vaswani et al. (2017) showed that **dot-product attention** (also called Luong attention) with scaling works just as well and is much faster.

| Aspect | Additive (Bahdanau) Attention | Dot-Product (Luong) Attention |
| :--- | :--- | :--- |
| Score function | $v_a^T \tanh(W_a [s_{t-1}; h_i])$ | $s_{t-1}^T h_i$ or $s_{t-1}^T W_a h_i$ |
| Computation cost | $O(d^2)$ per score | $O(d)$ per score |
| With scaling | N/A | $\frac{s_{t-1}^T h_i}{\sqrt{d_k}}$ |
| Trainable params | Yes ($v_a$, $W_a$) | Optional ($W_a$) |
| Empirical performance | Comparable | Comparable (faster) |

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $v_a$ | Learned attention vector (additive) | Projects combined query-key features to scalar score |
| $W_a$ | Learned weight matrix for alignment | Projects query and key into common space |
| $\tanh$ | Hyperbolic tangent activation | Non-linear transformation for additive attention |
| $[s_{t-1}; h_i]$ | Concatenation of decoder and encoder states | Combines query and key information |

**ML Connection:** Modern transformers use scaled dot-product attention exclusively because it is matrix-compute friendly (highly parallelizable on GPUs). Additive attention is rarely used in modern architectures.

---

## 9. Code Example — Seq2Seq with Attention in PyTorch

### Motivation and Intuition

We now combine the encoder (with all hidden states), the attention mechanism, and the decoder into a complete Seq2Seq model with attention.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionDecoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_vocab_size)
        self.W_a = nn.Linear(hidden_dim, hidden_dim)  # query projection

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch, 1), encoder_outputs: (batch, src_len, hidden_dim)
        embedded = self.embedding(x)  # (batch, 1, embedding_dim)

        # Attention
        query = self.W_a(hidden[-1:]).transpose(0, 1)  # (batch, 1, hidden_dim)
        scores = torch.bmm(query, encoder_outputs.transpose(1, 2))  # (batch, 1, src_len)
        scores = scores / math.sqrt(encoder_outputs.size(-1))
        attn_weights = F.softmax(scores, dim=-1)  # (batch, 1, src_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_dim)

        # Concatenate context with embedded input
        lstm_input = torch.cat([embedded, context], dim=-1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Output prediction
        combined = torch.cat([output.squeeze(1), context.squeeze(1)], dim=-1)
        prediction = self.fc_out(combined)
        return prediction, hidden, cell, attn_weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        all_attention = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)

        hidden, cell, encoder_outputs = self.encoder(src)
        input_token = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden, cell, attn = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            all_attention[:, t, :] = attn.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs, all_attention


# Quick test
def test_seq2seq_attention():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = Encoder(input_vocab_size=50, embedding_dim=16, hidden_dim=32, num_layers=1)
    dec = AttentionDecoder(output_vocab_size=50, embedding_dim=16, hidden_dim=32, num_layers=1)
    model = Seq2SeqAttention(enc, dec, device).to(device)

    src = torch.randint(0, 50, (4, 10)).to(device)
    trg = torch.randint(0, 50, (4, 12)).to(device)

    outputs, attention = model(src, trg, teacher_forcing_ratio=0.8)
    print(f"Output shape: {outputs.shape}")        # (4, 12, 50)
    print(f"Attention shape: {attention.shape}")   # (4, 12, 10)

    # Print attention weights averaged over batch
    avg_attention = attention.mean(dim=0).detach().cpu().numpy()
    print(f"\nAverage attention weights (target x source):")
    print(f"{'':>6}", " ".join(f"{f's{i}':>8}" for i in range(avg_attention.shape[1])))
    for t in range(avg_attention.shape[0]):
        print(f"{f't{t}':>6}", " ".join(f"{avg_attention[t,s]:.3f}" for s in range(avg_attention.shape[1])))

test_seq2seq_attention()
```

---

## 10. Pros and Cons of Seq2Seq with Attention

### Motivation and Intuition

Understanding when to use Seq2Seq with attention versus modern alternatives (Transformers) is essential for practical ML engineering.

| Aspect | Advantage | Disadvantage |
| :--- | :--- | :--- |
| Long sequences | Attention removes the fixed-context bottleneck | O(n) sequential RNN processing is slow |
| Interpretability | Attention weights show alignment | Hard to interpret for very deep models |
| Training stability | Teacher forcing helps | Exposure bias during inference |
| Parallelization | Attention computation is parallelizable over positions | RNN forward pass is inherently sequential (O(n) steps) |
| Information flow | Gradients flow directly between input and output tokens | Recurrent connections can still cause vanishing gradients |
| Flexibility | Works for any seq2seq task without task-specific architecture | Requires aligned input-output pairs for training |

### When to Use Seq2Seq with Attention vs Transformers

| Scenario | Recommendation |
| :--- | :--- |
| Short sequences (length < 100) | Seq2Seq with attention works well |
| Very long sequences (length > 1000) | Transformer with sparse attention is better |
| Resource-constrained deployment | Seq2Seq has fewer parameters than Transformer |
| Streaming/online inference | RNN decoder processes one step at a time naturally |
| Maximum quality | Transformers (encoder+decoder) outperform RNN seq2seq |
| Interpretability needed | Attention visualization works for both |

> **Check your intuition:** In the scaled dot-product attention formula $\text{softmax}(Q K^T / \sqrt{d_k}) V$, what happens if we remove the scaling factor $1/\sqrt{d_k}$ and $d_k = 1024$? The softmax saturates and gradients vanish. What would the attention weights look like in that case (nearly one-hot)? Why would this make learning difficult?

---

## Prerequisites and Further Reading

- **StatQuest videos:** Encoder-Decoder LSTMs (Seq2Seq), Attention Mechanism, RNNs and LSTMs, Word Embeddings
- **Original Seq2Seq paper:** Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (2014)
- **Attention paper:** Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- **Luong attention:** Luong et al., "Effective Approaches to Attention-based Neural Machine Translation" (2015)
- **Modern application:** Vaswani et al., "Attention Is All You Need" (2017) — the Transformer
- **Concepts:** RNN/LSTM fundamentals, word embeddings, softmax function, cross-entropy loss, backpropagation through time
