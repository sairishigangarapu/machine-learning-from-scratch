# Recurrent Neural Networks (RNNs)

## 1. Why RNNs?

Fully connected networks and CNNs assume inputs are **independent** and **fixed-size**. But for sequences (text, time series, speech, video), the order and length matter fundamentally — the meaning of a word depends on the words before it, and a stock price tomorrow depends on the trend today.

An RNN processes sequences by maintaining a **hidden state** that acts as a memory of past inputs. At each time step, the hidden state is updated based on the current input and the previous hidden state, creating a recurrent loop.

### Motivation and Intuition

Standard neural networks and CNNs assume inputs are independent and fixed-size. But many problems involve sequences: stock prices over time, sentences, audio, or video. Recurrent Neural Networks handle sequential data by maintaining a hidden state that carries information from one time step to the next. The same weights are used at every time step (weight sharing), which keeps the parameter count manageable regardless of sequence length.

The fundamental idea: instead of mapping input -> output directly, the network maintains an evolving internal state that encodes information about the entire sequence seen so far. This is analogous to how a computer program maintains variables that accumulate information as it runs.

| Property | Feedforward | Recurrent |
| :--- | :--- | :--- |
| Input type | Fixed-size vector | Variable-length sequence |
| State | None (stateless) | Hidden state (memory) |
| Weight sharing | None across positions | Across all time steps |
| Parameter count | Grows with input size | Fixed (independent of sequence length) |
| Order sensitivity | Positional encoding needed | Order encoded by recurrence |

---

## 2. Vanilla RNN

### Motivation and Intuition

The simplest recurrent architecture. At each time step t, the network takes the current input x_t and the previous hidden state h_{t-1}, and produces a new hidden state h_t. This hidden state is the network's memory — it should encode all relevant information from the sequence up to time t.

The output at each step y_t is computed from the hidden state. In a "many-to-many" setup, we get an output at every step. In "many-to-one", we only use the final hidden state.

### Formal Definition

$$
\begin{aligned}
\mathbf{h}_t &= \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h) \\
\mathbf{y}_t &= \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}_t$ | Input vector at time step $t$ | The current token/feature being processed — e.g., a word embedding or sensor reading |
| $\mathbf{h}_t$ | Hidden state at time step $t$ | The network's "memory" — encodes all past information up to step $t$ |
| $\mathbf{h}_{t-1}$ | Previous hidden state | The memory from the last step — carries forward what happened before |
| $\mathbf{y}_t$ | Output vector at time step $t$ | The prediction at this time step (e.g., next word probability) |
| $\mathbf{W}_{hh}$ | Recurrent weight matrix (hidden -> hidden) | Controls how the previous hidden state influences the current one — learns temporal dynamics |
| $\mathbf{W}_{xh}$ | Input weight matrix (input -> hidden) | Projects current input into hidden state space |
| $\mathbf{W}_{hy}$ | Hidden-to-output weight matrix | Maps the hidden state to the output space for predictions |
| $\mathbf{b}_h$ | Hidden layer bias | Shift parameter for hidden state |
| $\mathbf{b}_y$ | Output layer bias | Shift parameter for output |
| $\tanh$ | Hyperbolic tangent activation | Introduces non-linearity, squashes to (-1, 1) |

### Worked Numerical Example

Suppose we have:
- h_0 = [0, 0] (initial hidden state, 2D)
- x_1 = [1, 0] (input, 2D)
- W_hh = [[0.5, 0.1], [-0.2, 0.3]]
- W_xh = [[0.8, -0.4], [0.2, 0.6]]
- b_h = [0.1, -0.1]

Step 1:
W_hh * h_0 = [0, 0]
W_xh * x_1 = [0.8*1 + (-0.4)*0, 0.2*1 + 0.6*0] = [0.8, 0.2]
Sum + b = [0.8 + 0.1, 0.2 + (-0.1)] = [0.9, 0.1]
h_1 = tanh([0.9, 0.1]) = [0.716, 0.100]

Step 2 (x_2 = [0, 1]):
W_hh * h_1 = [0.5*0.716 + 0.1*0.100, -0.2*0.716 + 0.3*0.100] = [0.368, -0.113]
W_xh * x_2 = [-0.4, 0.6]
Sum + b = [0.368 + (-0.4) + 0.1, -0.113 + 0.6 + (-0.1)] = [0.068, 0.387]
h_2 = tanh([0.068, 0.387]) = [0.068, 0.369]

The hidden state evolves step by step, carrying information forward.

### Unfolding in Time

The recurrence defines a deep network when "unfolded" across time. Crucially, W_hh, W_xh, and W_hy are the **same** at every time step:

```
x_1 -> [RNN] -> h_1 -> [RNN] -> h_2 -> [RNN] -> h_3 -> output
           ^            ^            ^
           | W_hh       | W_hh       | W_hh (same weights)
```

This weight sharing is what makes RNNs generalizable to sequences of any length. The same learned dynamics apply regardless of how many steps the sequence contains.

---

## 3. Backpropagation Through Time (BPTT) and Vanishing Gradients

### Motivation and Intuition

RNNs are trained by unrolling the computation graph and applying backpropagation across all time steps. However, this creates a deep network whose depth equals the sequence length. The gradient at time step t depends on all previous time steps through repeated multiplication by W_hh.

### Formal Definition

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| L | Total loss across all time steps | The sum of losses at each time step — what we minimize |
| L_t | Loss at time step t | How wrong the prediction is at this specific step |
| W | All model weights (shared across time steps) | The same weights are used at every step — gradients are accumulated across time |
| T | Total number of time steps | The length of the input sequence |

### The Vanishing Gradient Problem

The gradient of the loss at time t with respect to the hidden state at time k << t involves repeated multiplication by W_hh^T (or the Jacobian of the activation). This leads to exponential behavior:

- If eigenvalues of W_hh are < 1, gradients **vanish** (shrink exponentially) -> model cannot learn long-range dependencies.
- If eigenvalues of W_hh are > 1, gradients **explode** (grow exponentially) -> training becomes unstable (gradient clipping helps).

This is why vanilla RNNs struggle with sequences longer than ~10-20 steps. The gradient signal from early time steps is either washed out (vanishing) or overwhelms the update (exploding).

### Gradient Clipping

A simple fix for exploding gradients: cap the gradient norm before updating weights.

```
If ||g|| > threshold: g = g * threshold / ||g||
```

This prevents any single gradient update from destabilizing the weights.

---

## 4. LSTM: Long Short-Term Memory

### Motivation and Intuition

LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem by introducing a dedicated **cell state** C_t that acts as a memory highway. Information can flow through the cell state with minimal modification, while three **gates** (forget, input, output) control what to keep, what to write, and what to read.

The LSTM maintains two states at each time step:
- **Cell state** C_t: The long-term memory, carried forward with linear interactions.
- **Hidden state** h_t: The short-term memory / output for this time step.

The key innovation: the cell state is updated via **addition** (not multiplication). When the forget gate is close to 1, the gradient of C_t with respect to C_{t-1} is approximately 1, allowing gradients to flow backward through many time steps without vanishing.

### Formal Definition

**Gates:**

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}
\end{aligned}
$$

**Cell State Update:**

$$
\begin{aligned}
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \quad \text{(candidate)} \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(new cell state)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(hidden output)}
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| f_t | Forget gate (0 to 1 per element) | Decides what to discard from the cell state — 0 = forget, 1 = keep |
| i_t | Input gate (0 to 1 per element) | Decides what new information to store — controls candidate admission |
| o_t | Output gate (0 to 1 per element) | Decides what to output from the cell state — controls hidden state exposure |
| C_tilda_t | Candidate cell state | New information that could be added to the cell state (before gating) |
| C_t | Current cell state | The "conveyor belt" — carries information across time with minimal modification |
| C_{t-1} | Previous cell state | Memory from the last time step |
| h_t | Final hidden output | The visible output of the LSTM cell at this time step |
| sigma | Sigmoid activation | Squashes to (0, 1) — acts as a soft switch for gates |
| tanh | Hyperbolic tangent | Squashes to (-1, 1) — normalizes values for cell state and candidate |
| * | Hadamard (element-wise) product | Each element is multiplied independently — gates operate per-dimension |
| [h_{t-1}, x_t] | Concatenated input | The previous hidden state and current input combined into a single vector |

### How LSTM Solves the Vanishing Gradient

The cell state C_t is updated via **addition** (not multiplication):

$$
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t
$$

When the forget gate $\mathbf{f}_t$ is close to 1, the gradient of $\mathbf{C}_t$ with respect to $\mathbf{C}_{t-1}$ is approximately 1. This means the gradient can flow backward through many time steps without vanishing or exploding. The additive (rather than multiplicative) recurrence is the key innovation.

### Peephole Connections

A common LSTM variant adds "peephole" connections that let the gates look at the cell state directly:

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{p}_f \odot \mathbf{C}_{t-1} + \mathbf{b}_f)
$$

where p_f is a learned peephole weight vector for the forget gate. This gives the gates direct access to the cell state content.

---

## 5. GRU: Gated Recurrent Unit

### Motivation and Intuition

GRUs (Cho et al., 2014) simplify LSTMs by merging the cell state and hidden state into a single vector and using only two gates (reset and update) instead of three. This reduces the parameter count while typically matching LSTM performance on many tasks.

The **update gate** z_t controls how much of the past to carry forward (like a combined forget + input gate). The **reset gate** r_t controls how much past information to forget when computing the new candidate.

### Formal Definition

**Gates:**

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(update gate)} \\
\mathbf{r}_t &= \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(reset gate)}
\end{aligned}
$$

**Hidden State Update:**

$$
\begin{aligned}
\tilde{\mathbf{h}}_t &= \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(candidate)} \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(new hidden state)}
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| z_t | Update gate (0 to 1 per element) | Controls the blend between old state and new candidate — like a learned interpolation |
| r_t | Reset gate (0 to 1 per element) | Controls how much of the past to forget when generating the candidate |
| h_tilda_t | Candidate hidden state | New information proposed for the hidden state (after reset gating) |
| h_t | New hidden state | Linear interpolation between old state and candidate, controlled by z_t |
| (1 - z_t) * h_{t-1} | Retained portion of old state | What we keep from the past — how much the old state persists |
| z_t * h_tilda_t | New information added | What we accept from the candidate — how much the new input influences the state |

GRUs have **fewer parameters** than LSTMs (2 gates vs 3, no separate cell state) and often perform comparably, making them a good default choice for many sequence tasks.

---

## 6. LSTM vs GRU vs Vanilla RNN

| Feature | Vanilla RNN | LSTM | GRU |
| :--- | :--- | :--- | :--- |
| Gates | 0 | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | No | Yes (separate) | No (merged with hidden) |
| Parameters | Fewest | Most (4 weight matrices) | Middle (3 weight matrices) |
| Long-range memory | Poor | Excellent | Good |
| Vanishing gradient | Severe | Solved (additive cell state) | Mitigated |
| Training speed | Fastest | Slowest | Middle |
| When to use | Very short sequences | Complex long sequences | Good default choice |
| Output range | (-1, 1) | (-1, 1) for hidden | (-1, 1) |
| LSTM parameter count | N/A | 4 * (4*hidden*input + 4*hidden^2) | 3 * (3*hidden*input + 3*hidden^2) |

**Parameter count comparison** for input_size=10, hidden_size=128:
- LSTM: 4 * (128*10 + 128*128 + 128) + 128*10 = ~71,168
- GRU: 3 * (128*10 + 128*128 + 128) + 128*10 = ~54,656
- Vanilla RNN: 128*10 + 128*128 + 128 + 128*10 = ~19,456

---

## 7. Applications of RNNs

| Task | Input | Output | Architecture |
| :--- | :--- | :--- | :--- |
| Sentiment Analysis | Sequence of words | Single label (positive/negative) | Many-to-one |
| Machine Translation | Sequence in source language | Sequence in target language | Many-to-many (encoder-decoder) |
| Time Series Forecasting | Past values | Future values | Many-to-one or many-to-many |
| Text Generation | Single char/word | Next char/word | One-to-many (autoregressive) |
| Speech Recognition | Audio features (time frames) | Text transcription | Many-to-many |
| Video Classification | Sequence of frames | Activity label | Many-to-one |
| Music Generation | Sequence of notes | Next notes | Many-to-many |

### Sentiment Analysis Example

Input: "The movie was absolutely fantastic"
Output: Positive (1)

The RNN processes each word sequentially. The final hidden state after reading the last word encodes information about the entire sentence, which is fed to a classifier.

### Machine Translation Example (Encoder-Decoder)

Input: "Je suis etudiant" (French)
Output: "I am a student" (English)

An **encoder** RNN reads the source sentence and produces a context vector (final hidden state). A **decoder** RNN generates the target sentence one word at a time, conditioned on this context and the previously generated word.

---

## 8. Code Example: LSTM for Sequence Classification

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from the top layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.fc(last_hidden)

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, h_n = self.gru(x)
        return self.fc(h_n[-1])

class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, h_n = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

# Example: classify sequences of length 20, 5 features each
model = LSTMClassifier(input_size=5, hidden_size=64, num_layers=2, num_classes=3)
x = torch.randn(32, 20, 5)  # batch=32, seq_len=20, features=5
out = model(x)
print(f"Output shape: {out.shape}")  # (32, 3)
```

---

## 9. The Shift to Transformers

While LSTMs and GRUs were state-of-the-art for sequences until approximately 2017, **Transformers** (Vaswani et al., "Attention is All You Need") now dominate NLP and sequence modeling.

| Limitation | RNN | Transformer |
| :--- | :--- | :--- |
| Processing | Sequential — O(n) time steps, cannot parallelize | Parallel — O(1) with self-attention, all positions processed simultaneously |
| Long sequences | Struggles beyond ~500 steps | Handles 512-4096+ tokens natively |
| Memory | Hidden state compresses all history (bottleneck) | Attention looks at all positions directly (full context) |
| Training speed | Slow on GPUs (sequential) | Fast on GPUs (parallel) |
| Position encoding | Natural (order = time step) | Requires positional embeddings |

### Bidirectional RNNs

An extension: process the sequence both forward and backward, concatenating the hidden states:

```
h_t = [h_t_forward, h_t_backward]
```

This gives each position access to both past and future context (like BERT would later achieve with transformers).

> Transformers are covered in advanced courses. For most NLP and sequence tasks in 2026, Transformers are the default. RNNs remain relevant for streaming/real-time data (where you cannot wait for the full sequence), edge deployment (smaller models), and reinforcement learning (partial observability).

> **Check your intuition:** If an RNN has a hidden state size of 128 and an input size of 10, how many parameters are in W_hh and W_xh combined?

> **Answer:** W_hh is 128x128 = 16,384 and W_xh is 128x10 = 1,280, total 17,664. For an LSTM, each gate has its own weights, so 4x this amount = 70,656 for the recurrent portion alone.

> **Check your intuition:** If the forget gate outputs are f_t = [0.99, 0.01, 0.50], what does this tell us about each component of the cell state?

> **Answer:** The first component keeps 99% of old memory, the second discards almost all (retains only 1%), and the third keeps half. The forget gate operates per-element, so different features can have different retention rates.

---

## 10. Advantages and Disadvantages

### Pros

- Naturally handles variable-length sequences (no fixed input size required).
- Captures temporal dependencies and sequential patterns.
- LSTM/GRU solve vanishing gradients for long sequences (up to ~500 steps).
- Fewer parameters than transformers for small to medium tasks.
- Applicable to virtually any sequential data (text, audio, time series, video).
- Streaming-friendly — can process input one step at a time without needing the full sequence.

### Cons

- Sequential processing is hard to parallelize (slow on long sequences, especially vs transformers).
- Still struggles with sequences longer than ~500 steps (transformers are better).
- Prone to overfitting on small datasets.
- Hyperparameter-sensitive (hidden size, number of layers, learning rate, dropout).
- Vanilla RNN (without LSTM/GRU) is practically useless for anything beyond very short sequences.
- Less interpretable than attention-based models (attention weights show which positions matter).

---

## Prerequisites and Further Reading

- **Prerequisites:** Artificial Neural Networks, Backpropagation, Vanishing Gradient Problem.
- **Related:** CNNs (spatial counterpart), Transformers (modern sequential processing).
- **Original papers:**
  - Hochreiter & Schmidhuber, "Long Short-Term Memory" (LSTM, 1997)
  - Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (GRU, 2014)
  - Vaswani et al., "Attention Is All You Need" (Transformers, 2017)
- **Next:** Word Embeddings and Word2Vec.
