# Recurrent Neural Networks (RNNs)

## 1. Why RNNs?

Fully connected networks and CNNs assume inputs are **independent** and **fixed-size**. But for sequences (text, time series, speech), the order and length matter. An RNN processes sequences by maintaining a **hidden state** that acts as a memory of past inputs.

$$
\begin{aligned}
h_t &= \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b) \\
y_t &= W_{hy} h_t + c
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x_t$ | Input vector at time step $t$ | The current token/feature being processed |
| $h_t$ | Hidden state at time step $t$ | The network's "memory" — encodes all past information up to step $t$ |
| $y_t$ | Output vector at time step $t$ | The prediction at this time step |
| $W_{hh}$ | Recurrent weight matrix (hidden → hidden) | Controls how the previous hidden state influences the current one — this is where temporal dependencies are learned |
| $W_{xh}$ | Input weight matrix (input → hidden) | Controls how the current input affects the hidden state |
| $W_{hy}$ | Hidden-to-output weight matrix | Maps the hidden state to the output space |
| $b$ | Hidden layer bias | Allows the hidden state to shift, fitting data that doesn't pass through the origin |
| $c$ | Output layer bias | Final adjustment to the output |
| $\sigma$ | Activation function (typically tanh or ReLU) | Introduces non-linearity; without it, the RNN collapses to a linear model |

---

## 2. Unrolling Through Time

The RNN is **unrolled** across time steps, revealing that it is essentially a deep network where each layer shares the same weights:

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → output
 ↑ ↑ ↑
 W_hh W_hh W_hh (same weights)
```

---

## 3. Training: Backpropagation Through Time (BPTT)

RNNs are trained by unrolling the network and applying backpropagation across all time steps. The gradient at time step $t$ depends on all previous time steps:

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $L$ | Total loss across all time steps | The sum of losses at each time step — what we minimize |
| $L_t$ | Loss at time step $t$ | How wrong the prediction is at this specific step |
| $W$ | All model weights (shared across time steps) | The same weights are used at every step — gradients are accumulated |
| $T$ | Total number of time steps | The length of the input sequence |

### The Vanishing Gradient Problem
Because gradients are multiplied repeatedly through $W_{hh}$, they either:
* **Vanish** (shrink exponentially) → model can't learn long-range dependencies.
* **Explode** (grow exponentially) → training becomes unstable.

This is why vanilla RNNs struggle with long sequences.

---

## 4. LSTM: Long Short-Term Memory

**LSTMs** (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem with a **cell state** and three **gates**:

| Gate | Formula | Purpose |
| :--- | :--- | :--- |
| **Forget Gate** | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ | Decides what to **discard** from the cell state |
| **Input Gate** | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ | Decides what new information to **store** |
| **Output Gate** | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | Decides what to **output** from the cell state |

### Cell State Update
$$
\begin{aligned}
\tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(candidate)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(new cell state)} \\
h_t &= o_t \odot \tanh(C_t) \quad \text{(hidden output)}
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\tilde{C}_t$ | Candidate cell state | New information that could be added to the cell state (before gating) |
| $W_C$ | Weight matrix for the candidate | Learns what new information to generate |
| $b_C$ | Bias for the candidate | Allows the candidate to shift |
| $C_t$ | Current cell state | The "conveyor belt" — carries information across time steps with minimal modification |
| $C_{t-1}$ | Previous cell state | The memory from the last time step |
| $f_t$ | Forget gate output (0 to 1 per element) | Element-wise multiplier: 0 = forget completely, 1 = keep completely |
| $i_t$ | Input gate output (0 to 1 per element) | Element-wise multiplier: 0 = ignore candidate, 1 = fully accept candidate |
| $\odot$ | Hadamard (element-wise) product | Each element is multiplied independently — gates operate per-dimension |
| $o_t$ | Output gate output (0 to 1 per element) | Controls how much of the cell state is exposed to the hidden output |
| $h_t$ | Final hidden output | The visible output of the LSTM cell at this time step |

The cell state $C_t$ is the "conveyor belt" — information can flow through it with only minor linear interactions (multiply by forget gate, add input), preserving gradients over long sequences.

---

## 5. GRU: Gated Recurrent Unit

**GRUs** (Cho et al., 2014) simplify LSTMs by merging the cell state and hidden state:

| Gate | Formula | Purpose |
| :--- | :--- | :--- |
| **Update Gate** | $z_t = \sigma(W_z [h_{t-1}, x_t])$ | Decides how much of the past to keep |
| **Reset Gate** | $r_t = \sigma(W_r [h_{t-1}, x_t])$ | Decides how much past info to forget for candidate |

$$
\begin{aligned}
\tilde{h}_t &= \tanh(W [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\tilde{h}_t$ | Candidate hidden state | New information proposed for the hidden state (before gating) |
| $W$ | Shared weight matrix | Learns both the candidate generation and gate behaviors |
| $r_t$ | Reset gate output | Controls how much of the past hidden state to forget when generating the candidate |
| $z_t$ | Update gate output | Controls the blend between old state ($1 - z_t$) and new candidate ($z_t$) |
| $\odot$ | Hadamard (element-wise) product | Element-wise multiplication — gates operate per-dimension |
| $(1 - z_t) \odot h_{t-1}$ | Retained portion of old state | What we keep from the past |
| $z_t \odot \tilde{h}_t$ | New information added | What we accept from the candidate |

> GRUs have **fewer parameters** than LSTMs (2 gates vs 3) and often perform comparably.

---

## 6. LSTM vs. GRU vs. Vanilla RNN

| Feature | Vanilla RNN | LSTM | GRU |
| :--- | :--- | :--- | :--- |
| Parameters | Fewest | Most | Middle |
| Long-range memory | Poor | Excellent | Good |
| Training speed | Fastest | Slowest | Middle |
| When to use | Very short sequences | Complex long sequences | Good default choice |

---

## 7. Applications

| Task | Input | Output | Architecture |
| :--- | :--- | :--- | :--- |
| Sentiment Analysis | Sequence | Single label | Many-to-one |
| Machine Translation | Sequence | Sequence | Many-to-many |
| Time Series Forecasting | Sequence | Future values | Many-to-one or many-to-many |
| Text Generation | Single char/word | Next char/word | One-to-many (autoregressive) |
| Speech Recognition | Audio features | Text | Many-to-many |

---

## 8. Code Example: LSTM for Sequence Classification

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
 def __init__(self, input_size, hidden_size, num_classes):
 super().__init__()
 self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
 self.fc = nn.Linear(hidden_size, num_classes)

 def forward(self, x):
 # x shape: (batch, seq_len, input_size)
 output, (h_n, c_n) = self.lstm(x)
 # Use last hidden state
 last_hidden = h_n[-1] # (batch, hidden_size)
 return self.fc(last_hidden)

# Example: classify sequences of length 20, 5 features each
model = LSTMClassifier(input_size=5, hidden_size=64, num_classes=3)
x = torch.randn(32, 20, 5) # batch=32, seq_len=20, features=5
out = model(x)
print(f"Output shape: {out.shape}") # (32, 3)
```

---

## 9. The Shift to Transformers

While LSTMs/GRUs were state-of-the-art for sequences until ~2017, **Transformers** (Vaswani et al., "Attention is All You Need") now dominate:

| RNN | Transformer |
| :--- | :--- |
| Sequential processing — $O(n)$ time steps | Parallel processing — $O(1)$ with attention |
| Struggles with very long sequences | Handles long sequences via self-attention |
| Hidden state compresses all history | Attention looks at all positions simultaneously |

> Transformers are covered in advanced courses. For most NLP and sequence tasks in 2026, Transformers are the default. RNNs remain relevant for streaming/real-time data and edge deployment.

---

## 10. Advantages & Disadvantages

### Pros
* Naturally handles variable-length sequences.
* Captures temporal dependencies.
* LSTM/GRU solve vanishing gradients for long sequences.
* Applicable to virtually any sequential data.

### Cons
* Sequential processing is hard to parallelize (slow on long sequences).
* Struggles with sequences >500 steps (Transformers are better).
* Prone to overfitting on small datasets.
* Hyperparameter-sensitive (hidden size, number of layers, learning rate).

---

**Previous:** [CNN](../CNN/Theory.md) | **Related:** [ANN](../../02-Supervised-Learning/ARTIFICIAL%20NEURAL%20NETWORKS/Theory.md) | **Related:** [Logistic Regression](../../02-Supervised-Learning/LOGISTIC%20REGRESSION/Theory.md)
