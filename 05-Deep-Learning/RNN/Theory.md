# Recurrent Neural Networks (RNNs) 📝

## 1. Why RNNs?

Fully connected networks and CNNs assume inputs are **independent** and **fixed-size**. But for sequences (text, time series, speech), the order and length matter. An RNN processes sequences by maintaining a **hidden state** that acts as a memory of past inputs.

$$
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)
$$
$$
y_t = W_{hy} h_t + c
$$

| Symbol | Meaning |
| :--- | :--- |
| $x_t$ | Input at time step $t$ |
| $h_t$ | Hidden state at time step $t$ (the "memory") |
| $y_t$ | Output at time step $t$ |
| $W_{hh}$ | Recurrent weights (hidden → hidden) |
| $W_{xh}$ | Input weights (input → hidden) |

---

## 2. Unrolling Through Time

The RNN is **unrolled** across time steps, revealing that it is essentially a deep network where each layer shares the same weights:

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → output
        ↑              ↑              ↑
       W_hh           W_hh           W_hh    (same weights)
```

---

## 3. Training: Backpropagation Through Time (BPTT)

RNNs are trained by unrolling the network and applying backpropagation across all time steps. The gradient at time step $t$ depends on all previous time steps:

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
$$

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
\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(candidate)}
$$
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(new cell state)}
$$
$$
h_t = o_t \odot \tanh(C_t) \quad \text{(hidden output)}
$$

The cell state $C_t$ is the "conveyor belt" — information can flow through it with only minor linear interactions (multiply by forget gate, add input), preserving gradients over long sequences.

---

## 5. GRU: Gated Recurrent Unit

**GRUs** (Cho et al., 2014) simplify LSTMs by merging the cell state and hidden state:

| Gate | Formula | Purpose |
| :--- | :--- | :--- |
| **Update Gate** | $z_t = \sigma(W_z [h_{t-1}, x_t])$ | Decides how much of the past to keep |
| **Reset Gate** | $r_t = \sigma(W_r [h_{t-1}, x_t])$ | Decides how much past info to forget for candidate |

$$
\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

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
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.fc(last_hidden)

# Example: classify sequences of length 20, 5 features each
model = LSTMClassifier(input_size=5, hidden_size=64, num_classes=3)
x = torch.randn(32, 20, 5)  # batch=32, seq_len=20, features=5
out = model(x)
print(f"Output shape: {out.shape}")  # (32, 3)
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

### ✅ Pros
* Naturally handles variable-length sequences.
* Captures temporal dependencies.
* LSTM/GRU solve vanishing gradients for long sequences.
* Applicable to virtually any sequential data.

### ❌ Cons
* Sequential processing is hard to parallelize (slow on long sequences).
* Struggles with sequences >500 steps (Transformers are better).
* Prone to overfitting on small datasets.
* Hyperparameter-sensitive (hidden size, number of layers, learning rate).

---

**Previous:** [CNN](../CNN/Theory.md) | **Related:** [ANN](../../02-Supervised-Learning/ARTIFICIAL%20NEURAL%20NETWORKS/Theory.md) | **Related:** [Logistic Regression](../../02-Supervised-Learning/LOGISTIC%20REGRESSION/Theory.md)
