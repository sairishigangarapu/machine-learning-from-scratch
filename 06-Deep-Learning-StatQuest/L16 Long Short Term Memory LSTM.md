## Long Short Term Memory (LSTM)

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Long Short-Term Memory Networks

### Motivation and Intuition

Vanilla RNNs suffer from the vanishing gradient problem — they struggle to learn dependencies that span many time steps. LSTMs (Hochreiter & Schmidhuber, 1997) solve this by introducing a dedicated **cell state** $\mathbf{c}_t$ that acts as a memory highway. Information can flow through the cell state with minimal modification, while three **gates** (forget, input, output) control what to keep, what to write, and what to read. This gating mechanism allows gradients to flow backward through time without vanishing.

### LSTM Architecture

The LSTM maintains two states at each time step:
- **Cell state** $\mathbf{c}_t$: The long-term memory, carried forward with linear interactions.
- **Hidden state** $\mathbf{h}_t$: The short-term memory / output for this time step.

### The Four Gate Formulas

All gates take the same input: concatenation of $\mathbf{h}_{t-1}$ and $\mathbf{x}_t$.

**Forget Gate:** What to discard from the cell state.

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

**Input Gate:** What new information to store.

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

**Candidate Cell State:** The new candidate values to add.

$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
$$

**Cell State Update:** Combine forget and input gates.

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

**Output Gate:** What to output from the cell state.

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

**Hidden State:** The filtered cell state.

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{f}_t$ | Forget gate (0 to 1) | Controls how much of $\mathbf{c}_{t-1}$ to retain |
| $\mathbf{i}_t$ | Input gate (0 to 1) | Controls how much of $\tilde{\mathbf{c}}_t$ to add |
| $\tilde{\mathbf{c}}_t$ | Candidate cell state | New information proposed by the current input |
| $\mathbf{c}_t$ | Cell state | Long-term memory; flows with linear updates |
| $\mathbf{o}_t$ | Output gate (0 to 1) | Controls how much of $\mathbf{c}_t$ to expose as $\mathbf{h}_t$ |
| $\mathbf{h}_t$ | Hidden state | Short-term memory / output for this time step |
| $\sigma$ | Sigmoid activation | Squashes to (0, 1) — acts as a soft switch |
| $\tanh$ | Hyperbolic tangent | Squashes to (-1, 1) — normalizes values |
| $\odot$ | Element-wise multiplication | Gating mechanism |

### How LSTM Solves the Vanishing Gradient

The cell state $\mathbf{c}_t$ is updated via **addition** (not multiplication):

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

When the forget gate $\mathbf{f}_t$ is close to 1, the gradient of $\mathbf{c}_t$ w.r.t. $\mathbf{c}_{t-1}$ is approximately 1. This means the gradient can flow backward through many time steps without vanishing or exploding. The additive (rather than multiplicative) recurrence is the key innovation.

### Peephole Connections

A common LSTM variant adds "peephole" connections that let the gates look at the cell state directly:

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{p}_f \odot \mathbf{c}_{t-1} + \mathbf{b}_f)
$$

where $\mathbf{p}_f$ is a learned peephole weight vector for the forget gate.

### Python Code: LSTM with PyTorch

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, (h_n, c_n) = self.lstm(x)
        # out: [B, seq_len, hidden_size]
        # h_n: [num_layers, B, hidden_size]
        # c_n: [num_layers, B, hidden_size]
        out = self.fc(out[:, -1, :])  # last time step
        return out
```

---

> **Check your intuition:** If the forget gate outputs are $\mathbf{f}_t = [0.99, 0.01, 0.50]$, what does this tell us about each component of the cell state?

<details>
<summary>Answer</summary>
The first component keeps 99% of old memory, the second discards almost all, and the third keeps half.
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L15 Recurrent Neural Networks, vanishing gradient problem.
- **Next:** L17 Word Embedding and Word2Vec.
