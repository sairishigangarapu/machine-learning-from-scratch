## Recurrent Neural Networks

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Recurrent Neural Networks

### Motivation and Intuition

Standard neural networks and CNNs assume inputs are independent and fixed-size. But many problems involve sequences: stock prices over time, sentences, audio, or video. Recurrent Neural Networks (RNNs) handle sequential data by maintaining a **hidden state** that carries information from one time step to the next. The same weights are used at every time step (weight sharing), which keeps the parameter count manageable regardless of sequence length.

### RNN Forward Pass

At each time step $t$, the RNN takes the current input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and computes:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

The output (if needed at each step) is:

$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}_t$ | Input at time step $t$ | Current element of the sequence |
| $\mathbf{h}_t$ | Hidden state at time step $t$ | Network's "memory" of past inputs |
| $\mathbf{W}_{xh}$ | Weight matrix from input to hidden | Projects current input into hidden space |
| $\mathbf{W}_{hh}$ | Weight matrix from hidden to hidden (recurrent) | Governs how past state influences current state |
| $\mathbf{W}_{hy}$ | Weight matrix from hidden to output | Produces prediction from hidden state |
| $\mathbf{b}_h, \mathbf{b}_y$ | Bias vectors | Shift activations |

### Unfolding in Time

The recurrence defines a deep network when "unfolded" across time. Backpropagation through time (BPTT) computes gradients by unrolling the computation graph and applying the chain rule backward across all time steps.

### Shared Weights Across Time Steps

Crucially, $\mathbf{W}_{hh}$, $\mathbf{W}_{xh}$, and $\mathbf{W}_{hy}$ are the **same** at every time step. This weight sharing is what makes RNNs generalizable to sequences of any length.

### The Vanishing Gradient Problem

When using BPTT, the gradient of the loss at time $t$ with respect to the hidden state at time $k \ll t$ involves repeated multiplication by $\mathbf{W}_{hh}^T$ (or the Jacobian of tanh). If the eigenvalues of $\mathbf{W}_{hh}$ are less than 1, gradients shrink exponentially — they **vanish**. If they are larger than 1, gradients grow exponentially — they **explode**. The vanishing gradient problem makes it difficult for vanilla RNNs to learn long-range dependencies.

### Python Code: Simple RNN with PyTorch

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, h_n = self.rnn(x)     # out: [B, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])  # take last time step
        return out
```

---

> **Check your intuition:** If an RNN has a hidden state size of 128 and an input size of 10, how many parameters are in $\mathbf{W}_{hh}$ and $\mathbf{W}_{xh}$ combined?

<details>
<summary>Answer</summary>
$\mathbf{W}_{hh}$ is $128 \times 128 = 16,384$ and $\mathbf{W}_{xh}$ is $128 \times 10 = 1,280$, total $17,664$.
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L02 Neural Networks Part 1, L06-L07 Backpropagation Details.
- **Next:** L16 Long Short Term Memory LSTM.
