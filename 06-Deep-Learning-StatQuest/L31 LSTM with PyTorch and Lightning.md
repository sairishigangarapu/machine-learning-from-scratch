## L31: LSTM with PyTorch + Lightning

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. LSTM Recap

### Motivation and Intuition

Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem of vanilla RNNs by introducing a **cell state** and **gates** that control information flow. The key equations for one LSTM cell at time step $t$:

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

| Gate | Symbol | Function | Effect |
| :--- | :--- | :--- | :--- |
| Forget gate | $f_t$ | How much of old cell state to keep | Controls long-term memory erasure |
| Input gate | $i_t$ | How much new candidate to add | Controls new information injection |
| Candidate | $\tilde{C}_t$ | New candidate values | Proposed update to cell state |
| Cell state | $C_t$ | Long-term memory | Carries information across time steps |
| Output gate | $o_t$ | How much cell state to expose | Controls hidden state output |
| Hidden state | $h_t$ | Short-term memory / output | Passed to next time step and layer |

In PyTorch, the LSTM layer handles all of this internally:

```python
import torch.nn as nn

lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True)
# input shape: (batch, seq_len, input_size)
# output shape: (batch, seq_len, hidden_size)
```

---

## 2. Data Preparation for Sequences

### Motivation and Intuition

Sequential data must be shaped into (batch, sequence_length, features). For time series forecasting, we create sliding windows.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y

# Example: sine wave
import numpy as np
t = np.linspace(0, 100, 1000)
data = np.sin(t) + 0.1 * np.random.randn(1000)

dataset = SequenceDataset(data.reshape(-1, 1), seq_length=20)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Each batch: X shape (32, 20, 1), y shape (32, 1)
```

For text data, tokens are converted to indices, then embedded:

```python
embedding = nn.Embedding(vocab_size, embedding_dim)
# X shape (batch, seq_len) -> (batch, seq_len, embedding_dim)
X_emb = embedding(X)
```

---

## 3. LSTM with LightningModule

### Motivation and Intuition

Wrapping the LSTM in a LightningModule keeps the code organized and gives us all the training automation.

```python
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        # Take the last time step's output
        last_out = out[:, -1, :]          # (batch, hidden_size)
        return self.fc(last_out)          # (batch, output_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)
```

### Key Design Choices

| Choice | Reason |
| :--- | :--- |
| `batch_first=True` | Input shape is (batch, seq, features) — more intuitive |
| `self.save_hyperparameters()` | Saves all init args for checkpoint loading |
| `out[:, -1, :]` | Takes the final time step's hidden state as the prediction |
| `dropout` | Regularization between LSTM layers |
| `predict_step` | Used by `trainer.predict()` |

---

## 4. Training the LSTM

### Motivation and Intuition

With Lightning, training is a few lines of code.

```python
from torch.utils.data import DataLoader, random_split

# Split data
dataset = SequenceDataset(data.reshape(-1, 1), seq_length=20)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

# Model
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1, lr=0.001)

# Trainer
trainer = L.Trainer(
    max_epochs=50,
    accelerator="auto",
    log_every_n_steps=10,
    enable_checkpointing=True,
)

trainer.fit(model, train_dl, val_dl)
```

---

## 5. Making Predictions (Inference)

### Motivation and Intuition

For multi-step forecasting, we feed predictions back as inputs (autoregressive generation).

```python
# Single-step prediction
model.eval()
with torch.no_grad():
    x, y = next(iter(val_dl))
    pred = model(x)

# Multi-step autoregressive prediction
def predict_future(model, seed_sequence, steps):
    model.eval()
    current_seq = seed_sequence.clone()   # (1, seq_len, 1)
    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            pred = model(current_seq)           # (1, 1)
            predictions.append(pred.item())
            # Shift window: drop first, append prediction
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                pred.unsqueeze(1)               # (1, 1, 1)
            ], dim=1)

    return predictions

seed = dataset[-20:].unsqueeze(0)           # (1, 20, 1)
future = predict_future(model, seed, 30)
```

### Using the Lightning Trainer for Prediction

```python
predictions = trainer.predict(model, dataloaders=val_dl)
predictions = torch.cat(predictions)
```

---

## 6. Complete Workflow: Sine Wave Forecasting

```python
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class SineDataset(Dataset):
    def __init__(self, seq_len=20):
        t = np.linspace(0, 100, 2000)
        data = np.sin(t) + 0.05 * np.random.randn(2000)
        self.data = torch.tensor(data, dtype=torch.float32).view(-1, 1)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len]

class LSTMModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

dataset = SineDataset()
train_ds, val_ds = random_split(dataset, [0.8, 0.2])
train_dl, val_dl = DataLoader(train_ds, 32, shuffle=True), DataLoader(val_ds, 32)

model = LSTMModel()
trainer = L.Trainer(max_epochs=30, accelerator="auto")
trainer.fit(model, train_dl, val_dl)
```

---

> **Check your intuition:** Why do we take `out[:, -1, :]` rather than using `h_n` (the final hidden state from the last layer)? When might they differ?

---

## Prerequisites and Further Reading

- **StatQuest:** Recurrent Neural Networks (L15), LSTM (L16), Introduction to PyTorch (L29), Coding NN with Lightning (L30)
- **PyTorch docs:** `nn.LSTM`, `nn.RNN`, `nn.GRU`
- **Paper:** Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
