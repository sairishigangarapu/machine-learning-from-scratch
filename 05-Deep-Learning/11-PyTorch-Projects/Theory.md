# PyTorch Projects

*Exam-ready theory: from tensors to a ChatGPT-like transformer*

---

## 1. PyTorch Fundamentals -- Tensors, Operations, and GPU

### Motivation and Intuition

PyTorch is the most popular deep learning framework for research and production. Its core data structure, the `torch.Tensor`, is like a NumPy array but with two critical additions: GPU acceleration and automatic differentiation (autograd). Every neural network is built on these two capabilities.

### Formal Definition

A `torch.Tensor` is a multi-dimensional array with:
- **Shape**: dimensions of the array
- **Dtype**: data type (`float32`, `int64`, etc.)
- **Device**: `cpu` or `cuda:N` (GPU)
- **Requires grad**: whether autograd tracks operations

```python
import torch

# Creating tensors
t1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
t2 = torch.zeros(3, 4)          # all zeros
t3 = torch.ones(2, 5)           # all ones
t4 = torch.randn(4, 4)          # standard normal
t5 = torch.arange(12).reshape(3, 4)  # 0..11 reshaped

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    t_gpu = t1.to(device)
```

| Concept | Code | Explanation |
| :--- | :--- | :--- |
| Shape access | `tensor.shape` | Returns `torch.Size` tuple |
| Data type | `tensor.dtype` | `torch.float32`, `torch.int64`, etc. |
| Device | `tensor.device` | `cpu` or `cuda:N` |
| Number of elements | `tensor.numel()` | Product of all dimensions |
| Move to GPU | `tensor.to("cuda")` | Also `.cuda()` -- prefer `.to(device)` |

### Autograd -- Automatic Differentiation

When `requires_grad=True`, PyTorch records all operations to build a computational graph. Calling `.backward()` on a scalar computes gradients for all leaf tensors.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dy/dx = 2x + 3 evaluated at [1,2,3] -> [5., 7., 9.]
```

| Mechanism | Purpose |
| :--- | :--- |
| `requires_grad` | Flag to enable gradient tracking |
| `.backward()` | Compute gradients via reverse-mode AD |
| `.grad` | Accumulated gradient tensor |
| `.zero_()` | Reset gradients (call every step) |
| `torch.no_grad()` | Disable tracking (inference/validation) |
| `.detach()` | Create new tensor disconnected from graph |

### Tensor Operations

| Operation | Code | Shape Rule |
| :--- | :--- | :--- |
| Element-wise add | `a + b` | Broadcastable shapes |
| Matrix multiply | `a @ b` | `(...,m,k) @ (...,k,n) -> (...,m,n)` |
| Reshape | `a.view(3, 4)` | Total numel must match |
| Transpose | `a.T` or `a.transpose(i,j)` | 2D or any dims |
| Concatenate | `torch.cat([a, b], dim=0)` | All dims except specified match |

---

## 2. nn.Module -- Defining Neural Networks

### Motivation and Intuition

All neural network layers and models in PyTorch inherit from `nn.Module`. It provides parameter management, training/eval mode switching, and device movement. You define layers in `__init__` and the forward pass in `forward`.

### Formal Definition

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(10, 32, 1)
print(model)
```

| Component | Description |
| :--- | :--- |
| `nn.Linear(in, out)` | Fully connected: `y = xW^T + b` |
| `nn.ReLU()` | `max(0, x)` activation |
| `nn.Sequential(*layers)` | Chain layers without writing forward |
| `model.parameters()` | Iterator over all trainable parameters |
| `model.train()` / `.eval()` | Toggle dropout/batch norm behavior |
| `model.to(device)` | Move all parameters to device |

### Parameter Management

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")  # fc1.weight: (32, 10), fc1.bias: (32,), ...

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

---

## 3. Training Loop -- Zero Grad, Backward, Step

### Motivation and Intuition

The training loop is the core of every deep learning experiment. The standard pattern: forward pass, compute loss, zero gradients, backward pass, update weights.

### Formal Definition

```python
import torch.optim as optim

epochs = 1000
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    # Forward
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log
    if epoch % 100 == 0:
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)
        print(f"Epoch {epoch}: train={loss.item():.4f}, val={val_loss.item():.4f}")
```

| Step | Purpose |
| :--- | :--- |
| `optimizer.zero_grad()` | Clear gradients from previous iteration |
| `loss.backward()` | Compute new gradients (populates `.grad`) |
| `optimizer.step()` | Update parameters using gradients |
| `torch.no_grad()` | Disable graph construction for validation |

### Loss Functions and Optimizers

| Loss Function | Use Case |
| :--- | :--- |
| `nn.MSELoss()` | Regression |
| `nn.CrossEntropyLoss()` | Multi-class classification |
| `nn.BCEWithLogitsLoss()` | Binary classification |
| `nn.L1Loss()` | Mean absolute error regression |

| Optimizer | Key Feature |
| :--- | :--- |
| `optim.SGD` | + momentum |
| `optim.Adam` | Adaptive LR per parameter |
| `optim.AdamW` | Adam with decoupled weight decay |

---

## 4. PyTorch Lightning -- Structured Training

### Motivation and Intuition

PyTorch Lightning removes training boilerplate by organizing code into a `LightningModule` (what to do) and a `Trainer` (how to do it). The same code scales from CPU to multi-GPU to TPU without changes.

### Formal Definition

```python
import lightning as L

class MyModel(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

model = MyModel(1, 10, 1)
trainer = L.Trainer(max_epochs=100, accelerator="auto")
trainer.fit(model, train_dataloader, val_dataloader)
```

| `LightningModule` Method | When It Runs | What to Define |
| :--- | :--- | :--- |
| `training_step` | Every batch | Forward + loss computation |
| `validation_step` | Every val interval | Validation logic |
| `configure_optimizers` | Start of training | Return optimizer(s) |
| `forward` | Inference | Model call |
| `test_step` | `trainer.test()` | Testing logic |

| `Trainer` Argument | Effect |
| :--- | :--- |
| `max_epochs` | Number of epochs |
| `accelerator="auto"` | Auto-detect GPU/CPU |
| `devices=4` | Use 4 GPUs |
| `precision=16` | Mixed precision training |
| `callbacks=[...]` | EarlyStopping, ModelCheckpoint |

---

## 5. Project 1 -- Neural Network from Scratch (Regression)

### Motivation and Intuition

A simple neural network with one hidden layer can approximate any continuous function (universal approximation theorem). This project builds and trains one on synthetic data.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Synthetic data: y = 2x + 1 + noise
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 1 + 0.1 * torch.randn_like(X)

# Model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

| Component | Value |
| :--- | :--- |
| Architecture | Linear(1,16) -> ReLU -> Linear(16,1) |
| Loss | MSELoss |
| Optimizer | Adam, lr=0.01 |
| Epochs | 1000 |
| Expected | Model learns y ~= 2x + 1 |

---

## 6. Project 2 -- LSTM for Sequence Prediction

### Motivation and Intuition

LSTMs capture temporal dependencies via a gated cell state that mitigates vanishing gradients. They excel at time series forecasting, text generation, and any sequential data.

```python
import lightning as L
import torch.nn as nn
import torch.optim as optim

class LSTMPredictor(L.LightningModule):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)             # (B, seq_len, hidden_size)
        return self.fc(out[:, -1, :])      # take last time step

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
```

| Data Shape | Meaning |
| :--- | :--- |
| `(batch, seq_len, input_size)` | Input to LSTM (batch_first=True) |
| `(batch, output_size)` | Prediction from last time step |
| `(batch, seq_len, hidden_size)` | Full output sequence (all time steps) |

### Sine Wave Forecasting Workflow

1. Create sliding windows from sine wave data
2. Train LSTM to predict next value from past 20 steps
3. Use autoregressive generation for multi-step forecasts

---

## 7. Project 3 -- Word Embedding (nn.Embedding)

### Motivation and Intuition

`nn.Embedding` is a lookup table mapping token IDs to dense vectors. It is the first layer of almost every NLP model. The embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$ is learned during training.

```python
vocab_size, emb_dim = 10000, 128
embed = nn.Embedding(vocab_size, emb_dim)

# Input: token indices (batch, seq_len)
tokens = torch.randint(0, vocab_size, (32, 20))   # (B, T)
embedded = embed(tokens)                            # (B, T, emb_dim)
```

| Property | Description |
| :--- | :--- |
| `nn.Embedding(V, D)` | Lookup table: V vocab, D embedding dim |
| Input shape | `(..., seq_len)` of integer indices |
| Output shape | `(..., seq_len, D)` of dense vectors |
| Pretrained | `nn.Embedding.from_pretrained(weights, freeze=True)` |

### Training Embeddings from Scratch

Embedding layers are trained via backpropagation just like any other layer. Words that appear in similar contexts develop similar vector representations.

```python
class TextClassifier(L.LightningModule):
    def __init__(self, vocab_size, emb_dim, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len * emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        emb = self.embedding(x)          # (B, T, D)
        return self.classifier(emb)
```

### Embedding + LSTM (Common Pattern)

```python
class EmbeddingLSTM(L.LightningModule):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.embedding(x)              # (B, T, D)
        out, _ = self.lstm(emb)              # (B, T, H)
        return self.classifier(out[:, -1, :])
```

---

## 8. Project 4 -- Mini Transformer (ChatGPT-Like)

### Motivation and Intuition

A decoder-only transformer (the architecture behind GPT, ChatGPT, Claude) generates text autoregressively: given a sequence of tokens, predict the next token. The core operation is causal self-attention -- each token attends only to itself and previous tokens.

### Architecture Components

**Positional Encoding:** Adds position information since self-attention is permutation-invariant.

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
**Causal Self-Attention:** Each token queries previous tokens (including itself) to gather context.

$$
\text{head} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \text{mask} \right) \mathbf{V}
$$
**Decoder Block:** Causal attention -> residual + layer norm -> MLP -> residual + layer norm.

**Generation:** Feed prompt, sample next token, append, repeat.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is None:
            mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        context = A @ V
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)
```

| Component | Shape | Purpose |
| :--- | :--- | :--- |
| Token embedding | `(V, d_model)` | Maps token IDs to dense vectors |
| Positional encoding | `(max_len, d_model)` | Adds position info |
| Causal mask | `(N, N)` triangular | Prevents attending to future |
| LM head | `(d_model, V)` | Projects hidden to vocab logits |
| Softmax + sample | `(V,)` -> token ID | Probabilistic next-token selection |

### Autoregressive Generation

```python
def generate(model, tokenizer, prompt, max_tokens=50, temperature=1.0):
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    return tokenizer.decode(ids[0].tolist())
```

| Temperature | Effect |
| :--- | :--- |
| 0 (greedy) | Always pick max probability -- repetitive |
| 1.0 | Balanced creativity |
| > 1.0 | More diverse, may be incoherent |

---

## 9. Debugging Tips

### Motivation and Intuition

Neural network debugging is mostly about checking tensor shapes, gradients, and device placement. Systematic checks prevent wasted training runs.

```python
# 1. Tensor shape debugging
def debug_forward(model, x):
    for name, layer in model.named_children():
        x = layer(x)
        print(f"{name}: {x.shape}")
    return x

# 2. Gradient checking (ensure gradients flow)
loss.backward()
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"WARNING: {name} has no gradient!")
    elif param.grad.norm() == 0:
        print(f"WARNING: {name} gradient is zero!")
    elif torch.isnan(param.grad).any():
        print(f"WARNING: {name} gradient has NaN!")

# 3. Device management
assert X.device == next(model.parameters()).device
```

| Check | Tool | What to Look For |
| :--- | :--- | :--- |
| Shapes | Print after each layer | Match expected dimensions |
| Gradients | `param.grad is None` | Dead layers (no gradient flow) |
| NaN loss | `torch.isnan(loss)` | Exploding gradients, bad LR |
| Device | `.device` on tensors | All inputs and model on same device |
| Overfitting | Small subset | Model can memorize one batch |

---

## 10. Best Practices

### Motivation and Intuition

These practices separate robust, reproducible experiments from ad-hoc scripts. They save time, prevent bugs, and make results trustworthy.

### DataLoaders and Dataset

```python
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(X, y)
train_len = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
```

### Validation and Early Stopping

```python
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
early_stop = EarlyStopping(monitor="val_loss", patience=10)

trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint, early_stop],
    accelerator="auto",
)
```

### Mixed Precision

```python
trainer = L.Trainer(precision=16)   # half the memory, ~2x speed on modern GPUs
```

### Checklist

| Practice | Why |
| :--- | :--- |
| `DataLoader` with shuffle | Prevents order bias |
| `random_split` for val set | Reproducible evaluation |
| `EarlyStopping` | Prevents overfitting |
| `ModelCheckpoint` | Save best model |
| `precision=16` | Faster training, less memory |
| `seed_everything(42)` | Reproducibility |
| `self.save_hyperparameters()` | Full experiment tracking |
| `prog_bar=True` in `self.log` | Training progress visibility |

---

> **Check your intuition:** Why must you call `optimizer.zero_grad()` before each `loss.backward()`? What would the gradients look like if you omitted it for 5 iterations?

---

## Prerequisites and Further Reading

- **StatQuest:** L29 Introduction to PyTorch, L30 Coding NN with PyTorch + Lightning, L31 LSTM with PyTorch + Lightning, L32 Word Embedding in PyTorch + Lightning, L33 Coding ChatGPT Like Transformer From Scratch
- **PyTorch docs:** `nn.Module`, `torch.optim`, `torch.autograd`, `torch.utils.data`
- **Lightning docs:** LightningModule, Trainer, Callbacks, Loggers
- **Paper:** Vaswani et al., "Attention Is All You Need" (2017); Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
