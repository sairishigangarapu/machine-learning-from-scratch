## L29: Introduction to PyTorch

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What is PyTorch?

### Motivation and Intuition

PyTorch is an open-source deep learning framework developed by Meta. It provides:
- **Tensors** — the fundamental data structure (like NumPy arrays but with GPU support)
- **Autograd** — automatic differentiation for backpropagation
- **nn.Module** — a base class for defining neural network layers and models
- **Optimizers** — SGD, Adam, etc., for gradient-based parameter updates
- **Data utilities** — `DataLoader`, `Dataset` for efficient batching

PyTorch uses **eager execution**: operations are run immediately as they are called, making debugging intuitive.

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

---

## 2. Tensors

### Motivation and Intuition

Tensors are PyTorch's version of arrays. They support all standard operations and can be moved to GPU seamlessly.

```python
# Creating tensors
data = [[1, 2], [3, 4], [5, 6]]
t = torch.tensor(data, dtype=torch.float32)
ones = torch.ones(3, 4)
zeros = torch.zeros(2, 5)
randn = torch.randn(4, 4)         # standard normal

# Operations
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
c = a + b           # addition
d = a * b           # element-wise multiply
e = a @ b           # dot product (1D) or matmul (2D+)
```

---

## 3. Autograd — Automatic Differentiation

### Motivation and Intuition

Manually computing derivatives is tedious and error-prone. Autograd records the computation graph during the forward pass and computes gradients automatically during the backward pass.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()               # computes dz/dx
print(x.grad)              # tensor([5., 7., 9.])  — dy/dx = 2x + 3 at each x_i
```

Key concepts:
- `requires_grad=True` tells PyTorch to track operations on this tensor
- `.backward()` computes gradients and accumulates them in `.grad`
- Gradients **accumulate** — call `.grad.zero_()` between optimization steps
- Use `torch.no_grad()` during inference to save memory and speed

---

## 4. Defining a Neural Network with nn.Module

### Motivation and Intuition

All models inherit from `nn.Module`. Layers are defined in `__init__`, and the forward pass is specified in `forward`.

```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(input_size=10, hidden_size=32, output_size=1)
print(model)
```

| Component | Purpose |
| :--- | :--- |
| `nn.Linear(in, out)` | Fully connected layer: $\mathbf{y} = \mathbf{x} \mathbf{W}^\top + \mathbf{b}$ |
| `nn.ReLU()` | Activation: $f(x) = \max(0, x)$ |
| `nn.Sigmoid()`, `nn.Tanh()` | Other activations |
| `nn.Dropout(p)` | Regularization by dropping neurons |
| `nn.BatchNorm1d(features)` | Normalizes activations for stable training |

---

## 5. Loss Functions and Optimizers

### Motivation and Intuition

PyTorch provides common loss functions and optimizers out of the box.

```python
# Loss functions
mse = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()
bce = nn.BCEWithLogitsLoss()      # binary CE + sigmoid combined

# Optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

| Loss | Use Case |
| :--- | :--- |
| `MSELoss` | Regression (mean squared error) |
| `CrossEntropyLoss` | Multi-class classification |
| `BCEWithLogitsLoss` | Binary classification |
| `L1Loss` | Regression (mean absolute error) |

| Optimizer | Key Feature |
| :--- | :--- |
| `SGD` | Vanilla gradient descent + momentum option |
| `Adam` | Adaptive learning rates per parameter |
| `AdamW` | Adam with decoupled weight decay |
| `RMSprop` | Often used for RNNs |

---

## 6. Training Loop

### Motivation and Intuition

The standard training loop follows a fixed sequence each epoch.

```python
epochs = 100
for epoch in range(epochs):
    # --- Forward pass ---
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # --- Backward pass ---
    optimizer.zero_grad()       # clear previous gradients
    loss.backward()             # compute gradients
    optimizer.step()            # update parameters

    # --- Logging ---
    if epoch % 10 == 0:
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)
        print(f"Epoch {epoch}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")
```

**Always use `torch.no_grad()` for validation/inference** to disable gradient computation and reduce memory usage.

---

## 7. Moving to GPU

### Motivation and Intuition

GPU acceleration is essential for training modern neural networks. With PyTorch, moving models and data between devices is a single call.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleNet(10, 32, 1).to(device)     # move model
X_train = X_train.to(device)                 # move data
y_train = y_train.to(device)

# Inside training loop — everything else stays the same
y_pred = model(X_train)                      # runs on GPU automatically
```

### Checking GPU

```python
print(torch.cuda.is_available())          # True/False
print(torch.cuda.device_count())          # number of GPUs
print(torch.cuda.get_device_name(0))      # e.g., "NVIDIA A100"
```

---

## 8. Complete Minimal Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Data: y = 2x + 1 with noise
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 1 + 0.1 * torch.randn_like(X)

# Model
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X, y = X.to(device), y.to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

---

> **Check your intuition:** Why is `optimizer.zero_grad()` necessary? What would happen if you omitted it?

---

## Prerequisites and Further Reading

- **StatQuest:** Tensors for Neural Networks (L26), Essential Matrix Algebra (L27), Backpropagation (L05–L07)
- **PyTorch docs:** `nn.Module`, `torch.optim`, `torch.autograd`, CUDA semantics
- **Tutorial:** PyTorch official "60-minute blitz"
