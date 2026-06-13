## L26: Tensors for Neural Networks

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What is a Tensor?

### Motivation and Intuition

In machine learning, a **tensor** is simply a container for numbers arranged on a regular grid. The number of axes (dimensions) determines its name:

| Tensor Type | Dimensions | Shape | Example |
| :--- | :--- | :--- | :--- |
| **Scalar** | 0D | `()` | `tensor(5)` |
| **Vector** | 1D | `(n,)` | `[1, 2, 3]` |
| **Matrix** | 2D | `(m, n)` | `[[1,2], [3,4]]` |
| **3-Tensor** | 3D | `(a, b, c)` | A batch of images: `(32, 28, 28)` |
| **4-Tensor** | 4D | `(a, b, c, d)` | Batch of color images: `(32, 3, 28, 28)` |

In PyTorch, everything is a `torch.Tensor`. A neural network's weights, input data, and intermediate activations are all tensors.

---

## 2. Tensor Operations

### Motivation and Intuition

Neural network computations reduce to a handful of tensor operations. Mastering these lets you read any PyTorch code.

### Reshape

Changes the shape without changing data order (view / reshape):

```python
import torch
x = torch.arange(12)           # shape (12,)
x_2d = x.reshape(3, 4)         # shape (3, 4)
x_3d = x.reshape(2, 2, 3)      # shape (2, 2, 3)
x_flat = x_3d.reshape(-1)      # shape (12,) — infer dimension
```

### Permute / Transpose

Reorders axes:

```python
x = torch.randn(2, 3, 5)       # shape (2, 3, 5)
y = x.permute(2, 0, 1)         # shape (5, 2, 3)
z = x.T                         # transpose (2D only): (5, 3) if x is 2D
```

### Matrix Multiplication

| Operation | Code | Shape Rule |
| :--- | :--- | :--- |
| Element-wise multiply | `a * b` | shapes must match or be broadcastable |
| Matrix multiply | `a @ b` or `torch.matmul(a, b)` | `(..., m, k) @ (..., k, n) -> (..., m, n)` |
| Batch matrix multiply | `torch.bmm(a, b)` | `(b, m, k) @ (b, k, n) -> (b, m, n)` |
| Dot product | `torch.dot(a, b)` | both 1D, same length |
| Einsum | `torch.einsum("ij,jk->ik", a, b)` | flexible notation |

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = a @ b  # tensor([[19, 22], [43, 50]])
```

### Broadcasting

PyTorch automatically expands dimensions when shapes differ:

```python
a = torch.tensor([[1], [2], [3]])    # shape (3, 1)
b = torch.tensor([10, 20, 30])       # shape (3,) -> broadcast to (1, 3) -> (3, 3)
c = a + b                            # shape (3, 3)
```

---

## 3. Tensors in PyTorch

### Creating Tensors

```python
# From data
t = torch.tensor([[1, 2], [3, 4]])

# Special tensors
z = torch.zeros(3, 4)       # all zeros
o = torch.ones(2, 5)        # all ones
r = torch.randn(3, 3)       # standard normal
e = torch.eye(4)            # identity matrix

# Like another tensor
x = torch.rand_like(t)      # same shape as t
```

### Tensor Attributes

```python
t.shape       # torch.Size([2, 2])
t.dtype       # torch.int64 (default for int)
t.device      # device(type='cpu')
t.requires_grad  # False by default
```

### Data Types

```python
t_float = t.float()                    # convert to float32
t_half  = t.half()                     # float16 (for GPU speed)
t_long  = t.long()                     # int64
t_double = t.double()                  # float64
```

---

## 4. GPU Tensors

### Motivation and Intuition

GPUs have thousands of cores optimized for parallel matrix operations. Moving tensors to a GPU accelerates training by 10–100x.

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    t_gpu = t.to(device)               # move to GPU
    t_cpu = t_gpu.cpu()                # move back to CPU
else:
    device = torch.device("cpu")
```

For multiple GPUs:

```python
t_gpu = t.cuda(0)       # first GPU
t_gpu = t.cuda(1)       # second GPU
```

Operations between tensors require all operands to be on the same device.

---

## 5. Automatic Differentiation

### Motivation and Intuition

Neural network training requires gradients of the loss with respect to every parameter. PyTorch's `autograd` records operations on tensors with `requires_grad=True` and automatically computes gradients.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
z = y.sum()             # scalar output
z.backward()            # compute gradients
x.grad                  # tensor([2., 4., 6.])
```

| Method | What it does |
| :--- | :--- |
| `.backward()` | Computes gradients of the scalar output w.r.t. all `requires_grad` tensors |
| `.detach()` | Creates a new tensor that does **not** track gradients |
| `.grad.zero_()` | Resets gradients (must do after each optimizer step) |
| `torch.no_grad()` | Context manager to disable gradient tracking (inference, validation) |

### Gradient Accumulation Graph

$$
\text{forward: } z = f(y),\; y = g(x) \quad\longrightarrow\quad \text{backward: } \frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| `requires_grad` | Flag enabling gradient tracking for a tensor | Only set on parameters you train |
| `grad_fn` | Stores the operation that created a tensor | Enables the backward graph traversal |
| `.grad` | Accumulated gradient after `.backward()` | Used by the optimizer to update weights |
| **computational graph** | Directed acyclic graph of operations | Built on forward pass, consumed on backward |

---

> **Check your intuition:** Why can't you call `.backward()` on a non-scalar tensor? What would you need to pass as an argument?

---

## Prerequisites and Further Reading

- **StatQuest:** Neural Networks Part 0 (L01), Essential Matrix Algebra for Neural Networks (L27)
- **PyTorch docs:** `torch.Tensor`, `torch.autograd`, CUDA semantics
- **Concepts:** Matrix multiplication, broadcasting rules, computational graphs
