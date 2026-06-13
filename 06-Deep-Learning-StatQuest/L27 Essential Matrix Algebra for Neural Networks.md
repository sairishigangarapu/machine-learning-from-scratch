## L27: Essential Matrix Algebra for Neural Networks

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Matrix Multiplication for the Forward Pass

### Motivation and Intuition

Every neural network layer computes $\mathbf{h} = \mathbf{x} \mathbf{W} + \mathbf{b}$. This single matrix equation replaces dozens of scalar equations. Understanding matrix multiplication is the key to reading all neural network code and documentation.

For a layer with $n$ inputs and $m$ outputs:

$$
\mathbf{h} = \mathbf{x} \mathbf{W} + \mathbf{b}
$$

where $\mathbf{x} \in \mathbb{R}^{1 \times n}$, $\mathbf{W} \in \mathbb{R}^{n \times m}$, $\mathbf{b} \in \mathbb{R}^{1 \times m}$, and $\mathbf{h} \in \mathbb{R}^{1 \times m}$.

Expanded for a batch of $B$ samples:

$$
\mathbf{H} = \mathbf{X} \mathbf{W} + \mathbf{b}
$$

where $\mathbf{X} \in \mathbb{R}^{B \times n}$, $\mathbf{H} \in \mathbb{R}^{B \times m}$, and broadcasting adds $\mathbf{b}$ to each row.

### Matrix Multiplication Rule

For $\mathbf{C} = \mathbf{A} \mathbf{B}$:

- $\mathbf{A}$ must have shape $(m \times k)$
- $\mathbf{B}$ must have shape $(k \times n)$
- Result $\mathbf{C}$ has shape $(m \times n)$
- Each element $c_{ij} = \sum_{t=1}^{k} a_{it} \cdot b_{tj}$

```python
import torch
x = torch.tensor([[0.5, -0.2, 0.1]])          # shape (1, 3)
W = torch.tensor([[0.4, -0.3],
                  [0.2, 0.5],
                  [-0.1, 0.6]])               # shape (3, 2)
b = torch.tensor([[0.1, -0.2]])               # shape (1, 2)
h = x @ W + b                                 # shape (1, 2)
```

---

## 2. Broadcasting

### Motivation and Intuition

Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions. This keeps code clean and memory efficient.

**Rules:** PyTorch compares dimensions from right to left. A dimension matches if:
1. The dimensions are equal, or
2. One dimension is 1 (it is stretched to match)

```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])                # shape (2, 3)
b = torch.tensor([10, 20, 30])               # shape (3,) -> broadcast to (1, 3) -> (2, 3)
c = a + b                                    # shape (2, 3)

# Bias addition
h = x @ W + b                                # b shape (1, m) broadcasts to (B, m)
```

---

## 3. Transposition

### Motivation and Intuition

Transposing flips rows and columns. It is used to align dimensions for multiplication, especially for dot products in attention mechanisms.

$$
\mathbf{A}^\top_{ij} = \mathbf{A}_{ji}
$$

```python
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])    # shape (2, 3)
A_T = A.T                        # shape (3, 2)

# For batched tensors:
x = torch.randn(4, 8, 16)
y = x.transpose(1, 2)            # shape (4, 16, 8)
z = x.permute(0, 2, 1)           # same result
```

---

## 4. Dot Product for Attention

### Motivation and Intuition

The dot product measures similarity between two vectors. In attention, query and key vectors are compared via dot product to determine how much each token should attend to others.

$$
\mathbf{q} \cdot \mathbf{k} = \sum_{i} q_i k_i = \mathbf{q} \mathbf{k}^\top
$$

For matrices of queries $\mathbf{Q} \in \mathbb{R}^{N \times d_k}$ and keys $\mathbf{K} \in \mathbb{R}^{M \times d_k}$:

$$
\text{Scores} = \mathbf{Q} \mathbf{K}^\top \quad \text{shape } (N, M)
$$

```python
q = torch.tensor([0.5, -0.2, 0.1])
k = torch.tensor([0.3, 0.8, -0.4])
score = torch.dot(q, k)          # scalar

# Batched:
Q = torch.randn(4, 8, 64)        # (batch, seq_len, d_k)
K = torch.randn(4, 6, 64)
scores = Q @ K.transpose(-2, -1) # (4, 8, 6)
```

---

## 5. Batch Matrix Multiplication

### Motivation and Intuition

When processing a batch of $B$ independent sequences, each with its own attention matrix, we need batched matrix multiplication. `torch.bmm` (or `@` with broadcasting) handles this.

```python
# bmm: batch matrix multiply
A = torch.randn(4, 8, 16)        # (B, n, k)
B = torch.randn(4, 16, 32)       # (B, k, m)
C = torch.bmm(A, B)              # (B, n, m)

# Equivalent with @ (preferred in modern code)
C = A @ B                        # (B, n, m)
```

---

## 6. Einsum Notation

### Motivation and Intuition

Einsum provides a compact, readable way to express any tensor contraction. It uses subscript labels to define the operation axis by axis.

```python
# Matrix multiplication: ij,jk->ik
C = torch.einsum("ij,jk->ik", A, B)

# Batched matmul: bij,bjk->bik
C = torch.einsum("bij,bjk->bik", A, B)

# Dot product: i,i-> (scalar)
s = torch.einsum("i,i->", q, k)

# Outer product: i,j->ij
O = torch.einsum("i,j->ij", a, b)

# Attention scores: bqk,bqk->bqq (batch, query_dim, key_dim -> batch, query_len, query_len)
scores = torch.einsum("bqd,bkd->bqk", Q, K)

# Weighted sum over key dimension: bqv,bvd->bqd
context = torch.einsum("bqk,bkd->bqd", weights, V)
```

| Pattern | Operation | Example Shape |
| :--- | :--- | :--- |
| `ij,jk->ik` | Matrix multiply | (3,4), (4,5) → (3,5) |
| `bij,bjk->bik` | Batch matmul | (2,3,4), (2,4,5) → (2,3,5) |
| `...ij,...jk->...ik` | Batched with ellipsis | Arbitrary batch dims |
| `bqk,bkd->bqd` | Attention output | (B, Q, K), (B, K, D) → (B, Q, D) |

---

### Summary of Key Operations

| Operation | Math | PyTorch Code | Shape Constraint |
| :--- | :--- | :--- | :--- |
| Matrix multiply | $\mathbf{C} = \mathbf{A}\mathbf{B}$ | `A @ B` | $(m,k),(k,n)\to(m,n)$ |
| Transpose | $\mathbf{A}^\top$ | `A.T` or `A.transpose(i,j)` | $(m,n)\to(n,m)$ |
| Dot product | $\mathbf{a} \cdot \mathbf{b}$ | `torch.dot(a,b)` | both length $n$ |
| Batch matmul | $\mathbf{C}_b = \mathbf{A}_b\mathbf{B}_b$ | `A @ B` | $(B,m,k),(B,k,n)\to(B,m,n)$ |
| Broadcast add | $\mathbf{H} = \mathbf{X}\mathbf{W} + \mathbf{b}$ | `X @ W + b` | $b$ shape $(1,m)$ |

---

> **Check your intuition:** If $\mathbf{X}$ has shape (64, 784) and $\mathbf{W}$ has shape (784, 256), what is the shape of $\mathbf{X}\mathbf{W}$? What about if you transpose $\mathbf{X}$ first to (784, 64)?

---

## Prerequisites and Further Reading

- **StatQuest:** Neural Networks Part 0 (L01), Backpropagation Main Ideas (L05), Tensors for Neural Networks (L26)
- **PyTorch docs:** `torch.matmul`, `torch.bmm`, `torch.einsum`, broadcasting semantics
- **Concepts:** Linear algebra basics, matrix dimensions, batched computation
