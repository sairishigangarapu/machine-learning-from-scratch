# Advanced Math for Deep Learning

*Exam-ready theory: from tensors to attention math and gradient flow*

---

## 1. Tensors -- Scalars, Vectors, Matrices, and Beyond

### Motivation and Intuition

Deep learning frameworks (PyTorch, TensorFlow, JAX) use tensors as their fundamental data container. A tensor is a generalization of a scalar, vector, and matrix to an arbitrary number of dimensions. Every weight, input, activation, and gradient in a neural network is stored as a tensor. Understanding how tensors work -- shape, strides, memory layout -- is the foundation for debugging, writing correct code, and reasoning about performance.

### Formal Definition

A **tensor** is a multi-dimensional array of numerical values. The number of axes is called the **rank** (or order, or number of dimensions). The **shape** is a tuple giving the size of each axis. The **stride** is the number of elements you must skip in memory to advance one step along each axis.

| Rank | Name | Example Shape | Typical Use |
| :--- | :--- | :--- | :--- |
| 0 | Scalar | `()` | Loss value, single accuracy number |
| 1 | Vector | `(d,)` | Single input sample, bias vector |
| 2 | Matrix | `(m, n)` | Weight matrix, single-layer output |
| 3 | 3-Tensor | `(B, H, W)` | Batch of grayscale images, sequence of embeddings |
| 4 | 4-Tensor | `(B, C, H, W)` | Batch of color images, batch of multi-head attention states |

$$
\text{Numel} = \prod_{i=0}^{r-1} \text{shape}[i] \qquad \text{Strides}[i] = \prod_{j=i+1}^{r-1} \text{shape}[j]
$$
where **numel** is the total number of elements and **strides** define the memory offset for each dimension.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Rank | Number of dimensions/tensor axes | Determines indexing depth (t[0] vs t[0,0]) |
| Shape | Tuple of axis lengths | Required for matrix multiplication compatibility |
| Stride | Tuple of memory step sizes | Affects performance of transposed/non-contiguous tensors |
| Contiguous | Elements stored in row-major order | Operations like `.view()` require contiguous memory |
| Numel | Total number of elements | Product of all shape entries; determines memory footprint |

### Worked Numerical Example

Consider a 2x3 matrix:

$$
\mathbf{T} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
$$
- Shape: `(2, 3)`  -- 2 rows, 3 columns
- Rank: 2
- Numel: `2 * 3 = 6`
- Strides (row-major): `(3, 1)` -- advancing one row jumps 3 elements; advancing one column jumps 1 element

In memory (contiguous): `[1, 2, 3, 4, 5, 6]`

If we transpose: `T.T` has shape `(3, 2)` and strides `(1, 3)`.

```python
import torch

# Create tensors of various ranks
scalar   = torch.tensor(3.14)          # shape ()
vector   = torch.tensor([1, 2, 3])     # shape (3,)
matrix   = torch.tensor([[1, 2], [3, 4]])  # shape (2, 2)
tensor3d = torch.randn(2, 3, 4)        # shape (2, 3, 4)

print(f"Scalar:  shape={scalar.shape},  numel={scalar.numel()}")
print(f"Vector:  shape={vector.shape},  numel={vector.numel()}")
print(f"Matrix:  shape={matrix.shape},  numel={matrix.numel()}")
print(f"3-Tensor: shape={tensor3d.shape}, numel={tensor3d.numel()}")
```

| Output | Explanation |
| :--- | :--- |
| Scalar: shape=(), numel=1 | 0D tensor, single value |
| Vector: shape=(3,), numel=3 | 1D tensor, 3 values |
| Matrix: shape=(2,2), numel=4 | 2D tensor, 4 values |
| 3-Tensor: shape=(2,3,4), numel=24 | 3D tensor, 24 values |

**ML connection:** Every batch of data in deep learning is a tensor. A batch of 64 color images of size 224x224 has shape `(64, 3, 224, 224)`. A weight matrix connecting a 512-unit layer to a 256-unit layer has shape `(512, 256)`. The output of a transformer with batch 8, sequence length 128, and embedding dimension 512 has shape `(8, 128, 512)`.

---

## 2. Matrix Operations for the Forward Pass -- Wx + b

### Motivation and Intuition

Every fully connected layer computes a linear transformation: multiply the input by a weight matrix and add a bias. This single operation, $\mathbf{h} = \mathbf{Wx} + \mathbf{b}$, replaces what would be a loop of scalar multiplications and additions. Understanding the shape dynamics is essential for debugging dimension mismatch errors.

### Formal Definition

For a single input vector $\mathbf{x} \in \mathbb{R}^{n}$ going into a layer with $m$ output units:

$$
\mathbf{h} = \mathbf{W}^\top \mathbf{x} + \mathbf{b} \quad \text{or} \quad \mathbf{h} = \mathbf{x} \mathbf{W} + \mathbf{b}
$$
depending on convention. In PyTorch, `nn.Linear(in_features=n, out_features=m)` stores $\mathbf{W} \in \mathbb{R}^{m \times n}$ and computes $\mathbf{y} = \mathbf{x} \mathbf{W}^\top + \mathbf{b}$.

We adopt the convention: weight matrix $\mathbf{W} \in \mathbb{R}^{n \times m}$ (input_dim x output_dim), input row vector $\mathbf{x} \in \mathbb{R}^{1 \times n}$:

$$
\mathbf{h} = \mathbf{x} \mathbf{W} + \mathbf{b}
$$
- $\mathbf{x} \in \mathbb{R}^{1 \times n}$ -- input row vector
- $\mathbf{W} \in \mathbb{R}^{n \times m}$ -- weight matrix
- $\mathbf{b} \in \mathbb{R}^{1 \times m}$ -- bias row vector
- $\mathbf{h} \in \mathbb{R}^{1 \times m}$ -- output (pre-activation)

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{W}$ | Weight matrix | Learns linear mapping from input to output space |
| $\mathbf{b}$ | Bias vector | Allows output offset independent of input |
| $n$ | Input dimension | Must match last dimension of $\mathbf{W}$'s rows |
| $m$ | Output dimension | Number of neurons in the layer |
| $\mathbf{h}$ | Pre-activation output | Passed through activation function (ReLU, etc.) |

### Worked Numerical Example

Input $\mathbf{x} = \begin{bmatrix} 0.5 & -0.2 & 0.1 \end{bmatrix}$, $\mathbf{W} = \begin{bmatrix} 0.4 & -0.3 \\ 0.2 & 0.5 \\ -0.1 & 0.6 \end{bmatrix}$, $\mathbf{b} = \begin{bmatrix} 0.1 & -0.2 \end{bmatrix}$

$$
\mathbf{x}\mathbf{W} = \begin{bmatrix} 0.5(-0.3) + (-0.2)(0.2) + 0.1(-0.1) \\ 0.5(-0.3) + (-0.2)(0.5) + 0.1(0.6) \end{bmatrix}^\top = \begin{bmatrix} 0.5(0.4) + (-0.2)(0.2) + 0.1(-0.1) & 0.5(-0.3) + (-0.2)(0.5) + 0.1(0.6) \end{bmatrix}
$$
Wait, let's do this carefully. $\mathbf{x} \in \mathbb{R}^{1 \times 3}$, $\mathbf{W} \in \mathbb{R}^{3 \times 2}$:

$$
\mathbf{x}\mathbf{W} = \begin{bmatrix} 0.5 & -0.2 & 0.1 \end{bmatrix} \begin{bmatrix} 0.4 & -0.3 \\ 0.2 & 0.5 \\ -0.1 & 0.6 \end{bmatrix} = \begin{bmatrix} 0.5 \times 0.4 + (-0.2) \times 0.2 + 0.1 \times (-0.1) & 0.5 \times (-0.3) + (-0.2) \times 0.5 + 0.1 \times 0.6 \end{bmatrix}
$$
$$
= \begin{bmatrix} 0.20 - 0.04 - 0.01 & -0.15 - 0.10 + 0.06 \end{bmatrix} = \begin{bmatrix} 0.15 & -0.19 \end{bmatrix}
$$
Then $\mathbf{h} = \mathbf{xW} + \mathbf{b} = \begin{bmatrix} 0.15 & -0.19 \end{bmatrix} + \begin{bmatrix} 0.1 & -0.2 \end{bmatrix} = \begin{bmatrix} 0.25 & -0.39 \end{bmatrix}$.

```python
x = torch.tensor([[0.5, -0.2, 0.1]])           # shape (1, 3)
W = torch.tensor([[0.4, -0.3],
                  [0.2, 0.5],
                  [-0.1, 0.6]])                 # shape (3, 2)
b = torch.tensor([[0.1, -0.2]])                 # shape (1, 2)
h = x @ W + b                                   # shape (1, 2)
print(f"h = {h}")                               # [[0.25, -0.39]]
```

**ML connection:** Every `nn.Linear` layer in PyTorch performs exactly this computation. A deep network is a composition of such layers interleaved with nonlinear activations.

---

## 3. Broadcasting Rules -- Why Broadcasting Matters for Batching

### Motivation and Intuition

When you add a bias vector of shape `(m,)` to a batch output of shape `(B, m)`, PyTorch automatically expands the bias along the batch dimension. This is **broadcasting**. Without it, you would need to manually tile the bias in memory, wasting compute and storage.

### Formal Definition

Two tensors are "broadcastable" if the following rules hold (compared from the **rightmost** dimension leftward):

1. If the dimensions differ, the tensor with the smaller shape is padded with 1s on the left.
2. For each dimension, the sizes must either be **equal** or one of them must be **1**.
3. If a dimension is 1, it is "stretched" to match the other dimension (no memory copy in the optimized implementation).

$$
(3, 1) + (1, 4) \rightarrow (3, 1) \text{ padded to } (3, 4) \;+\; (1, 4) \text{ padded to } (3, 4) \rightarrow (3, 4)
$$
| Term | Definition | Significance |
| :--- | :--- | :--- |
| Broadcasting | Automatic dimension expansion for element-wise ops | Eliminates manual tiling, saves memory |
| Rightmost alignment | Comparison starts at the last dimension | The trailing dimensions must be compatible |
| Stretch dimension | A size-1 axis is virtually repeated | No actual data duplication in optimized backends |
| Incompatible shapes | Neither equal nor 1 in some dimension | Raises `RuntimeError` -- shapes must match |

### Worked Numerical Example

```python
a = torch.tensor([[1], [2], [3]])     # shape (3, 1)
b = torch.tensor([10, 20, 30])        # shape (3,) -> broadcast to (1, 3)
c = a + b                             # (3, 1) + (3,) -> (3, 1) + (1, 3) -> (3, 3)
print(c)
```

$$
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}_{3\times1} + \begin{bmatrix} 10 & 20 & 30 \end{bmatrix}_{1\times3} = \begin{bmatrix} 11 & 21 & 31 \\ 12 & 22 & 32 \\ 13 & 23 & 33 \end{bmatrix}_{3\times3}
$$
**Why this matters for batching:** The bias $\mathbf{b}$ has shape `(1, m)` and the batch output $\mathbf{H} = \mathbf{XW}$ has shape `(B, m)`. Broadcasting adds the same bias to every row without a loop.

| Shape Pair | Compatible? | Result Shape | Reason |
| :--- | :--- | :--- | :--- |
| (3,1) + (1,4) | Yes | (3,4) | Both have a 1 to stretch |
| (3,2) + (2,) | Yes | (3,2) | Trailing dims match (2=2) |
| (3,2) + (3,) | No | Error | Rightmost: 2 vs 3, neither is 1 |
| (4,3,2) + (2,) | Yes | (4,3,2) | (2,) -> (1,1,2), then stretch |
| (4,3,2) + (3,2) | Yes | (4,3,2) | (3,2) -> (1,3,2), then stretch |

**ML connection:** Broadcasting is used everywhere: adding bias, applying layer normalization across a batch, computing attention logits across heads, and adding positional encodings to token embeddings.

---

## 4. Matrix Multiplication for Batch Forward Pass

### Motivation and Intuition

Real neural networks process **batches** of multiple samples simultaneously. Instead of looping over each sample, we stack all input vectors into a matrix $\mathbf{X}$ of shape $(B, D)$ and compute the forward pass for the whole batch with a single matrix multiplication.

### Formal Definition

Given a batch of $B$ input samples, each with $D$ features:

$$
\mathbf{X} \in \mathbb{R}^{B \times D}, \quad \mathbf{W} \in \mathbb{R}^{D \times H}, \quad \mathbf{b} \in \mathbb{R}^{1 \times H}
$$
$$
\mathbf{H} = \mathbf{X} \mathbf{W} + \mathbf{b} \in \mathbb{R}^{B \times H}
$$
Each row $i$ of $\mathbf{H}$ is the output for sample $i$: $\mathbf{H}_{i,:} = \mathbf{X}_{i,:} \mathbf{W} + \mathbf{b}$.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{X}$ | Input batch matrix | Each row is one training sample |
| $B$ | Batch size | Number of samples processed in parallel |
| $D$ | Input dimension | Number of features per sample |
| $H$ | Hidden/output dimension | Number of neurons in this layer |
| $\mathbf{XW}$ | Batched linear transformation | All $B$ outputs computed simultaneously |

### Worked Numerical Example

With $B=3$, $D=4$, $H=2$:

$$
\mathbf{X} = \begin{bmatrix} 1 & 0 & 2 & -1 \\ -1 & 2 & 0 & 1 \\ 0 & 1 & -1 & 2 \end{bmatrix}_{3\times4}, \quad \mathbf{W} = \begin{bmatrix} 0.2 & -0.1 \\ 0.3 & 0.0 \\ -0.2 & 0.4 \\ 0.1 & 0.5 \end{bmatrix}_{4\times2}
$$
$$
\mathbf{XW} = \begin{bmatrix} 1(0.2)+0(0.3)+2(-0.2)+(-1)(0.1) & 1(-0.1)+0(0.0)+2(0.4)+(-1)(0.5) \\ -1(0.2)+2(0.3)+0(-0.2)+1(0.1) & -1(-0.1)+2(0.0)+0(0.4)+1(0.5) \\ 0(0.2)+1(0.3)+(-1)(-0.2)+2(0.1) & 0(-0.1)+1(0.0)+(-1)(0.4)+2(0.5) \end{bmatrix}
$$
$$
= \begin{bmatrix} 0.2+0-0.4-0.1 & -0.1+0+0.8-0.5 \\ -0.2+0.6+0+0.1 & 0.1+0+0+0.5 \\ 0+0.3+0.2+0.2 & 0+0-0.4+1.0 \end{bmatrix} = \begin{bmatrix} -0.3 & 0.2 \\ 0.5 & 0.6 \\ 0.7 & 0.6 \end{bmatrix}
$$
```python
X = torch.tensor([[1., 0., 2., -1.],
                  [-1., 2., 0., 1.],
                  [0., 1., -1., 2.]])          # shape (3, 4)
W = torch.tensor([[0.2, -0.1],
                  [0.3, 0.0],
                  [-0.2, 0.4],
                  [0.1, 0.5]])                 # shape (4, 2)
b = torch.tensor([[0.05, -0.05]])              # shape (1, 2)
H = X @ W + b                                  # shape (3, 2)
print(f"H =\n{H}")
```

**ML connection:** The forward pass of a mini-batch is exactly this: one matrix multiply, one broadcast add. All deep learning frameworks optimize this operation to run on GPUs with maximum parallelism.

---

## 5. Transposition and Why We Transpose Keys in Attention

### Motivation and Intuition

Matrix multiplication requires the inner dimensions to match. In the attention mechanism, we need to compute dot products between every query (shape `[T, d_k]`) and every key (also shape `[T, d_k]`). The result must be a `[T, T]` matrix. To achieve this, we transpose the key matrix so that its last two dimensions are swapped: $\mathbf{K}^\top$ has shape `[d_k, T]`, enabling $\mathbf{Q} \mathbf{K}^\top$ of shape `[T, T]`.

### Formal Definition

The transpose of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ is $\mathbf{A}^\top \in \mathbb{R}^{n \times m}$ where:

$$
(\mathbf{A}^\top)_{ij} = \mathbf{A}_{ji}
$$
For batched tensors, we transpose only the last two dimensions using `.transpose(-2, -1)` or `.permute()`.

In attention, the score computation is:

$$
\text{Scores} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times N}
$$
where $\mathbf{Q} \in \mathbb{R}^{N \times d_k}$, $\mathbf{K} \in \mathbb{R}^{N \times d_k}$, so $\mathbf{K}^\top \in \mathbb{R}^{d_k \times N}$.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{A}^\top$ | Transpose matrix | Flips rows and columns |
| $\mathbf{K}^\top$ | Transposed key matrix | Aligns key dimension for dot product with Q |
| `.transpose(-2, -1)` | Swap last two dimensions on batched tensors | Preserves batch dimension |
| `.T` | Convenience for 2D transpose | Only works on 2D tensors |

### Worked Numerical Example

$$
\mathbf{Q} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \end{bmatrix}_{3\times2}, \quad \mathbf{K} = \begin{bmatrix} 0.7 & 0.8 \\ 0.9 & 1.0 \\ 1.1 & 1.2 \end{bmatrix}_{3\times2}
$$
$$
\mathbf{K}^\top = \begin{bmatrix} 0.7 & 0.9 & 1.1 \\ 0.8 & 1.0 & 1.2 \end{bmatrix}_{2\times3}
$$
$$
\mathbf{QK}^\top = \begin{bmatrix} 0.1(0.7)+0.2(0.8) & 0.1(0.9)+0.2(1.0) & 0.1(1.1)+0.2(1.2) \\ 0.3(0.7)+0.4(0.8) & 0.3(0.9)+0.4(1.0) & 0.3(1.1)+0.4(1.2) \\ 0.5(0.7)+0.6(0.8) & 0.5(0.9)+0.6(1.0) & 0.5(1.1)+0.6(1.2) \end{bmatrix} = \begin{bmatrix} 0.23 & 0.29 & 0.35 \\ 0.53 & 0.67 & 0.81 \\ 0.83 & 1.05 & 1.27 \end{bmatrix}_{3\times3}
$$
```python
Q = torch.tensor([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6]])               # shape (3, 2)
K = torch.tensor([[0.7, 0.8],
                  [0.9, 1.0],
                  [1.1, 1.2]])               # shape (3, 2)
scores = Q @ K.T                             # shape (3, 3)
print(f"Attention scores:\n{scores}")
```

**ML connection:** Transposition is the key operation that makes attention work. In multi-head attention with batched inputs `(B, N, d_model)`, you split heads to get `(B, H, N, d_k)` and then compute `scores = Q @ K.transpose(-2, -1)` which gives `(B, H, N, N)`.

---

## 6. Einstein Summation (Einsum) Notation

### Motivation and Intuition

As tensor operations grow complex, tracking axis indices becomes error-prone. Einstein summation (einsum) provides a concise notation that specifies which axes are multiplied and which are summed, leaving the framework to handle the details. It reduces bugs and makes code more readable by making the axis semantics explicit.

### Formal Definition

Einsum uses subscript labels to describe tensor contractions. The pattern `ij,jk->ik` means:

- Take axis `i` from tensor A and axis `i` from tensor B (they must match or be summed)
- Take axis `j` from tensor A and axis `j` from tensor B (they are summed over because `j` appears on the left but not the right)
- Axis `k` appears on both sides, so it is kept

General rule: labels that appear on the left but not the right are **summed over** (contracted). Labels that appear on both sides are **kept**.

| Pattern | Operation | Shapes | Meaning |
| :--- | :--- | :--- | :--- |
| `ij,jk->ik` | Matrix multiply | (m,k) x (k,n) -> (m,n) | Multiply and sum over k |
| `bij,bjk->bik` | Batched matmul | (B,m,k) x (B,k,n) -> (B,m,n) | Batch matrix multiply |
| `bthd,bThd->bhtT` | Batch attention scores | (B,T,H,d) x (B,T',H,d) -> (B,H,T,T') | Batched multi-head attention |
| `i,i->` | Dot product | (n,) x (n,) -> () | Sum elementwise product |
| `i,j->ij` | Outer product | (m,) x (n,) -> (m,n) | All pairs |
| `bqk,bkd->bqd` | Attention output | (B,Q,K) x (B,K,D) -> (B,Q,D) | Weighted sum over keys |
| `bhtd,bhTd->bhtT` | Multi-head attention | (B,H,T,d) x (B,H,T',d) -> (B,H,T,T') | Head-preserving scores |

### Worked Numerical Example

Standard matmul vs. einsum:

```python
A = torch.randn(4, 5)
B = torch.randn(5, 6)

C_matmul = A @ B               # shape (4, 6)
C_einsum = torch.einsum("ij,jk->ik", A, B)  # shape (4, 6)

# Batched: (2, 4, 5) @ (2, 5, 6) -> (2, 4, 6)
A_batch = torch.randn(2, 4, 5)
B_batch = torch.randn(2, 5, 6)
C_batch = torch.einsum("bij,bjk->bik", A_batch, B_batch)

# Attention scores with batch and heads: (B, H, T, d) x (B, H, T, d) -> (B, H, T, T)
Q = torch.randn(2, 8, 16, 64)
K = torch.randn(2, 8, 16, 64)
scores = torch.einsum("bhtd,bhTd->bhtT", Q, K)
print(f"Scores shape: {scores.shape}")  # (2, 8, 16, 16)
```

**ML connection:** Einsum is used extensively in modern transformer code (especially in libraries like xformers, flash-attention) to express attention computation, tensor contractions, and mixed precision kernels. Mastering einsum lets you read and write complex tensor operations with confidence.

---

## 7. Attention Math -- Q, K, V and Scaled Dot-Product Attention

### Motivation and Intuition

The attention mechanism allows each token to "look at" every other token in the sequence, weighting their contributions by relevance. This is the core innovation that made transformers outperform RNNs. The math involves three projections (Query, Key, Value), a scaled dot-product similarity, softmax normalization, and a weighted sum.

### Formal Definition

Given input $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}}$ for a sequence of $N$ tokens:

1. **Projections:**
   $$ \mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}^V $$
   where $\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

2. **Scaled Dot-Product Scores:**
   $$ \text{Scores} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{N \times N} $$

3. **Attention Weights (Softmax):**
   $$ \mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}, \text{dim}=-1 \right) \in \mathbb{R}^{N \times N} $$
   Each row sums to 1.

4. **Output (Weighted Sum):**
   $$ \text{Output} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{N \times d_v} $$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{Q}$ | Query matrix | What each token is looking for |
| $\mathbf{K}$ | Key matrix | What each token contains |
| $\mathbf{V}$ | Value matrix | What information each token passes on |
| $\sqrt{d_k}$ | Scaling factor | Prevents dot products from being too large (which saturates softmax) |
| $\mathbf{A}$ | Attention weight matrix | Row $i$ says how much token $i$ attends to each token $j$ |

### Worked Numerical Example

Let $N = 3$, $d_k = d_v = 4$.

```python
N, d_k = 3, 4
X = torch.randn(N, d_k)                    # (3, 4)

# Random projections for demonstration
W_Q = torch.randn(d_k, d_k)
W_K = torch.randn(d_k, d_k)
W_V = torch.randn(d_k, d_k)

Q = X @ W_Q                                # (3, 4)
K = X @ W_K                                # (3, 4)
V = X @ W_V                                # (3, 4)

scores = Q @ K.T / (d_k ** 0.5)            # (3, 3)
A = torch.softmax(scores, dim=-1)          # row-wise softmax
output = A @ V                             # (3, 4)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Scores shape: {scores.shape}")
print(f"Attention weights (row 0): {A[0]}")
print(f"Sum of row 0: {A[0].sum():.4f}")   # should be 1.0
```

**ML connection:** Scaled dot-product attention is the fundamental building block of every transformer model (GPT, BERT, T5, LLaMA, Claude, Gemini). The same math applies whether it's self-attention (Q, K, V all from X) or cross-attention (Q from decoder, K, V from encoder).

---

## 8. Multi-Head Attention Math

### Motivation and Intuition

A single attention head can only capture one type of relationship between tokens. By using multiple heads (typically 8-16), the model can simultaneously attend to different aspects: syntactic relations, semantic similarity, positional proximity, etc. Each head operates on a lower-dimensional subspace, making the computation efficient.

### Formal Definition

Given input $\mathbf{X} \in \mathbb{R}^{B \times N \times d_{\text{model}}}$, $H$ heads, each with dimension $d_k = d_{\text{model}} / H$:

1. **Project and split into heads:**

   For each head $h$, compute projections then reshape to separate heads.

   $$ \mathbf{Q}_h = \mathbf{X} \mathbf{W}^Q_h \in \mathbb{R}^{N \times d_k} $$

   In batched form: $\mathbf{Q} \in \mathbb{R}^{B \times H \times N \times d_k}$ after reshaping.

   Reshape: project to `(B, N, H*d_k)`, then view as `(B, N, H, d_k)`, then transpose to `(B, H, N, d_k)`.

2. **Per-head attention:**

   $$ \text{head}_h = \text{softmax}\left( \frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}} \right) \mathbf{V}_h \in \mathbb{R}^{N \times d_k} $$

3. **Concatenate and project:**

   $$ \text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) \mathbf{W}^O $$

   where Concat yields `(B, N, H*d_k)` and $\mathbf{W}^O \in \mathbb{R}^{H d_k \times d_{\text{model}}}$.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $H$ | Number of heads | Typically 8-16; each head learns different patterns |
| $d_k$ | Head dimension | $d_{\text{model}} / H$; smaller per-head dim keeps compute manageable |
| $\text{head}_h$ | Output of one attention head | Captures one type of token relationship |
| $\mathbf{W}^O$ | Output projection | Mixes information from all heads back to full model dimension |

### Shape Flow

```
Input:  (B, N, d_model) = (2, 16, 512)

After Q projection:        (2, 16, 512)
Reshape to heads:          (2, 16, 8, 64)
Transpose:                 (2, 8, 16, 64)   <-- Q, K, V all (B, H, N, d_k)

Attention scores:          (2, 8, 16, 16)   = Q @ K^T / sqrt(64)
Attention weights:         (2, 8, 16, 16)   = softmax(scores)
Context:                   (2, 8, 16, 64)   = weights @ V

Transpose back:            (2, 16, 8, 64)
Reshape:                   (2, 16, 512)     = concatenate heads

Output projection:         (2, 16, 512)     = concat @ W_O
```

```python
B, N, d_model, H = 2, 16, 512, 8
d_k = d_model // H

X = torch.randn(B, N, d_model)
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)
W_O = torch.randn(d_model, d_model)

# Project
Q = X @ W_Q   # (B, N, d_model)
K = X @ W_K
V = X @ W_V

# Split heads
Q = Q.view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
K = K.view(B, N, H, d_k).transpose(1, 2)
V = V.view(B, N, H, d_k).transpose(1, 2)

# Attention
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (B, H, N, N)
A = torch.softmax(scores, dim=-1)
context = A @ V                                    # (B, H, N, d_k)

# Concatenate heads
context = context.transpose(1, 2).contiguous().view(B, N, d_model)

# Output projection
output = context @ W_O                             # (B, N, d_model)
print(f"Output shape: {output.shape}")
```

**ML connection:** Multi-head attention is used in every modern transformer. GPT-3 uses 96 heads, BERT-base uses 12 heads. The parallel head computation is what makes it efficient on GPUs -- each head processes independently, and GPUs handle many small matrix multiplications efficiently through tensor cores when dimensions are aligned.

---

## 9. Gradient Flow -- Backpropagation Through Matrices

### Motivation and Intuition

Training a neural network requires computing the gradient of the loss with respect to every weight. For a linear layer $\mathbf{Y} = \mathbf{XW} + \mathbf{b}$, the gradient $\partial \text{Loss} / \partial \mathbf{W}$ must be computed from the upstream gradient $\partial \text{Loss} / \partial \mathbf{Y}$. Understanding the **outer product form** of this gradient is essential for debugging backpropagation and implementing custom layers.

### Formal Definition

For $\mathbf{Y} = \mathbf{XW}$ where $\mathbf{X} \in \mathbb{R}^{B \times D}$, $\mathbf{W} \in \mathbb{R}^{D \times H}$:

Forward: $\mathbf{Y} = \mathbf{XW}$

Backward (given upstream gradient $\mathbf{dL/dY} \in \mathbb{R}^{B \times H}$):

$$
\frac{\partial \text{Loss}}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial \text{Loss}}{\partial \mathbf{Y}} \in \mathbb{R}^{D \times H}
$$
This is the **outer product** of $\mathbf{X}$'s columns with $\mathbf{dL/dY}$'s rows.

For the bias: $\frac{\partial \text{Loss}}{\partial \mathbf{b}} = \text{sum}\left( \frac{\partial \text{Loss}}{\partial \mathbf{Y}}, \text{dim}=0 \right) \in \mathbb{R}^{H}$

And the gradient that flows back to the input:

$$
\frac{\partial \text{Loss}}{\partial \mathbf{X}} = \frac{\partial \text{Loss}}{\partial \mathbf{Y}} \mathbf{W}^\top \in \mathbb{R}^{B \times D}
$$
| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\partial \text{Loss} / \partial \mathbf{W}$ | Weight gradient | Used to update weights via SGD/Adam |
| $\mathbf{X}^\top \cdot \mathbf{dL/dY}$ | Outer product form | $\mathbf{X}$'s features times upstream errors |
| $\partial \text{Loss} / \partial \mathbf{b}$ | Bias gradient | Sum of upstream gradients over batch |
| $\partial \text{Loss} / \partial \mathbf{X}$ | Input gradient | Propagated to previous layer |
| Shape matching | $\mathbf{X}^\top \in \mathbb{R}^{D \times B}$, $\mathbf{dL/dY} \in \mathbb{R}^{B \times H}$ | Inner dimension $B$ matches |

### Worked Numerical Example

```python
B, D, H = 3, 4, 2
X = torch.randn(B, D, requires_grad=True)
W = torch.randn(D, H, requires_grad=True)

Y = X @ W
loss = Y.sum()           # scalar loss
loss.backward()          # compute gradients

# W.grad should equal X.T @ dL/dY where dL/dY = 1 (since loss = sum(Y))
print(f"W.grad shape: {W.grad.shape}")   # (4, 2)
print(f"Expected: {(X.T @ torch.ones(B, H)).shape}")  # (4, 2)

# Manual: dL/dW = X^T * dL/dY
dL_dY = torch.ones(B, H)   # upstream gradient
dL_dW_manual = X.T @ dL_dY
print(f"Manual gradient matches: {torch.allclose(W.grad, dL_dW_manual)}")
```

**Why shapes must match for backprop:**

The gradient tensor for every parameter must have the **exact same shape** as the parameter itself. This is the "shape compatibility" rule: every element of $\mathbf{W}$ gets its own gradient so the optimizer can update it.

$$
\mathbf{W} \in \mathbb{R}^{D \times H} \quad\Longrightarrow\quad \frac{\partial \text{Loss}}{\partial \mathbf{W}} \in \mathbb{R}^{D \times H}
$$

If shapes don't match, you get a runtime error or (worse) silently incorrect updates.

**ML connection:** Understanding gradient shapes is crucial for:
- Debugging "RuntimeError: The size of tensor a must match..."
- Implementing custom autograd functions
- Reasoning about gradient flow in deep networks
- Diagnosing vanishing/exploding gradients by checking gradient norms

---

## 10. Common Pitfalls in Tensor and Matrix Operations

### Motivation and Intuition

Even experienced practitioners encounter tensor shape bugs daily. These errors typically arise from dimension mismatch, in-place operations that break the autograd graph, or non-contiguous memory layouts that trigger unexpected copies. Awareness of these pitfalls prevents hours of debugging.

### Pitfall 1: Dimension Mismatch in Matrix Multiplication

```python
# Wrong: inner dimensions don't match
A = torch.randn(4, 5)
B = torch.randn(4, 6)         # should be (5, 6), not (4, 6)
# C = A @ B                    # RuntimeError: mat1 and mat2 shapes cannot be multiplied

# Correct: check shapes
assert A.shape[-1] == B.shape[0], f"Inner dims: {A.shape[-1]} vs {B.shape[0]}"
C = A @ B.T if A.shape[-1] != B.shape[0] else A @ B
```

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `mat1 and mat2 shapes cannot be multiplied` | Inner dimensions don't match | Check `A.shape[-1] == B.shape[-2]` |
| `Expected 2D input, got 3D` | Forgot to flatten or handle batch dim | Use `torch.matmul` or add batch dim |
| `The expanded size...must match...non-singleton dimension` | Broadcasting mismatch | Check shapes with print statements |

### Pitfall 2: In-Place Operations and Autograd

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
# y[0] = 100                       # RuntimeError: a leaf Variable that requires grad...
# Use .detach() if you need to modify
y = x ** 2
y = y.clone()                      # clone() detaches from graph
z = y.sum()
# z.backward()                      # would work only if y kept grad_fn
```

In-place operations like `x += 1`, `x.copy_(y)`, `x.fill_(0)` on tensors with `requires_grad=True` will raise errors because they would invalidate the autograd graph.

| Operation | Safe? | Alternative |
| :--- | :--- | :--- |
| `x.add_(1)` | No (requires_grad) | `x = x + 1` |
| `x.view(-1)` | Yes | Creates new view |
| `x.resize_(...)` | No | `x = x.reshape(...)` |
| `x.fill_(0)` | No (requires_grad) | Create new tensor |
| `model.zero_grad()` | Yes | Specifically designed for gradients |

### Pitfall 3: Non-Contiguous Tensors

```python
x = torch.randn(3, 4)
x_t = x.T                           # transposed view -- non-contiguous
# x_t.view(-1)                      # RuntimeError: view size is not compatible...
x_contig = x_t.contiguous()         # makes a contiguous copy
x_flat = x_t.reshape(-1)            # reshape works on non-contiguous (may copy)
```

Non-contiguous tensors arise from `.T`, `.transpose()`, `.permute()`, and `.narrow()`. Operations like `.view()` require contiguous memory. Use `.reshape()` (which handles non-contiguous) or `.contiguous()` first.

| Tensor | Contiguous? | `.view()` works? |
| :--- | :--- | :--- |
| Original `randn(3,4)` | Yes | Yes |
| After `.T` | No | No |
| After `.transpose(0,1)` | No | No |
| After `.contiguous()` | Yes | Yes |
| After `.reshape(...)` | May copy but works | Yes |

### Pitfall 4: Device Mismatch

```python
# cpu_tensor + gpu_tensor raises RuntimeError
cpu_t = torch.tensor([1., 2.])
# gpu_t = cpu_t.cuda()              # would work only with GPU
# result = cpu_t + gpu_t            # RuntimeError: expected device cuda but got cpu

# Always move data and model to the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X = X.to(device)
y = y.to(device)
```

### Pitfall 5: Forgetting to Zero Gradients

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10):
    pred = model(X)
    loss = loss_fn(pred, y)
    # optimizer.zero_grad()          # MISSING! Gradients accumulate
    loss.backward()
    optimizer.step()
```

Without zero_grad, gradients accumulate across epochs, leading to ever-growing gradient values and training instability.

### Pitfall 6: Softmax Along Wrong Dimension

```python
logits = torch.randn(4, 10)          # (batch, vocab_size)
probs_wrong = torch.softmax(logits, dim=0)   # WRONG: softmax over batch
probs_correct = torch.softmax(logits, dim=1)  # correct: softmax over classes
```

For attention, softmax must be along the last dimension (over keys) so each query sums to 1.

| Context | Correct dim | Explanation |
| :--- | :--- | :--- |
| Classification logits | `dim=1` | Softmax over classes |
| Attention scores (B, H, T, T) | `dim=-1` | Softmax over keys for each query |
| Cross-entropy target | `dim=1` | Softmax internally |

---

> **Check your intuition:** If $\mathbf{X} \in \mathbb{R}^{B \times D}$ and $\mathbf{W} \in \mathbb{R}^{D \times H}$, what is the shape of $\frac{\partial \text{Loss}}{\partial \mathbf{X}}$ given $\frac{\partial \text{Loss}}{\partial \mathbf{Y}} \in \mathbb{R}^{B \times H}$? What would happen if you incorrectly used $\mathbf{W}$ instead of $\mathbf{W}^\top$?

---

## Prerequisites and Further Reading

- **StatQuest:** L01 Neural Networks Part 0, L26 Tensors for Neural Networks, L27 Essential Matrix Algebra for Neural Networks, L28 Matrix Math Behind Transformer Neural Networks
- **PyTorch docs:** `torch.Tensor`, `torch.matmul`, `torch.einsum`, broadcasting semantics, `torch.autograd`
- **Papers:** Vaswani et al., "Attention Is All You Need" (2017)
- **Concepts:** Linear algebra fundamentals, broadcasting rules, backpropagation chain rule, contiguous vs non-contiguous memory
