## Low-Rank Approximation

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Matrix Rank and Efficiency

### Motivation: The Billion Parameter Problem
In modern artificial intelligence, Large Language Models (LLMs) like Llama or GPT possess $70+$ billion parameters housed inside gigantic dense weight matrices $W$. Storing and updating these massive full-rank matrices requires overwhelming amounts of VRAM.

However, researchers discovered most of the "knowledge" inside these matrices is redundantly distributed. The matrix $W$ might mathematically be full-rank, but practically, the overwhelming bulk of its intelligence lies in a much lower-dimensional **Subspace**.

By forcing an incredibly large matrix $W$ into a **Low-Rank Approximation**, we can save absolutely astronomical amounts of computing power and memory with minimal degradation to intelligence. This exact mathematical phenomenon is the entire baseline origin of **LoRA (Low-Rank Adaptation)** for fine-tuning LLMs!

### The Factorization Perspective
If a massive matrix $A$ ($m \times n$) can be well-approximated by an artificial bottleneck rank $r$, we can slice it into two radically smaller matrices:

$$
\boxed{A \approx B \times C^T}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A$ | The original $m \times n$ matrix | The full-rank matrix we wish to compress |
| $B$ | $m \times r$ basis matrix (left factor) | Stores the low-dimensional column space representation |
| $C^T$ | $r \times n$ coordinate matrix (right factor) | Stores the low-dimensional row space representation |
| $r$ | The bottleneck rank ($r \ll \min(m,n)$) | Controls the compression ratio and approximation quality |

* **$B$ ($m \times r$):** A tight, low-dimensional basis matrix.
* **$C^T$ ($r \times n$):** The linear coordinates.

> **Storage Reality Check:** Consider a $10,000 \times 10,000$ weight matrix (100,000,000 parameter variables).
> If we aggressively factorize it to a low bottleneck rank $r = 100$:
> $B$ takes $10,000 \times 100 = 1,000,000$ variables.
> $C^T$ takes $100 \times 10,000 = 1,000,000$ variables.
> **Result:** We compressed 100M parameters down to 2M. This 98% space savings allows massive LLMs to be trained on consumer graphics cards.

---

## 2. Defining the Low-Rank Approximation Problem

We seek a matrix $A_k$ intentionally artificially capped at rank $k$ that minimizes the Euclidean distance error to the original matrix $A$:

$$
\min_{A_k} \|A - A_k\|_F \quad \text{subject to} \quad \text{rank}(A_k) \le k
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A_k$ | The rank-$k$ approximating matrix | The compressed representation we are optimizing |
| $\|A - A_k\|_F$ | Frobenius norm of the approximation error | Measures how closely $A_k$ reconstructs the original $A$ |
| $\text{rank}(A_k)$ | The rank of the approximating matrix | Constrains $A_k$ to a lower-dimensional subspace |
| $k$ | The target rank (compression parameter) | Trades off between compression and reconstruction fidelity |

**The Optimization Hurdle:** This constraint problem is strictly geometrically **non-convex**. Traditional optimization strategies (like standard gradient descent) tend to fail massively on non-convex hard-constraints. Fortunately, Linear Algebra provides a pristine shortcut.

---

## 3. The Eckart-Young-Mirsky Theorem (The SVD Savior)

Even though the landscape is non-convex, finding the global optimum does not require a brute-force optimizer. Singular Value Decomposition analytically gifts us the exact optimal answer on a silver platter. 

If $A = U \Sigma V^T$, the mathematically infallible "best" rank-$k$ approximation simply deletes the smallest singular values in $\Sigma$, keeping only the top $k$ dominant structural axes:

$$
\boxed{A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A_k$ | The optimal rank-$k$ approximation to $A$ | Minimizes reconstruction error among all rank-$k$ matrices |
| $\sigma_i$ | The $i$-th singular value (sorted descending) | Weights the contribution of each rank-1 component |
| $\mathbf{u}_i$ | The $i$-th left singular vector | Defines the output direction of the $i$-th component |
| $\mathbf{v}_i$ | The $i$-th right singular vector | Defines the input direction of the $i$-th component |
| $k$ | Number of singular values retained | Determines the compression level and approximation quality |

In matrix form, we blindly truncate columns from $U$ and $V$:

$$
A_k = U_k \Sigma_k V_k^T
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A_k$ | The rank-$k$ approximation of $A$ | Computed by truncating the full SVD |
| $U_k$ | First $k$ columns of $U$ | Retains only the most important left singular vectors |
| $\Sigma_k$ | $k \times k$ diagonal matrix of top $k$ singular values | Contains the dominant scaling factors |
| $V_k^T$ | First $k$ rows of $V^T$ | Retains only the most important right singular vectors |

---

## 4. Comprehensive Worked Example

**Original Matrix ($3 \times 3$):**

$$
A = \begin{bmatrix} 3 & 2 & 1 \\ 0 & 2 & 1 \\ 0 & 0 & 1 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A$ | Original full-rank matrix ($3 \times 3$) | Upper triangular matrix to be approximated by a rank-$k$ matrix |
| $3, 2, 1, \dots$ | Entries of $A$ | Coefficients of the linear transformation |

```python
import numpy as np

A = np.array([[3, 2, 1], 
              [0, 2, 1], 
              [0, 0, 1]])

# 1. Full SVD Extraction
# Singular values approx: [4.12, 1.73, 0.70]
U, S, Vt = np.linalg.svd(A)

# 2. Rank-2 Approximation (k=2)
# We brutally truncate the smallest singular value (0.70) to 0.
k = 2
Sk = np.diag(S[:k])
Ak = U[:, :k] @ Sk @ Vt[:k, :]

# A_k is mathematically guaranteed to be the closest rank-2 matrix to A 
print(Ak)
```

**Output Matrix ($A_2$):**

$$
A_2 \approx \begin{bmatrix} 2.99 & 2.01 & 0.98 \\ 0.01 & 1.99 & 1.02 \\ 0.00 & 0.01 & 0.99 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A_2$ | Optimal rank-2 approximation of $A$ | Closest rank-2 matrix to the original $A$ under Frobenius norm |
| $2.99, 2.01, 0.98, \dots$ | Entries of $A_2$ | Nearly identical to original $A$ despite discarding the third singular value |

Notice how stunningly close $A_2$ perfectly mimics original $A$, despite losing an entire dimension of variance. The error $\|A - A_2\|$ is strictly minimized.

---

## 5. Visualizing the Application: Image Compression

Digital images are massive matrices of raw pixel intensities.

* **Input Image:** A high-res facial photo might technically have rank $1000$.
* **SVD Rank 1-10 Approximation:** We only see vague "ghostly" sweeping blob colors representing broad lighting and skin tone layouts.
* **SVD Rank 50 Approximation:** The shape of the eyes, nose, and mouth snap into terrifying clarity. The reconstruction is nearly indistinguishable from reality to the human eye, entirely mathematically assembled using just 5% of the original structural information.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 13: SVD Properties](Lecture%2013%20SVD%20%3A%20Properties%20and%20Applications.md) — Matrix norms and pseudo-inverse concepts underpin the error guarantees of low-rank approximation
- **Next:** [Lecture 16: PCA](Lecture%2016%20Principal%20Component%20Analysis.md) — Applies low-rank structure to find directions of maximum variance for dimensionality reduction
- **Related:** [Lecture 12: SVD](Lecture%2012%20Singular%20Value%20Decomposition.md) — The Eckart-Young-Mirsky theorem uses SVD to find optimal rank-$k$ approximations
