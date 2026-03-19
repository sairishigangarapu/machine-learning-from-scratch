## LDA: Linear Discriminant Analysis

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Why LDA? The Failure of PCA for Classification

### Motivation
PCA finds the direction of **maximum variance** and projects data onto it. But maximum variance is not always useful for classification.

**The Problem:** Consider a 2D dataset with two linearly separable classes. PCA projects this data onto its axis of maximum variance. After projection to 1D, the two classes may become *completely mixed* — the data is no longer linearly separable.

> **Key Insight:** The direction of maximum variance may be *orthogonal* to the direction that best separates the classes.

### The LDA Goal
Find a projection direction $\mathbf{v}$ such that when the data is projected onto the line spanned by $\mathbf{v}$, samples from **different classes are maximally well-separated**.

---

## 2. Formalizing the Objective

### Setup
- $N$ data points in a $d$-dimensional space: $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N$
- $N_1$ samples from class $\mathcal{C}_1$, $N_2$ from class $\mathcal{C}_2$ (so $N_1 + N_2 = N$)
- The projection of a point $\mathbf{x}_i$ onto a unit vector $\mathbf{v}$ is: $y_i = \mathbf{v}^T \mathbf{x}_i$

### Projected Means
Let $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2$ be the class means **before** projection. The projected class means are:

$$
\tilde{\mu}_1 = \mathbf{v}^T \boldsymbol{\mu}_1, \qquad \tilde{\mu}_2 = \mathbf{v}^T \boldsymbol{\mu}_2
$$

### Why Maximizing Projected Mean Separation Alone Fails

Maximizing $|\tilde{\mu}_1 - \tilde{\mu}_2|$ is insufficient. If one projection direction has a large mean separation but also large *within-class spread (variance)*, the classes will still overlap. We must **normalize** the separation by the within-class scatter.

---

## 3. The Fisher Criterion (Objective Function $J(\mathbf{v})$)

### Within-Class Scatter (after projection)
Define the scatter of projected samples for each class (variance without the $\frac{1}{N}$ factor):

$$
\tilde{s}_1^2 = \sum_{y_i \in \mathcal{C}_1} (y_i - \tilde{\mu}_1)^2, \qquad \tilde{s}_2^2 = \sum_{y_i \in \mathcal{C}_2} (y_i - \tilde{\mu}_2)^2
$$

### The Fisher Criterion
Maximize the ratio of **between-class separation** to **within-class scatter**:

$$
\boxed{J(\mathbf{v}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2}}
$$

This simultaneously enforces:
1. The projected class means should be **far apart** (large numerator).
2. The within-class spread of each projected cluster should be **small** (small denominator).

---

## 4. Deriving $J(\mathbf{v})$ in Terms of $\mathbf{v}$

### Step 1: Define the Within-Class Scatter Matrices ($S_W$)
The per-class scatter matrices in the **original** $d$-dimensional space are:

$$
S_1 = \sum_{\mathbf{x}_i \in \mathcal{C}_1} (\mathbf{x}_i - \boldsymbol{\mu}_1)(\mathbf{x}_i - \boldsymbol{\mu}_1)^T
$$

$$
S_2 = \sum_{\mathbf{x}_i \in \mathcal{C}_2} (\mathbf{x}_i - \boldsymbol{\mu}_2)(\mathbf{x}_i - \boldsymbol{\mu}_2)^T
$$

The **Within-Class Scatter Matrix** is:

$$
S_W = S_1 + S_2
$$

It can be shown that the projected scatter terms are:
$\tilde{s}_1^2 = \mathbf{v}^T S_1 \mathbf{v}$ and $\tilde{s}_2^2 = \mathbf{v}^T S_2 \mathbf{v}$

Therefore, the denominator of $J(\mathbf{v})$ becomes:

$$
\tilde{s}_1^2 + \tilde{s}_2^2 = \mathbf{v}^T (S_1 + S_2) \mathbf{v} = \mathbf{v}^T S_W \mathbf{v}
$$

### Step 2: Define the Between-Class Scatter Matrix ($S_B$)
The **Between-Class Scatter Matrix** captures the separation between the class means:

$$
S_B = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T
$$

The numerator of $J(\mathbf{v})$ becomes:

$$
(\tilde{\mu}_1 - \tilde{\mu}_2)^2 = \mathbf{v}^T S_B \mathbf{v}
$$

### Step 3: The Final Matrix Form
$$
\boxed{J(\mathbf{v}) = \frac{\mathbf{v}^T S_B \mathbf{v}}{\mathbf{v}^T S_W \mathbf{v}}}
$$

This is a **Generalized Rayleigh Quotient**. Maximizing it is a classic result in linear algebra.

---

## 5. Solving the Optimization (The Eigenvalue Problem)

### Derivation
To maximize $J(\mathbf{v})$, set $\frac{d}{d\mathbf{v}} J(\mathbf{v}) = 0$. This yields:

$$
S_B \mathbf{v} - J(\mathbf{v}) \cdot S_W \mathbf{v} = \mathbf{0}
$$

Let $\lambda = J(\mathbf{v})$ (the scalar we are maximizing):

$$
S_B \mathbf{v} = \lambda S_W \mathbf{v}
$$

If $S_W$ is **invertible** (full rank), we can multiply both sides by $S_W^{-1}$:

$$
\underbrace{S_W^{-1} S_B}_{M} \mathbf{v} = \lambda \mathbf{v}
$$

**This is a standard eigenvalue equation!** The optimal projection direction $\mathbf{v}$ is the **eigenvector** of $M = S_W^{-1} S_B$ corresponding to the **largest eigenvalue** $\lambda$.

> **Why the largest eigenvalue?** Because $\lambda = J(\mathbf{v})$, and we want to *maximize* $J(\mathbf{v})$.

### The Shortcut (Two-Class Case Only)

For any vector $\mathbf{x}$:

$$
S_B \mathbf{x} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) \underbrace{(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \mathbf{x}}_{\text{scalar}} \propto (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

So $S_B \mathbf{v}$ always points in the direction of $(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$. This means we do not need to compute $S_B$ explicitly. The optimal direction is simply:

$$
\boxed{\mathbf{v} = S_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)}
$$

*(If $S_W$ is not full rank, use a pseudo-inverse.)*

---

## 6. Summary: LDA Algorithm (Two-Class)

| Step | Operation |
|------|-----------|
| 1 | Compute class means $\boldsymbol{\mu}_1$, $\boldsymbol{\mu}_2$ |
| 2 | Compute scatter matrices $S_1$, $S_2$, then $S_W = S_1 + S_2$ |
| 3 | Compute $\mathbf{v} = S_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ |
| 4 | Project all samples: $y_i = \mathbf{v}^T \mathbf{x}_i$ |

---

## 7. Worked Example

**Data:**
- $\mathcal{C}_1$ (5 samples): $\{(1,2),(2,3),(3,3),(3,4),(5,4)\}$
- $\mathcal{C}_2$ (6 samples): $\{(1,6),(1,5),(2,2),(3,2),(3,1),(5,2)\}$

---

**Step 1: Compute Class Means**

$$
\boldsymbol{\mu}_1 =
\begin{bmatrix}
3.0 \\
3.6
\end{bmatrix},
\qquad
\boldsymbol{\mu}_2 =
\begin{bmatrix}
3.3 \\
2.0
\end{bmatrix}
$$

---

**Step 2: Compute Scatter Matrices**

$$
S_1 = 4 \cdot \text{Cov}(\mathcal{C}_1) \approx
\begin{bmatrix}
10 & 8 \\
8  & 7.2
\end{bmatrix}
$$

$$
S_2 = 5 \cdot \text{Cov}(\mathcal{C}_2) \approx
\begin{bmatrix}
17.3 & 16 \\
16   & 16
\end{bmatrix}
$$

$$
S_W = S_1 + S_2 =
\begin{bmatrix}
27.3 & 24 \\
24   & 23.2
\end{bmatrix}
$$

---

**Step 3: Invert $S_W$ and Find $\mathbf{v}$**

$$
S_W^{-1} \approx
\begin{bmatrix}
 0.39 & -0.41 \\
-0.41 &  0.47
\end{bmatrix}
$$

$$
\mathbf{v} = S_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
= S_W^{-1}
\begin{bmatrix}
-0.3 \\
 1.6
\end{bmatrix}
\approx
\begin{bmatrix}
-0.79 \\
 0.89
\end{bmatrix}
$$

**Result:** Projecting all points onto the direction $[-0.79,\ 0.89]^T$ yields two well-separated 1D clusters — far superior to the PCA projection on the same data.

---

## 8. Generalizing to Multiple Classes ($C > 2$)

For $C$ classes, LDA can reduce dimensionality to **at most $C - 1$ dimensions** (since $S_B$ has rank at most $C-1$).

- The projection is now a full **matrix** $W \in \mathbb{R}^{d \times (C-1)}$, found by the top $C-1$ eigenvectors of $S_W^{-1} S_B$.
- Each sample is projected as: $\mathbf{y}_i = W^T \mathbf{x}_i$

| Scenario | LDA Output Dimensions |
|---|---|
| 2 classes | Up to 1D |
| 3 classes | Up to 2D |
| $C$ classes | Up to $(C-1)$D |

---

## 9. LDA vs. PCA — A Direct Comparison

| Property | PCA | LDA |
|---|---|---|
| **Goal** | Maximize variance | Maximize class separation |
| **Supervised?** | No (unsupervised) | Yes (uses class labels) |
| **Projection Direction** | Eigenvectors of $\Sigma_x$ | Eigenvectors of $S_W^{-1} S_B$ |
| **Max Output Dims** | $\min(n, d)$ | $C - 1$ |
| **Preserves Linear Separability?** | Not guaranteed | Yes, by design |

---

## 10. Next Steps

We have now seen two fundamental dimensionality reduction techniques: PCA (unsupervised, variance-preserving) and LDA (supervised, class-separation-preserving). The natural next step is to see how these ideas translate into **Python implementation**, where we will use NumPy to compute $S_W$, $S_B$, the key eigenvectors, and visualize the projected data.
