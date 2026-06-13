## LDA: Linear Discriminant Analysis

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Motivation: Why LDA? (PCA's Blind Spot)

In previous lectures, we mastered **Principal Component Analysis (PCA)**. PCA is fantastic for finding the "broadest" part of the data—its maximum variance. But PCA has a major flaw for classification: **It is unsupervised.** It doesn't know about class labels.

### The "Failure of Variance" Analogy
Imagine two distinct clusters of data (e.g., "Malignant" vs "Benign" tumors). 
* **PCA's Goal:** Find the direction of maximum spread. If that spread is horizontal, but the classes are separated vertically, PCA will squash both clusters into the same horizontal line. The result? A single mixed "cloud" where you can no longer tell the classes apart.
* **LDA's Goal:** Ignore the overall variance. Instead, find a projection that **pushes the classes apart** while keeping each cluster as tight as possible.

> **Hacker's Catch:** PCA is a **Variance Specialist**. LDA is a **Separation Specialist**. Use PCA for compression; use LDA for classification preparation.

---

## 2. Formalizing the Objective

### 2.1 Setup
- $N$ data points in a $d$-dimensional space: $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N$
- $N_1$ samples from class $\mathcal{C}_1$, $N_2$ from class $\mathcal{C}_2$
- The projection of a point $\mathbf{x}_i$ onto a unit vector $\mathbf{v}$ is: $y_i = \mathbf{v}^T \mathbf{x}_i$

### 2.2 Projected Means
The projected class means are:

$$
\tilde{\mu}_1 = \mathbf{v}^T \boldsymbol{\mu}_1, \qquad \tilde{\mu}_2 = \mathbf{v}^T \boldsymbol{\mu}_2
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\tilde{\mu}_1, \tilde{\mu}_2$ | Projected class means (scalars) | Measure the center of each class after projection |
| $\mathbf{v}$ | The projection direction (unit vector) | The direction we are optimizing to maximize class separation |
| $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2$ | Original class mean vectors in $d$-dimensional space | Centroids of each class before projection |

### Why Maximizing Projected Mean Separation Alone Fails
Maximizing $|\tilde{\mu}_1 - \tilde{\mu}_2|$ is insufficient. If one projection direction has a large mean separation but also **large within-class spread (variance)**, the classes will still overlap. We must **normalize** the separation by the "messiness" of the clusters.

---

## 3. The Fisher Criterion (Objective Function $J(\mathbf{v})$)

Maximize the ratio of **between-class separation** to **within-class scatter**:

$$
\boxed{J(\mathbf{v}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2}}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $J(\mathbf{v})$ | The Fisher criterion (objective function) | Ratio to maximize for optimal class separation |
| $\tilde{\mu}_1 - \tilde{\mu}_2$ | Difference of projected class means | Numerator: between-class separation |
| $(\tilde{\mu}_1 - \tilde{\mu}_2)^2$ | Squared mean difference | Ensures positive value; penalizes small separation |
| $\tilde{s}_1^2, \tilde{s}_2^2$ | Projected within-class variances | Denominator: within-class scatter (compactness) |

This simultaneously enforces:
1. **Far apart means:** The projected class means should be distant (large numerator).
2. **Compact Clusters:** The within-class spread of each projected cluster should be small (small denominator).

---

## 4. Deriving $J(\mathbf{v})$ in Terms of $\mathbf{v}$

To solve for $\mathbf{v}$, we define two matrices in the original $d$-dimensional space:

### Step 1: Within-Class Scatter Matrix ($S_W$)
Captures the "spread" of each class around its own mean (the "Un-tightness" of the clusters).

$$
S_W = \sum_{x \in \mathcal{C}_1} (x - \mu_1)(x - \mu_1)^T + \sum_{x \in \mathcal{C}_2} (x - \mu_2)(x - \mu_2)^T
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $S_W$ | Within-class scatter matrix | Measures the spread of data within each class |
| $\mathcal{C}_1, \mathcal{C}_2$ | The two class sets | Partition of data by class labels |
| $x - \mu_k$ | Deviation of sample $x$ from its class mean $\mu_k$ | Measures how far a point is from its class center |
| $(x - \mu_k)(x - \mu_k)^T$ | Outer product forming a scatter contribution | Accumulates directional spread information |

The denominator of $J(\mathbf{v})$ becomes: $\tilde{s}_1^2 + \tilde{s}_2^2 = \mathbf{v}^T S_W \mathbf{v}$.

### Step 2: Between-Class Scatter Matrix ($S_B$)
Captures the "distance" between the class centroids (the "Separation metric").

$$
S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $S_B$ | Between-class scatter matrix | Measures the separation between class centroids |
| $\mu_1 - \mu_2$ | Vector connecting the two class means | Direction of maximum inter-class separation |
| $(\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$ | Outer product of the mean difference | Forms a rank-1 matrix pointing along the separation direction |

The numerator of $J(\mathbf{v})$ becomes: $(\tilde{\mu}_1 - \tilde{\mu}_2)^2 = \mathbf{v}^T S_B \mathbf{v}$.

### Step 3: Matrix Form (Generalized Rayleigh Quotient)

$$
J(\mathbf{v}) = \frac{\mathbf{v}^T S_B \mathbf{v}}{\mathbf{v}^T S_W \mathbf{v}}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $J(\mathbf{v})$ | The Fisher criterion as a Rayleigh quotient | Ratio of between-class to within-class variance |
| $\mathbf{v}^T S_B \mathbf{v}$ | Projected between-class scatter | Scalar measuring class separation along $\mathbf{v}$ |
| $\mathbf{v}^T S_W \mathbf{v}$ | Projected within-class scatter | Scalar measuring intra-class spread along $\mathbf{v}$ |
| $\mathbf{v}$ | The optimal projection direction | Maximized when $\mathbf{v}$ is the top eigenvector of $S_W^{-1} S_B$ |

---

## 5. Solving the Engine: Eigenvalues to the Rescue

### Derivation
To maximize $J(\mathbf{v})$, set $\frac{d}{d\mathbf{v}} J(\mathbf{v}) = 0$. This leads to a standard eigenvalue equation:

$$
\boxed{S_W^{-1} S_B \mathbf{v} = \lambda \mathbf{v}}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $S_W^{-1}$ | Inverse of the within-class scatter matrix | "Un-stretches" data to normalize class spreads |
| $S_B$ | Between-class scatter matrix | Encodes the separation between class means |
| $\mathbf{v}$ | The optimal projection direction (eigenvector) | Maximizes the Fisher criterion $J(\mathbf{v})$ |
| $\lambda$ | The eigenvalue corresponding to $\mathbf{v}$ | Value of $J(\mathbf{v})$ at the optimal direction |

The optimal projection direction $\mathbf{v}$ is the **eigenvector** of $M = S_W^{-1} S_B$ corresponding to the **largest eigenvalue**.

### The Shortcut (Two-Class Case Only)
For a simple two-class problem, the optimal direction $\mathbf{v}$ is simply:

$$
\boxed{\mathbf{v} = S_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\mathbf{v}$ | The optimal 2-class LDA projection direction | Directly computed without eigendecomposition |
| $S_W^{-1}$ | Inverse of the within-class scatter matrix | Corrects for the shape and spread of the clusters |
| $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$ | Vector between the two class means | Points from one cluster center to the other |

*   **Intuition:** Take the vector pointing from one cluster center to the other $(\mu_1 - \mu_2)$, then "correct" it using $S_W^{-1}$ to account for the shape and spread of the clusters. This is effectively "un-stretching" the data to make the separation most obvious.

---

## 6. Worked Example (The Numerical Test)

**Data:**
- $\mathcal{C}_1$: $\{(1,2),(2,3),(3,3),(3,4),(5,4)\} \to \boldsymbol{\mu}_1 = [3.0, 3.6]^T$
- $\mathcal{C}_2$: $\{(1,6),(1,5),(2,2),(3,2),(3,1),(5,2)\} \to \boldsymbol{\mu}_2 = [3.3, 2.0]^T$

**Step 1: Compute Scatter Matrices**

$$
S_W = S_1 + S_2 =
\begin{bmatrix}
27.3 & 24 \\
24   & 23.2
\end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $S_W$ | Within-class scatter matrix ($2 \times 2$) | Sum of scatter matrices for both classes; measures intra-class spread |
| $S_1$ | Scatter matrix for class $\mathcal{C}_1$ | Accumulated deviations of class 1 samples from $\boldsymbol{\mu}_1$ |
| $S_2$ | Scatter matrix for class $\mathcal{C}_2$ | Accumulated deviations of class 2 samples from $\boldsymbol{\mu}_2$ |
| $27.3, 23.2$ | Diagonal entries of $S_W$ | Variance of features weighted by within-class spread |
| $24$ | Off-diagonal entry of $S_W$ | Covariance of features within classes |

**Step 2: Solve for $\mathbf{v}$**

$$
\mathbf{v} = S_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) \approx
\begin{bmatrix}
-0.39 & -0.41 \\
-0.41 &  0.47
\end{bmatrix}
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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{v}$ | Optimal LDA projection direction | Unit vector maximizing class separation |
| $S_W^{-1}$ | Inverse of within-class scatter matrix | Corrects for intra-class shape and spread |
| $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$ | Difference of class mean vectors | Vector pointing from one class center to the other |
| $\begin{bmatrix} -0.79 \\ 0.89 \end{bmatrix}$ | Computed projection direction | Components defining the line that best separates the two classes |

**Result:** Projecting onto $[-0.79,\ 0.89]^T$ yields two perfectly separated 1D clusters.

---

## 7. Generalizing to Multiple Classes ($C > 2$)

For $C$ classes, LDA can reduce dimensionality to **at most $C - 1$ dimensions**.
*   **3 classes?** Max 2D projection.
*   **20 classes?** Max 19D projection.
*   The projection is a matrix $W \in \mathbb{R}^{d \times (C-1)}$ formed by the top eigenvectors of $S_W^{-1} S_B$.

---

## 8. Summary API Reference

| Property | PCA | LDA |
|---|---|---|
| **Goal** | Preservation of Signal | Separation of Classes |
| **Labels** | Blind (Unsupervised) | Aware (Supervised) |
| **Projection** | Eigenvectors of $\Sigma_x$ | Eigenvectors of $S_W^{-1} S_B$ |
| **Max Dims** | $\min(n, d)$ | $C - 1$ |

**Next Step:** move to **Python Implementation**, where we'll handle "singular" matrices using the **Moore-Penrose Pseudoinverse**.

---

### Practical Application
- **Unsupervised Learning Lab:** See how PCA (LDA's cousin) is implemented in [pca_lab.py](../03-Unsupervised-Learning/PRINCIPAL%20COMPONENT%20ANALYSIS/pca_lab.py).
- **Theory Comparison:** Read the [PCA Theory](../03-Unsupervised-Learning/PRINCIPAL%20COMPONENT%20ANALYSIS/Theory.md) to understand the Unsupervised vs. Supervised extraction gap.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 18: Python Implementation of PCA](Lecture%2018%20Python%20implementation%20of%20PCA.md) — Hands-on PCA implementation for unsupervised reduction
- **Next:** [Lecture 20: Python Implementation of LDA](Lecture%2020%20Python%20implementation%20of%20LDA.md) — Hands-on coding of LDA using Moore-Penrose pseudoinverse
- **Related:** [Lecture 09: Eigenvalues and Eigenvectors](Lecture%2009%20Eigenvalues%20and%20Eigenvectors.md) — Eigenvalue problem for solving LDA
- **Related:** [Lecture 16: Principal Component Analysis](Lecture%2016%20Principal%20Component%20Analysis.md) — Unsupervised counterpart to supervised LDA
