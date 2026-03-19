## Least Square Approximation and Minimum Norm Solution

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Linear Systems — Three Cases

Consider the linear system $Ax = b$, where $A \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^n, b \in \mathbb{R}^m$.

| Condition | Name | Solutions |
|---|---|---|
| $m = n$, $A$ invertible | **Square system** | Unique: $x = A^{-1}b$ |
| $m > n$ | **Over-determined** | Rarely exact — seek **least square approximation** |
| $m < n$ | **Under-determined** | Infinitely many — seek **minimum norm solution** |

---

## 2. Over-Determined Systems

### Intuition
When $m > n$, there are **more equations than unknowns**. An exact solution almost never exists. We instead find the $x$ that comes *closest* to satisfying all equations simultaneously.

**Classic example:** Fitting a line to $m$ data points when $m > 2$.

### The Residual
For a candidate solution $x$, the **residual vector** is:

$$
r = b - Ax
$$

Each component $r_i$ is the directed distance between the $i$-th observed value $b_i$ and the value predicted by the model $Ax$.

### The Least Square Problem
Minimize the squared Euclidean norm of the residual:

$$
\min_{x \in \mathbb{R}^n} \|Ax - b\|_2^2
$$

This is equivalent to minimizing the **sum of squared residuals** $\sum r_i^2$ — hence the name *least squares*.

---

## 3. Solving the Least Square Problem — Normal Equations

### Derivation
Set the partial derivatives of $E(x) = \|Ax - b\|_2^2$ to zero:

$$
\frac{\partial E}{\partial x} = 2A^T(Ax - b) = 0
$$

This yields the **Normal Equations**:

$$
\boxed{A^T A x = A^T b}
$$

### The Pseudo-Inverse Solution
$A^T A$ is an $n \times n$ matrix. If $\text{rank}(A) = n$, then $A^T A$ is invertible and:

$$
x_{\text{LS}} = (A^T A)^{-1} A^T b = A^{+} b
$$

where $A^{+} = (A^T A)^{-1} A^T$ is the **right pseudo-inverse** of $A$.

---

## 4. Worked Example — Over-Determined System

**Problem:** Find the least square solution of $Ax = b$ where:

$$
A = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \qquad b = \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix}
$$

**Step 1: Compute $A^T A$**

$$
A^T A = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}
$$

**Step 2: Compute $(A^T A)^{-1}$**

$$
\det(A^T A) = 3 \cdot 5 - 3 \cdot 3 = 6
$$

$$
(A^T A)^{-1} = \frac{1}{6} \begin{bmatrix} 5 & -3 \\ -3 & 3 \end{bmatrix}
$$

**Step 3: Compute $A^T b$**

$$
A^T b = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 6 \\ 0 \end{bmatrix}
$$

**Step 4: Compute $x_{\text{LS}} = (A^T A)^{-1} A^T b$**

$$
x_{\text{LS}} = \frac{1}{6} \begin{bmatrix} 5 & -3 \\ -3 & 3 \end{bmatrix} \begin{bmatrix} 6 \\ 0 \end{bmatrix} = \begin{bmatrix} 5 \\ -3 \end{bmatrix}
$$

**Result:** $x_1 = 5, x_2 = -3$.

---

## 5. Under-Determined Systems

### Intuition
When $m < n$, there are **fewer equations than unknowns**. Out of infinitely many solutions, we seek the one with the **smallest Euclidean norm**.

### The Minimum Norm Problem

$$
\min_{x \in \mathbb{R}^n} \|x\|_2 \quad \text{subject to} \quad Ax = b
$$

### The Left Pseudo-Inverse Solution
If $\text{rank}(A) = m$, then $AA^T$ is invertible and:

$$
\boxed{x_{\text{MN}} = A^T (AA^T)^{-1} b = A^{+} b}
$$

where $A^{+} = A^T(AA^T)^{-1}$ is the **left pseudo-inverse** of $A$.

---

## 6. Worked Example — Under-Determined System

**Problem:** Find the minimum norm solution of $Ax = b$ where:

$$
A = \begin{bmatrix} 1 & 1 & 1 \\ -1 & -1 & 1 \end{bmatrix}, \qquad b = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

**Step 1: Compute $AA^T$**

$$
AA^T = \begin{bmatrix} 1 & 1 & 1 \\ -1 & -1 & 1 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ 1 & -1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 3 & -1 \\ -1 & 3 \end{bmatrix}
$$

**Step 2: Compute $(AA^T)^{-1}$**

$$
\det(AA^T) = 9 - 1 = 8
$$

$$
(AA^T)^{-1} = \frac{1}{8} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
$$

**Step 3: Compute the Left Pseudo-Inverse $A^{+} = A^T (AA^T)^{-1}$**

$$
A^{+} = \frac{1}{8} \begin{bmatrix} 1 & -1 \\ 1 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 1/4 & -1/4 \\ 1/4 & -1/4 \\ 1/2 & 1/2 \end{bmatrix}
$$

**Step 4: Compute $x_{\text{MN}} = A^{+} b$**

$$
x_{\text{MN}} = \begin{bmatrix} 1/4 & -1/4 \\ 1/4 & -1/4 \\ 1/2 & 1/2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1/4 \\ 1/4 \\ 1/2 \end{bmatrix}
$$

**Result:** $x_1 = 1/4, x_2 = 1/4, x_3 = 1/2$.

---

## 7. Summary

| System | Condition | Pseudo-Inverse | Formula |
|---|---|---|---|
| Over-determined | $m > n$ | Right | $A^{+} = (A^T A)^{-1} A^T$ |
| Under-determined | $m < n$ | Left | $A^{+} = A^T (AA^T)^{-1}$ |

In both cases, $x = A^{+} b$.
