## Least Square Approximation and Minimum Norm Solution

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Linear Systems — Three Cases

Consider the linear system $A\mathbf{x} = \mathbf{b}$, where $A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

| Condition | Name | Solutions |
|---|---|---|
| $m = n$, $A$ invertible | **Square system** | Unique: $\mathbf{x} = A^{-1}\mathbf{b}$ |
| $m > n$ | **Over-determined** | Rarely exact — seek **least square approximation** |
| $m < n$ | **Under-determined** | Infinitely many — seek **minimum norm solution** |

---

## 2. Over-Determined Systems

### Intuition
When $m > n$, there are **more equations than unknowns** (more observations than free parameters). An exact solution almost never exists. We instead find the $\mathbf{x}$ that comes *closest* to satisfying all equations simultaneously.

**Classic example:** Fitting a line to $m$ data points when $m > 2$.

### The Residual
For a candidate solution $\mathbf{x}$, the **residual vector** is:

$$
\mathbf{r} = \mathbf{b} - A\mathbf{x}
$$

Each component $r_i$ is the directed distance between the $i$-th observed value $b_i$ and the value predicted by the model $A\mathbf{x}$.

### The Least Square Problem
Minimize the squared Euclidean norm of the residual:

$$
\min_{\mathbf{x} \in \mathbb{R}^n} \|A\mathbf{x} - \mathbf{b}\|_2^2
$$

This is equivalent to minimizing the **sum of squared residuals** $\sum_{i=1}^{m} r_i^2$ — hence the name *least squares*.

---

## 3. Solving the Least Square Problem — Normal Equations

### Derivation
Set the partial derivatives of $E(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|_2^2$ to zero:

$$
\frac{\partial E}{\partial \mathbf{x}} = 2A^T(A\mathbf{x} - \mathbf{b}) = \mathbf{0}
$$

This yields the **Normal Equations**:

$$
\boxed{A^T A \, \mathbf{x} = A^T \mathbf{b}}
$$

### The Pseudo-Inverse Solution
$A^T A$ is an $n \times n$ matrix. If $\text{rank}(A) = n$ (columns are linearly independent), then $A^T A$ is invertible and:

$$
\mathbf{x}_{\text{LS}} = (A^T A)^{-1} A^T \mathbf{b} = A^{+} \mathbf{b}
$$

where $A^{+} = (A^T A)^{-1} A^T$ is the **right pseudo-inverse** of $A$.

---

## 4. Worked Example — Over-Determined System

**Problem:** Find the least square solution of $A\mathbf{x} = \mathbf{b}$ where:

$$
A =
\begin{bmatrix}
1 & 0 \\
1 & 1 \\
1 & 2
\end{bmatrix}, \qquad
\mathbf{b} =
\begin{bmatrix}
6 \\
0 \\
0
\end{bmatrix}
$$

**Step 1: Compute $A^T A$**

$$
A^T A =
\begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 2
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
1 & 1 \\
1 & 2
\end{bmatrix}
=
\begin{bmatrix}
3 & 3 \\
3 & 5
\end{bmatrix}
$$

**Step 2: Compute $(A^T A)^{-1}$**

$$
\det(A^T A) = 3 \cdot 5 - 3 \cdot 3 = 6
$$

$$
(A^T A)^{-1} = \frac{1}{6}
\begin{bmatrix}
 5 & -3 \\
-3 &  3
\end{bmatrix}
$$

**Step 3: Compute $A^T \mathbf{b}$**

$$
A^T \mathbf{b} =
\begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 2
\end{bmatrix}
\begin{bmatrix}
6 \\
0 \\
0
\end{bmatrix}
=
\begin{bmatrix}
6 \\
0
\end{bmatrix}
$$

**Step 4: Compute $\mathbf{x}_{\text{LS}} = (A^T A)^{-1} A^T \mathbf{b}$**

$$
\mathbf{x}_{\text{LS}} = \frac{1}{6}
\begin{bmatrix}
 5 & -3 \\
-3 &  3
\end{bmatrix}
\begin{bmatrix}
6 \\
0
\end{bmatrix}
=
\begin{bmatrix}
5 \\
-3
\end{bmatrix}
$$

**Result:** $x_1 = 5$, $x_2 = -3$.

---

## 5. Under-Determined Systems

### Intuition
When $m < n$, there are **fewer equations than unknowns**. Any $n - m$ of the unknown variables can be chosen freely — infinitely many solutions exist. Out of all these solutions, we seek the one with the **smallest Euclidean norm** (shortest vector).

### The Minimum Norm Problem

$$
\min_{\mathbf{x} \in \mathbb{R}^n} \|\mathbf{x}\|_2 \quad \text{subject to} \quad A\mathbf{x} = \mathbf{b}
$$

### Why Not Use $(A^{T} A)^{-1} A^T$?
Here $A$ is $m \times n$ with $m < n$. The matrix $A^T A$ is $n \times n$ but has $\text{rank}(A) \le m < n$ — it is **rank-deficient** and its inverse does not exist.

### The Left Pseudo-Inverse Solution
Instead, form $A A^T$, which is $m \times m$. If $\text{rank}(A) = m$, then $A A^T$ is invertible and:

$$
\boxed{\mathbf{x}_{\text{MN}} = A^T (A A^T)^{-1} \mathbf{b} = A^{+} \mathbf{b}}
$$

where $A^{+} = A^T(AA^T)^{-1}$ is the **left pseudo-inverse** of $A$. This $\mathbf{x}_{\text{MN}}$ is the unique solution to $A\mathbf{x} = \mathbf{b}$ with minimum Euclidean norm.

---

## 6. Worked Example — Under-Determined System

**Problem:** Find the minimum norm solution of $A\mathbf{x} = \mathbf{b}$ where:

$$
A =
\begin{bmatrix}
 1 &  1 & 1 \\
-1 & -1 & 1
\end{bmatrix}, \qquad
\mathbf{b} =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

Note: $m = 2$, $n = 3$ — under-determined, infinitely many solutions.

**Step 1: Compute $A A^T$**

$$
A A^T =
\begin{bmatrix}
 1 &  1 & 1 \\
-1 & -1 & 1
\end{bmatrix}
\begin{bmatrix}
 1 & -1 \\
 1 & -1 \\
 1 &  1
\end{bmatrix}
=
\begin{bmatrix}
3 & -1 \\
-1 & 3
\end{bmatrix}
$$

**Step 2: Compute $(A A^T)^{-1}$**

$$
\det(A A^T) = 9 - 1 = 8
$$

$$
(A A^T)^{-1} = \frac{1}{8}
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}
$$

**Step 3: Compute the Left Pseudo-Inverse $A^{+} = A^T (A A^T)^{-1}$**

$$
A^{+} = \frac{1}{8}
\begin{bmatrix}
 1 & -1 \\
 1 & -1 \\
 1 &  1
\end{bmatrix}
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}
=
\frac{1}{8}
\begin{bmatrix}
2 & -2 \\
2 & -2 \\
4 &  4
\end{bmatrix}
=
\begin{bmatrix}
 1/4 & -1/4 \\
 1/4 & -1/4 \\
 1/2 &  1/2
\end{bmatrix}
$$

**Step 4: Compute $\mathbf{x}_{\text{MN}} = A^{+} \mathbf{b}$**

$$
\mathbf{x}_{\text{MN}} =
\begin{bmatrix}
 1/4 & -1/4 \\
 1/4 & -1/4 \\
 1/2 &  1/2
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
1/4 \\
1/4 \\
1/2
\end{bmatrix}
$$

**Verification:**

$$
\|\mathbf{x}_{\text{MN}}\|_2 = \sqrt{\left(\frac{1}{4}\right)^2 + \left(\frac{1}{4}\right)^2 + \left(\frac{1}{2}\right)^2} = \sqrt{\frac{6}{16}} = \frac{\sqrt{6}}{4} \approx 0.612
$$

Any other solution (e.g., $[0,\ 1/2,\ 1/2]^T$ with norm $1/\sqrt{2} \approx 0.707$) has a **strictly larger** norm — confirming this is the minimum.

---

## 7. Summary — Pseudo-Inverse Variants

| System Type | Condition | Pseudo-Inverse | Formula |
|---|---|---|---|
| Over-determined | $m > n$, $\text{rank}(A)=n$ | Right pseudo-inverse | $A^{+} = (A^T A)^{-1} A^T$ |
| Under-determined | $m < n$, $\text{rank}(A)=m$ | Left pseudo-inverse | $A^{+} = A^T (A A^T)^{-1}$ |

In both cases, the solution takes the unified form $\mathbf{x} = A^{+} \mathbf{b}$. For a general matrix, the full **Moore-Penrose pseudo-inverse** (computed via SVD) handles all cases simultaneously.

---

## 8. Connection to Linear Regression

The least square solution is the mathematical backbone of **linear regression**:

- Each data point $(x_i, y_i)$ gives one row of $A$ and one entry of $\mathbf{b}$.
- With $m$ data points and $n$ model parameters ($m > n$), the system is over-determined.
- The least square solution $\mathbf{x}_{\text{LS}} = (A^T A)^{-1} A^T \mathbf{b}$ gives the **optimal regression coefficients** — the line (or hyperplane) that minimizes the total squared error across all data points.
