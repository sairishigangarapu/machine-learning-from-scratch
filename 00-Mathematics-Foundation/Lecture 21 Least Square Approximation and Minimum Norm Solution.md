## Least Square Approximation and Minimum Norm Solution

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Linear Systems: The Three Realities

Every time you "train" a linear model, you are actually solving a system of equations $A\mathbf{x} = \mathbf{b}$. Depending on how much data you have versus how many parameters you're trying to find, you land in one of three mathematical realities:

| Reality | Condition | The Outcome |
|:---|:---|:---|
| **Square system** | $m = n$ | Unique solution: $\mathbf{x} = A^{-1}\mathbf{b}$. Perfect, but rare in ML. |
| **Over-determined** | $m > n$ | More data than parameters. No exact solution exists. We seek the **Least Square Approximation**. |
| **Under-determined** | $m < n$ | More parameters than data. Infinite solutions exist. We seek the **Minimum Norm Solution**. |

---

## 2. Over-Determined Systems

### Intuition
When $m > n$, there are **more equations than unknowns** (more observations than free parameters). This is the standard Machine Learning scenario: you have $10,000$ rows of data but only $10$ features. There is no line that passes through every point because the world is noisy. 

Instead of an impossible "perfect" solution, we find the $\mathbf{x}$ that is the **Least Wrong**.

**Classic example:** Fitting a line to $m$ data points when $m > 2$.

### The Residual
For a candidate solution $\mathbf{x}$, the **residual vector** is:

$$ \mathbf{r} = \mathbf{b} - A\mathbf{x} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{r}$ | Residual vector | Error between observed and predicted values |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $A\mathbf{x}$ | Model predictions | Linear combination of features with parameters |
| $\mathbf{x}$ | Parameter vector | Unknown coefficients being solved for |
Each component $r_i$ is the directed distance between the $i$-th observed value $b_i$ and the value predicted by the model $A\mathbf{x}$.

### The Least Square Problem
Minimize the squared Euclidean norm of the residual:

$$ \min_{\mathbf{x} \in \mathbb{R}^n} \|A\mathbf{x} - \mathbf{b}\|_2^2 $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $A\mathbf{x}$ | Model predictions | Linear combination of features with parameters |
| $\mathbf{x}$ | Parameter vector | Unknown coefficients being solved for |
| $\|A\mathbf{x} - \mathbf{b}\|_2^2$ | Squared residual norm | Objective function for least squares |
| $\mathbb{R}^n$ | $n$-dimensional real space | Domain of real-valued vectors |
This is equivalent to minimizing the **sum of squared residuals** $\sum_{i=1}^{m} r_i^2$ — hence the name *least squares*.

---

## 3. Solving the Least Square Problem — Normal Equations

### Derivation
Set the partial derivatives of $E(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|_2^2$ to zero:

$$ \frac{\partial E}{\partial \mathbf{x}} = 2A^T(A\mathbf{x} - \mathbf{b}) = \mathbf{0} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $A\mathbf{x}$ | Model predictions | Linear combination of features with parameters |
| $\mathbf{x}$ | Parameter vector | Unknown coefficients being solved for |
| $\frac{\partial E}{\partial \mathbf{x}}$ | Gradient of error | Set to zero to derive normal equations |
| $E$ | Sum of squared errors | Objective function $\|A\mathbf{x} - \mathbf{b}\|_2^2$ |
This yields the **Normal Equations**:

$$ \boxed{A^T A \, \mathbf{x} = A^T \mathbf{b}} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $\mathbf{x}$ | Parameter vector | Unknown coefficients being solved for |
| $A^T A$ | Gram matrix | Product encoding pairwise column dot products; key in normal equations |
| $A^T \mathbf{b}$ | Cross-product vector | Projects observations into column space of $A$ |
### The Pseudo-Inverse Solution
$A^T A$ is an $n \times n$ matrix. If $\text{rank}(A) = n$ (columns are linearly independent), then $A^T A$ is invertible and:

$$ \mathbf{x}_{\text{LS}} = (A^T A)^{-1} A^T \mathbf{b} = A^{+} \mathbf{b} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $\mathbf{x}_{\text{LS}}$ | Least squares solution | Optimal parameters minimizing squared error |
| $A^T A$ | Gram matrix | Product encoding pairwise column dot products; key in normal equations |
| $(A^T A)^{-1}$ | Inverse Gram matrix | Invertible when $A$ has full column rank |
| $A^{+}$ | Moore-Penrose pseudoinverse | Generalizes matrix inverse to non-square systems |
| $A^T \mathbf{b}$ | Cross-product vector | Projects observations into column space of $A$ |
where $A^{+} = (A^T A)^{-1} A^T$ is the **right pseudo-inverse** of $A$.

---

## 4. Worked Example — Over-Determined System

**Problem:** Find the least square solution of $A\mathbf{x} = \mathbf{b}$ where:

$$ A = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
**Step 1: Compute $A^T A$**

$$ A^T A = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A^T A$ | Gram matrix | Product encoding pairwise column dot products; key in normal equations |
**Step 2: Compute $(A^T A)^{-1}$**

$$ \det(A^T A) = 3 \cdot 5 - 3 \cdot 3 = 6 $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\det(A^T A)$ | Determinant of Gram matrix | Non-zero confirms $A^T A$ is invertible |

$$ (A^T A)^{-1} = \frac{1}{6} \begin{bmatrix} 5 & -3 \\ -3 & 3 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A^T A$ | Gram matrix | Product encoding pairwise column dot products; key in normal equations |
| $(A^T A)^{-1}$ | Inverse Gram matrix | Invertible when $A$ has full column rank |
| RHS | Computed result | Result of matrix multiplication / arithmetic |
**Step 3: Compute $A^T \mathbf{b}$**

$$ A^T \mathbf{b} = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 6 \\ 0 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $A^T \mathbf{b}$ | Cross-product vector | Projects observations into column space of $A$ |
**Step 4: Compute $\mathbf{x}_{\text{LS}} = (A^T A)^{-1} A^T \mathbf{b}$**

$$ \mathbf{x}_{\text{LS}} = \frac{1}{6} \begin{bmatrix} 5 & -3 \\ -3 & 3 \end{bmatrix} \begin{bmatrix} 6 \\ 0 \end{bmatrix} = \begin{bmatrix} 5 \\ -3 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}_{\text{LS}}$ | Least squares solution | Optimal parameters minimizing squared error |
**Result:** $x_1 = 5, x_2 = -3$.

---

## 5. Under-Determined Systems

### Intuition
When $m < n$, there are **fewer equations than unknowns**. Any $n - m$ of the unknown variables can be chosen freely — infinitely many solutions exist. Out of all these solutions, we seek the one with the **smallest Euclidean norm** (shortest vector).

### The Minimum Norm Problem

$$ \min_{\mathbf{x} \in \mathbb{R}^n} \|\mathbf{x}\|_2 \quad \text{subject to} \quad A\mathbf{x} = \mathbf{b} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $A\mathbf{x}$ | Model predictions | Linear combination of features with parameters |
| $\mathbf{x}$ | Parameter vector | Unknown coefficients being solved for |
| $\|\mathbf{x}\|_2$ | Euclidean norm | Length of the vector; minimized in minimum norm problems |
| $\mathbb{R}^n$ | $n$-dimensional real space | Domain of real-valued vectors |
| $\min \|\mathbf{x}\|_2$ | Minimum norm objective | Find the shortest possible solution |
| $A\mathbf{x} = \mathbf{b}$ | Equality constraint | Solution must satisfy original system |
### Why Not Use $(A^T A)^{-1} A^T$?
Here $A$ is $m \times n$ with $m < n$. The matrix $A^T A$ is $n \times n$ but has $\text{rank}(A) \le m < n$ — it is **rank-deficient** and its inverse does not exist.

### The Left Pseudo-Inverse Solution
Instead, form $AA^T$, which is $m \times m$. If $\text{rank}(A) = m$, then $AA^T$ is invertible and:

$$ \boxed{\mathbf{x}_{\text{MN}} = A^T (AA^T)^{-1} \mathbf{b} = A^{+} \mathbf{b}} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $\mathbf{x}_{\text{MN}}$ | Minimum norm solution | Shortest vector satisfying constraints |
| $AA^T$ | Gram matrix (row-wise) | Product encoding pairwise row dot products |
| $(AA^T)^{-1}$ | Inverse row Gram matrix | Invertible when $A$ has full row rank |
| $A^{+}$ | Moore-Penrose pseudoinverse | Generalizes matrix inverse to non-square systems |
where $A^{+} = A^T(AA^T)^{-1}$ is the **left pseudo-inverse** of $A$. This $\mathbf{x}_{\text{MN}}$ is the unique solution to $A\mathbf{x} = \mathbf{b}$ with minimum Euclidean norm.

> **Hacker's Connection:** This Minimum Norm solution is the theoretical grandparent of **L2 Regularization (Ridge Regression)**. By forcing the weights to have a small norm, we prevent the model from becoming too "wild" and overfitting the few samples we have.

---

## 6. Worked Example — Under-Determined System

**Problem:** Find the minimum norm solution of $A\mathbf{x} = \mathbf{b}$ where:

$$ A = \begin{bmatrix} 1 & 1 & 1 \\ -1 & -1 & 1 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{b}$ | Observation vector | Actual values being modeled / right-hand side |
| $A$ | Matrix result | Numerical value of the computation |
| RHS | Computed result | Result of matrix multiplication / arithmetic |
Note: $m = 2, n = 3$ — under-determined, infinitely many solutions.

**Step 1: Compute $AA^T$**

$$ AA^T = \begin{bmatrix} 1 & 1 & 1 \\ -1 & -1 & 1 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ 1 & -1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 3 & -1 \\ -1 & 3 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $AA^T$ | Gram matrix (row-wise) | Product encoding pairwise row dot products |
**Step 2: Compute $(AA^T)^{-1}$**

$$ \det(AA^T) = 9 - 1 = 8 $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\det(AA^T)$ | Determinant of $AA^T$ | Non-zero confirms $AA^T$ is invertible |

$$ (AA^T)^{-1} = \frac{1}{8} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $AA^T$ | Gram matrix (row-wise) | Product encoding pairwise row dot products |
| $(AA^T)^{-1}$ | Inverse row Gram matrix | Invertible when $A$ has full row rank |
| RHS | Computed result | Result of matrix multiplication / arithmetic |
**Step 3: Compute the Left Pseudo-Inverse $A^{+} = A^T (AA^T)^{-1}$**

$$ A^{+} = \frac{1}{8} \begin{bmatrix} 1 & -1 \\ 1 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = \frac{1}{8} \begin{bmatrix} 2 & -2 \\ 2 & -2 \\ 4 & 4 \end{bmatrix} = \begin{bmatrix} 1/4 & -1/4 \\ 1/4 & -1/4 \\ 1/2 & 1/2 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A^{+}$ | Moore-Penrose pseudoinverse | Generalizes matrix inverse to non-square systems |
**Step 4: Compute $\mathbf{x}_{\text{MN}} = A^{+} \mathbf{b}$**

$$ \mathbf{x}_{\text{MN}} = \begin{bmatrix} 1/4 & -1/4 \\ 1/4 & -1/4 \\ 1/2 & 1/2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1/4 \\ 1/4 \\ 1/2 \end{bmatrix} $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}_{\text{MN}}$ | Minimum norm solution | Shortest vector satisfying constraints |
**Verification:**

$$ \|\mathbf{x}_{\text{MN}}\|_2 = \sqrt{\left(\frac{1}{4}\right)^2 + \left(\frac{1}{4}\right)^2 + \left(\frac{1}{2}\right)^2} = \sqrt{\frac{6}{16}} = \frac{\sqrt{6}}{4} \approx 0.612 $$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}_{\text{MN}}$ | Minimum norm solution | Shortest vector satisfying constraints |
Any other solution (e.g., $[0, 1/2, 1/2]^T$ with norm $1/\sqrt{2} \approx 0.707$) has a **strictly larger** norm — confirming this is the minimum.

---

## 7. Summary — Pseudo-Inverse Variants

| System Type | Condition | Pseudo-Inverse | Formula |
|---|---|---|---|
| Over-determined | $m > n, \text{rank}(A)=n$ | Right pseudo-inverse | $A^{+} = (A^T A)^{-1} A^T$ |
| Under-determined | $m < n, \text{rank}(A)=m$ | Left pseudo-inverse | $A^{+} = A^T (AA^T)^{-1}$ |

In both cases, the solution takes the unified form $\mathbf{x} = A^{+} \mathbf{b}$. For a general matrix, the full **Moore-Penrose pseudo-inverse** (computed via SVD) handles all cases simultaneously.

---

## 8. Connection to Linear Regression

The least square solution is the mathematical backbone of **linear regression**:

- Each data point $(x_i, y_i)$ gives one row of $A$ and one entry of $\mathbf{b}$.
- With $m$ data points and $n$ model parameters ($m > n$), the system is over-determined.
- The least square solution $\mathbf{x}_{\text{LS}} = (A^T A)^{-1} A^T \mathbf{b}$ gives the **optimal regression coefficients** — the line (or hyperplane) that minimizes the total squared error across all data points.

**Next Step:** We will apply this "Regression Engine" to solve complex forecasting problems in **Multiple and Polynomial Regression**.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 20: Python Implementation of LDA](Lecture%2020%20Python%20implementation%20of%20LDA.md) — Supervised dimensionality reduction before regression
- **Next:** [Lecture 22: Linear and Multiple Regression](Lecture%2022%20Linear%20and%20Multiple%20Regression.md) — Applying least squares to regression models
- **Related:** [Lecture 08: Orthogonal Complement and Projection](Lecture%2008%20Orthogonal%20Complement%20and%20Projection%20Mapping.md) — Projection interpretation of least squares
- **Related:** [Lecture 12: Singular Value Decomposition](Lecture%2012%20Singular%20Value%20Decomposition.md) — SVD for computing the Moore-Penrose pseudoinverse
- **Related:** [Lecture 35: Chain Rule](Lecture%2035%20Chain%20Rule.md) — Gradient derivation via chain rule for optimization
