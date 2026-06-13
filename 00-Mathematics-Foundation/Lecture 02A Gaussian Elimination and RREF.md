## Lecture 02A: Gaussian Elimination & RREF

*Essential Mathematics for ML — Structured Notes*

---

### The Hacker's Catch
Most ML practitioners just call `np.linalg.solve()`. But if you want to pass a college exam or understand **Rank Deficiency** and **Null Spaces**, you must master the "Manual Transmission" of Linear Algebra: **Gaussian Elimination**. 

Gaussian Elimination isn't just "finding $x$"; it's a systematic way of tearing a matrix apart row-by-row until its hidden secrets (Rank, Free variables, and Pivots) are exposed.

---

### 1. Forward Elimination (The Upper Triangle)
The goal is to turn our matrix $A$ into an **Upper Triangular Matrix ($U$)**. We do this by adding multiples of one row to another to create zeros underneath each "Pivot".

**The Rules of the Game:**
1.  **Row Swapping:** You can swap any two rows.
2.  **Scaling:** You can multiply any row by a non-zero scalar.
3.  **Combination:** You can add a multiple of one row to another.

#### Example:
$$
A = \begin{bmatrix} 2 & 4 & -2 \\ 4 & 9 & -3 \\ -2 & -3 & 7 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A$ | Coefficient matrix | Represents the linear system to be solved via elimination |
| $2, 4, -2, \dots$ | Matrix entries | Coefficients of the variables in each equation |

**Step 1:** Use Row 1 as the pivot row. Multiply R1 by 2 and subtract from R2.
$$
\begin{bmatrix} 2 & 4 & -2 \\ 0 & 1 & 1 \\ -2 & -3 & 7 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\begin{bmatrix} 2 & 4 & -2 \\ 0 & 1 & 1 \\ -2 & -3 & 7 \end{bmatrix}$ | Matrix after Step 1 elimination | The $(2,1)$ entry is zeroed by subtracting $2 \times R1$ from R2 |

**Step 2:** Add R1 to R3.
$$
\begin{bmatrix} 2 & 4 & -2 \\ 0 & 1 & 1 \\ 0 & 1 & 5 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\begin{bmatrix} 2 & 4 & -2 \\ 0 & 1 & 1 \\ 0 & 1 & 5 \end{bmatrix}$ | Matrix after Step 2 elimination | The $(3,1)$ entry is zeroed by adding R1 to R3; upper triangular form emerging |

---

### 2. Row Reduced Echelon Form (RREF)
To reach the ultimate "Pure" form, we continue until:
1.  Every Pivot is exactly **1**.
2.  The area **above** each Pivot is also cleared to **0**.

In RREF, the matrix becomes so simple that the solution to $Ax=b$ can be read directly by looking at the rows.

#### Pivot vs. Free Variables
-   **Pivot Variable:** A column that contains a "1" in the RREF form. These are the "Locked" dimensions.
-   **Free Variable:** A column that lacks a pivot. You can pick *any* value for these, and they define the **Null Space** (the "wiggle room" of the system).

---

### 3. How to Hack RREF in Python
While `NumPy` is great for numbers, it doesn't give you the clean "Fractional" RREF we use in exams. For that, we use **SymPy**.

```python
import sympy as sp

# Define the matrix
A = sp.Matrix([[2, 4, -2], 
               [4, 9, -3], 
               [-2, -3, 7]])

# The rref() method returns the RREF form AND the indices of pivot columns
rref_matrix, (pivots) = A.rref()

print("RREF Form:")
sp.pprint(rref_matrix)
print(f"Pivot Columns: {pivots}")
```

#### Outcome:
If you end up with a row of all zeros (e.g., `[0 0 0]`), you have **Rank Deficiency**. This means the dimensions of your data are redundant.

---

### Practical Application
- **Null Space Magic:** Use RREF to find the [Null Space (L04)](Lecture%2004%20Vector%20Subspace.md). If you have free variables, your Null Space is non-trivial.
- **Inversion Check:** If RREF of an $n \times n$ matrix is the Identity ($I$), then the matrix is **Invertible**. If not, it's Singular.
- **Syllabus Link:** This is the manual mechanical counterpart to [Lecture 21: Least Squares](Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md).

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 02: Matrix Algebra](Lecture%2002%20Basics%20of%20Matrix%20Algebra.md) — Foundational matrix operations required before elimination
- **Next:** [Lecture 03: Vector Spaces](Lecture%2003%20Vector%20Space%20Definitions.md) — Abstract framework built on the rank and null space concepts from RREF
- **Related:** [Lecture 21: Least Squares](Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md) — Practical application of RREF to overdetermined systems
