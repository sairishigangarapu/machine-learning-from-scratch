## ASCII Matrix Intuition

*Essential Mathematics for ML — Visual Notes*

---

## 1. How an ($m \times n$) Matrix Turns Into a Vector Map

In Machine Learning datasets, remember this golden rule:
**Rows $m$ = Data Samples** (e.g., 10,000 individual houses)
**Columns $n$ = Features** (e.g., square footage, number of bedrooms, location)

An $m \times n$ matrix $A$ represents $m$ separate vectors piled exactly on top of each other. 

**Matrix $A \in \mathbb{R}^{m \times n}$**

```text
      Feature 1     Feature 2       ...       Feature n
     ┌─────────┬─────────┬───────┬─────────┐
Row 1│  a_{11} │  a_{12} │ \dots │  a_{1n} │ -> Sample 1 (Vector 1)
     ├─────────┼─────────┼───────┼─────────┤
Row 2│  a_{21} │  a_{22} │ \dots │  a_{2n} │ -> Sample 2 (Vector 2)
     ├─────────┼─────────┼───────┼─────────┤
 ... │  \dots  │  \dots  │ \dots │  \dots  │ -> \dots
     ├─────────┼─────────┼───────┼─────────┤
Row m│  a_{m1} │  a_{m2} │ \dots │  a_{mn} │ -> Sample m (Vector m)
     └─────────┴─────────┴───────┴─────────┘
```

When you perform operations on this matrix (like Neural Network layer multiplying $A \times W$), the linear algebra engine processes all $m$ rows in absolute parallel simultaneously. This is the secret to GPU acceleration!
