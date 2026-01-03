
# 🧭 Vectors in Machine Learning

_Essential Mathematics for ML — Structured Notes_

---

## 📌 1. What is a Vector?

A **vector** is a mathematical object that has:

- **Magnitude (length)**
    
- **Direction**
    

### Formal definition

A vector is an element of a **vector space**, which supports:

1. **Vector addition**
    
2. **Scalar multiplication**
    

### Representation

- **Row vector:** `[v₁ v₂ ... vₙ]`
    
- **Column vector:**
    
    ```
    [ v₁
      v₂
      ⋮
      vₙ ]
    ```
    

---

## 📌 2. Vectors in ℝⁿ

A vector  
`v = (v₁, v₂, …, vₙ)`  
belongs to the vector space **ℝⁿ** if all components are real numbers.

### Examples

- **ℝ²:** (1, 2)
    
- **ℝ³:** (1, 2, 3)
    
- Higher dimensions exist but cannot be visualized.
    

---

## 📌 3. Geometric Interpretation

- Each component = coordinate along an axis
    
- In 2D → (x, y)
    
- In 3D → (x, y, z)
    

A vector = **arrow from origin** representing direction + length.

---

# 🧮 Vector Algebra

---

## ✔️ 4. Addition & Subtraction

Defined **component-wise**.

Given  
`v₁ = (x₁, x₂, …, xₙ)`  
`v₂ = (y₁, y₂, …, yₙ)`

### Addition

```
v₁ + v₂ = (x₁ + y₁, x₂ + y₂, …, xₙ + yₙ)
```

### Subtraction

```
v₁ - v₂ = (x₁ - y₁, x₂ - y₂, …, xₙ - yₙ)
```

---

## ✔️ 5. Dot Product (Inner Product)

Given two vectors in ℝⁿ:

```
v₁ · v₂ = Σ (xᵢ yᵢ)
```

Example in ℝ³:  
`(1, 1, -1) · (2, 3, 1) = 4`

→ Result is always a **scalar**.

---

## ✔️ 6. Magnitude (Length / Norm)

```
‖v‖ = √(v · v)
```

Example:  
`v = (1, -1, 2)`

```
‖v‖ = √(1² + (-1)² + 2²) = √6
```

---

## ✔️ 7. Angle Between Two Vectors

```
cos θ = (v₁ · v₂) / (‖v₁‖ ‖v₂‖)
```
```
sin θ = (v₁ x v₂) / (‖v₁‖ ‖v₂‖)
```

So,

```
θ = cos⁻¹( (v₁ · v₂) / (‖v₁‖ ‖v₂‖) )
```
```
θ = sin⁻¹( (v₁ x v₂) / (‖v₁‖ ‖v₂‖) )
```
---

# 🔗 8. Linear Combination

Given vectors `{v₁, v₂, …, vₖ}`:

A **linear combination** is:

```
α₁v₁ + α₂v₂ + ... + αₖvₖ
```

α’s are scalars (usually real numbers).

Example:  
Using vectors v₁, v₂, v₃ in ℝ³, any vector formed like:

```
α₁v₁ + α₂v₂ + α₃v₃
```

is a linear combination.

---

# 🔍 9. Linear Independence & Dependence

## ✔️ Linear Independence (LI)

Set `{v₁, v₂, …, vₙ}` is LI if:

```
α₁v₁ + α₂v₂ + ... + αₙvₙ = 0
```

only when:

```
α₁ = α₂ = ... = αₙ = 0
```

### Intuition

You **cannot build** one vector using others.

---

## ❌ Linear Dependence (LD)

Vectors are LD if:

```
α₁v₁ + α₂v₂ + ... = 0
```

for **some non-zero scalars**.

### Example

(1,1) and (3,3) → LD because:

```
3(1,1) - 1(3,3) = 0
```

---

## Important Remarks

- In **ℝⁿ**, any set of **> n vectors is LD**
    
- Any set **containing the zero vector is LD**
    

---

# 🎯 10. Orthogonal & Orthonormal Vectors

## ✔️ Orthogonal

```
vᵢ · vⱼ = 0, for all i ≠ j
```

→ They are perpendicular.

### Important

Orthogonal vectors are **automatically linearly independent**.

---

## ✔️ Orthonormal

Set is orthonormal if:

1. Vectors are orthogonal
    
2. Each vector has length = 1
    

Example in ℝ²:

```
(1/√2, 1/√2)
(1/√2, -1/√2)
```

---

# 📊 11. Vectors as Feature Vectors in ML

Consider this dataset:

|Employee|Height|Weight|
|---|---|---|
|E₁|α₁|β₁|
|E₂|α₂|β₂|
|…|…|…|
|Eₖ|αₖ|βₖ|

For employee E₂:

```
Feature vector = ( height, weight ) = (α₂, β₂)
```

In ML:

- **rows** = samples
    
- **columns** = features
    
- each row vector = **feature vector**
    

---

# 🐍 12. Python / NumPy Operations

```python
import numpy as np

v = np.array([1, -1, 2])
w = np.array([2, 5, 2])

print(v + w)          # Addition
print(v - w)          # Subtraction
print(3 * v)          # Scalar multiplication
print(np.linalg.norm(v)) # Length
print(np.dot(v, w))      # Dot product
```

Outputs include addition, subtraction, norm, and dot product.

---

