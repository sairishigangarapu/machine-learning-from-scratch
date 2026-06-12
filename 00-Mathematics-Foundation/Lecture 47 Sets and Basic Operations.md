## Sets and Basic Operations

*Essential Mathematics for ML — Structured Notes*

---

## 1. What is a Set?

### Motivation and Intuition
Every concept in machine learning operates on sets. A dataset is a set of feature vectors. A class label defines a subset of the data. The hypothesis space is a set of possible models. The parameter space is a set of all possible weights. Before we can define probability, random variables, or distributions, we need the language of **set theory**.

### Definition
A **set** is a well-defined collection of distinct objects, called **elements** or **members**.

$$
A = \{1, 2, 3, 4, 5\}
$$

We write $x \in A$ to say "$x$ is an element of $A$."

### Notation

| Notation | Meaning |
|:---|:---|
| $A = \{1, 2, 3\}$ | Explicit set |
| $A = \{x : x > 0\}$ | Set-builder notation |
| $\emptyset$ | Empty set |
| $\mathbb{R}$ | Set of all real numbers |
| $\mathbb{R}^n$ | $n$-dimensional Euclidean space |

---

## 2. Set Operations

### Union ($A \cup B$)
Elements in $A$ **or** $B$ (or both):

$$
A \cup B = \{x : x \in A \text{ or } x \in B\}
$$

### Intersection ($A \cap B$)
Elements in both $A$ **and** $B$:

$$
A \cap B = \{x : x \in A \text{ and } x \in B\}
$$

### Difference ($A \setminus B$)
Elements in $A$ but **not** in $B$:

$$
A \setminus B = \{x : x \in A \text{ and } x \notin B\}
$$

### Complement ($A^c$)
Elements not in $A$:

$$
A^c = \{x : x \notin A\}
$$

```python
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

print(A | B)    # Union: {1, 2, 3, 4, 5, 6, 7, 8}
print(A & B)    # Intersection: {4, 5}
print(A - B)    # Difference: {1, 2, 3}
print(B - A)    # Difference: {6, 7, 8}
```

---

## 3. Key Laws

| Law | Formula |
|:---|:---|
| **Commutative** | $A \cup B = B \cup A$ |
| **Associative** | $(A \cup B) \cup C = A \cup (B \cup C)$ |
| **Distributive** | $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ |
| **De Morgan's** | $(A \cup B)^c = A^c \cap B^c$ |
| **De Morgan's** | $(A \cap B)^c = A^c \cup B^c$ |

**ML Connection:** De Morgan's laws appear in logic circuits and binary decision trees. When a decision tree splits on "feature $x_1 > 5$ AND $x_2 < 3$", the complement of this region is "$x_1 \le 5$ OR $x_2 \ge 3$" — exactly De Morgan's law.

---

## 4. Subsets and Power Sets

### Subset
$A \subseteq B$ if every element of $A$ is also in $B$.

### Power Set
The power set $\mathcal{P}(A)$ is the set of **all subsets** of $A$. If $|A| = n$, then $|\mathcal{P}(A)| = 2^n$.

```python
from itertools import combinations

def power_set(A):
    """Generate all subsets of set A."""
    result = []
    for r in range(len(A) + 1):
        for subset in combinations(A, r):
            result.append(set(subset))
    return result

A = {1, 2, 3}
print(f"|P(A)| = {len(power_set(A))}")  # 8 = 2^3
```

---

## 5. Cartesian Product

$$
A \times B = \{(a, b) : a \in A, \; b \in B\}
$$

If $|A| = m$ and $|B| = n$, then $|A \times B| = mn$.

**ML Connection:** The feature space of a dataset with $n$ features is the Cartesian product of the domain of each feature: $\mathcal{X} = \mathcal{X}_1 \times \mathcal{X}_2 \times \dots \times \mathcal{X}_n$.

---

## 6. Countable vs Uncountable Sets

* **Finite:** $\{1, 2, 3\}$ — exactly $n$ elements.
* **Countably infinite:** $\mathbb{N} = \{1, 2, 3, \dots\}$ — can be put in one-to-one correspondence with $\mathbb{N}$.
* **Uncountable:** $\mathbb{R}$ — cannot be listed; there are "more" real numbers than natural numbers.

**ML Connection:** When we discretize a continuous feature (e.g., binning ages into groups), we map from an uncountable set to a finite set. This loses information but enables certain algorithms (like Naive Bayes with discrete features).

---

## 7. Why Sets Matter in ML

| Concept | Set Interpretation |
|:---|:---|
| **Dataset** | Set of $(x_i, y_i)$ pairs |
| **Class** | Subset of data with label $y = k$ |
| **Feature space** | Cartesian product of feature domains |
| **Hypothesis space** | Set of all candidate models |
| **Decision boundary** | Partition of feature space into subsets |

> **Check your intuition:** If a dataset has 3 binary features (each can be 0 or 1), how many possible feature vectors exist? *(Answer: The feature space is $\{0,1\} \times \{0,1\} \times \{0,1\}$, so $|\mathcal{X}| = 2^3 = 8$. This is the Cartesian product of three copies of $\{0, 1\}$.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 46: Python Implementation of Convex Optimization](Lecture%2046%20Python%20Implementation%20of%20Convex%20Optimization.md) — Python implementation of mathematical optimization
- **Next:** [Lecture 48: Introduction to Probability](Lecture%2048%20Introduction%20to%20Probability.md) — Applies set theory to probability spaces
- **Related:** [Lecture 37: Convex Sets](Lecture%2037%20Convex%20Sets.md) — Specialized set concepts for optimization
