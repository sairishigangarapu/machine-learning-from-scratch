## 1. Metric Spaces and Norms

### Motivation and Intuition
How do you teach a machine that a prediction is "bad"? If an algorithm predicts a house costs $\$200,000$and the real price is$\$250,000$, the error is $\$50,000$.

But what if the algorithm is predicting an image? How do you mathematically measure the "distance" between a picture of a cat and a picture of a dog? We need a rigorous mathematical formulation of what "distance" (Metric) and "size/length" (Norm) actually mean in high-dimensional spaces. The entire concept of Loss Functions in Machine Learning is built on Norms.

### Metric Space Definition
A metric is a function $d(x,y)$representing the distance between$x$and$y$, satisfying:
1. $d(x, y) \ge 0$(Distances can't be negative).
2.$d(x, y) = d(y, x)$(Symmetry).
3.$d(x, z) \le d(x, y) + d(y, z)$(Triangle Inequality: shortcuts are always faster).

### Normed Space Definition
A norm$\|x\|$is the specific "length" of a single vector. Every norm automatically produces a metric:$d(x,y) = \|x - y\|$.
1. $\|x\| \ge 0$2.$\|\alpha x\| = |\alpha| \cdot \|x\|$3.$\|x + y\| \le \|x\| + \|y\|$---

## 2. Common Norms in Machine Learning

For a vector$x \in \mathbb{R}^n$, there are infinite ways to measure its size, but ML primarily uses two variants of the $p$-norm family.

$$ \lVert x \rVert_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p} $$

### 1. L2 Norm (Euclidean Norm)
The standard straight-line "as the crow flies" distance. This is the foundation of **Mean Squared Error (MSE)** loss functions and Ridge (L2) Regularization.
$$\lVert x \rVert_2 = \sqrt{\sum x_i^2}$$

### 2. L1 Norm (Manhattan Norm)
The "city block" distance, summing the absolute steps across each axis. It is the foundation of **Mean Absolute Error (MAE)** and Lasso (L1) Regularization.
$$\lVert x \rVert_1 = \sum |x_i|$$

### 3. $L_\infty$Norm (Max Norm)
Simply the largest single component in the vector. 
$$\lVert x \rVert_\infty = \max |x_i|$$

```python
import numpy as np

x = np.array([1, 0, -2])

# L1 Norm = 1 + 0 + |-2| = 3
l1 = np.linalg.norm(x, ord=1)       

# L2 Norm = sqrt(1^2 + 0^2 + (-2)^2) = 2.236
l2 = np.linalg.norm(x, ord=2)       

# L_inf = max(|1|, |0|, |-2|) = 2
l_inf = np.linalg.norm(x, ord=np.inf) 
```

---

## 3. Deep Learning Failure Modes & Convexity

All true norms are **Convex functions**. In optimization, a convex loss function guarantees that gradient descent will find a single, global minimum point.

**The sparsity problem:** In ML, we often want our neural network to aggressively shut down useless features (setting weights to exactly$0.000$).
* **L2 Regularization** shrinks weights smoothly, but rarely hits exactly zero.
* **L1 Regularization** is visually shaped like a diamond. Because of its sharp corners, gradient descent naturally settles exactly on the axes, aggressively coercing weights to $0.000$. This induces mathematical sparsity.

### The $L_0$"Norm"
The$L_0$norm just counts the number of non-zero elements.
$$\|x\|_0 = |\{i : x_i \neq 0\}|$$
Strictly speaking,$L_0$ is **not a formal norm** because it fails scalar multiplication ($\|2x\|_0 = \|x\|_0$). It is also completely non-convex and non-differentiable (a piecewise step function). Attempting to use $L_0$in Deep Learning backpropagation will crash the optimizer because the gradient is either zero everywhere or infinitely undefined. We universally use L1 as a convex, differentiable approximation of L0 to achieve sparsity.

---

## 4. Inner Product Spaces

Inner product spaces generalize the dot product. While Norms give vectors *length*, Inner Products give vector spaces *angles*.

An inner product$\langle x, y \rangle$satisfies:
1. Non-negativity:$\langle x, x \rangle \ge 0$.
2. Linearity: $\langle \alpha x, y \rangle = \alpha \langle x, y \rangle$.
3. Symmetry: $\langle x, y \rangle = \langle y, x \rangle$.

### Angle Between Vectors
The most critical relationship in NLP (Cosine Similarity) is derived directly from the inner product:

$$\langle x, y \rangle = \|x\| \|y\| \cos \theta$$

By isolating $\cos \theta$, we can instantly measure how deeply "aligned" two sentence embeddings are in space, ignoring their magnitude.
