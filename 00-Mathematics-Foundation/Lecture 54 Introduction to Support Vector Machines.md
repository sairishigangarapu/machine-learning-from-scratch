## Introduction to Support Vector Machines

*Essential Mathematics for ML — Structured Notes*

---

## 1. What is an SVM?

### Motivation and Intuition
Given a dataset of red and blue points in the plane, infinitely many lines can separate them. Which line is *best*? An SVM finds the separating hyperplane with the **maximum margin** — the largest possible distance to the nearest data points from either class. This maximization of margin leads to better generalization.

### The Core Idea
An SVM finds the hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ that maximizes the distance to the closest data point (the *support vectors*).

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=40, centers=2, random_state=6)
svm = SVC(kernel='linear', C=1000)
svm.fit(X, y)

print(f"Support vectors: {svm.support_vectors_.shape[0]}")  # 3
print(f"Coefficients: {svm.coef_}")
print(f"Intercept: {svm.intercept_}")
```

---

## 2. Linear Classifiers Recap

### Decision Function
A linear classifier assigns:

$$
\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)
$$

* $\mathbf{w}^T\mathbf{x} + b > 0 \Rightarrow \hat{y} = +1$
* $\mathbf{w}^T\mathbf{x} + b < 0 \Rightarrow \hat{y} = -1$
* $\mathbf{w}^T\mathbf{x} + b = 0$ is the **decision boundary**

### Margin
The **functional margin** of a point $(\mathbf{x}_i, y_i)$ is:

$$
\hat{\gamma}_i = y_i(\mathbf{w}^T\mathbf{x}_i + b)
$$

This is positive if classified correctly, negative if misclassified.

### Geometric Margin
The actual perpendicular distance from point $\mathbf{x}_i$ to the hyperplane:

$$
\gamma_i = \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{||\mathbf{w}||}
$$

The factor $\frac{1}{||\mathbf{w}||}$ converts functional margin to geometric distance.

---

## 3. Why Maximum Margin?

### Statistical Learning Theory
VC dimension theory shows that the generalization error is bounded by:

$$
R(\mathbf{w}) \le R_{\text{emp}}(\mathbf{w}) + O\left(\sqrt{\frac{d_{\text{VC}} \log(n)}{n}}\right)
$$

where $d_{\text{VC}}$ is the VC dimension. Maximizing the margin **reduces the effective VC dimension**, providing tighter generalization bounds.

### Robustness Intuition
A larger margin means the classifier is less sensitive to small perturbations in the data. Points far from the boundary don't affect the decision — only the support vectors matter.

---

## 4. Maximum Margin Classifier Formulation

### Primal Formulation
$$
\min_{\mathbf{w}, b} \quad \frac{1}{2}||\mathbf{w}||^2
$$
$$
\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 \quad \forall i
$$

**Key Insight:** Minimizing $||\mathbf{w}||^2$ is equivalent to maximizing the margin $\frac{2}{||\mathbf{w}||}$.

### Why $||\mathbf{w}||^2$ and Not $||\mathbf{w}||$?
The margin is $\frac{2}{||\mathbf{w}||}$. Minimizing $||\mathbf{w}||$ is equivalent to minimizing $||\mathbf{w}||^2$ (both are monotonic for $||\mathbf{w}|| > 0$), but $||\mathbf{w}||^2$ is differentiable everywhere and easier to optimize.

```python
import numpy as np
from scipy.optimize import minimize

def svm_objective(w):
    return 0.5 * np.sum(w[:-1]**2)

def margin_constraint(w, X, y, i):
    return y[i] * (np.dot(w[:-1], X[i]) + w[-1]) - 1

X, y = make_blobs(n_samples=40, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)

w_init = np.zeros(X.shape[1] + 1)
constraints = [{'type': 'ineq', 'fun': margin_constraint, 'args': (X, y, i)} 
               for i in range(len(X))]
result = minimize(svm_objective, w_init, constraints=constraints)
print(f"Optimal weights: {result.x[:2]}")
print(f"Optimal bias: {result.x[2]}")
```

---

## 5. Support Vectors

**Support vectors** are the data points that lie exactly on the margin boundaries. They are the points for which:

$$
y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1
$$

These are the only points that affect the optimal hyperplane. Removing any non-support-vector point does not change the solution.

**ML Connection:** This sparsity makes SVMs memory-efficient. Only support vectors need to be stored for prediction.

---

## 6. Geometric Interpretation

The margin is the region between the two parallel hyperplanes:

$$
\mathbf{w}^T\mathbf{x} + b = +1 \quad \text{(positive boundary)}
$$
$$
\mathbf{w}^T\mathbf{x} + b = -1 \quad \text{(negative boundary)}
$$

The distance between these boundaries is $\frac{2}{||\mathbf{w}||}$. Maximizing this distance is equivalent to minimizing $||\mathbf{w}||^2$.

> **Check your intuition:** If you multiply all weights and bias by 2, how does the margin change? *(Answer: The decision boundary doesn't change (same hyperplane), but the functional margin doubles while the geometric margin stays the same since $||\mathbf{w}||$ also doubles. This is why SVMs use geometric margin.)*
