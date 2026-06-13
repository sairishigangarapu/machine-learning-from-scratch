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

| Term | Definition | Significance |
|:---|:---|:---|
| $\hat{y}$ | Predicted class label ($+1$ or $-1$) | Output of the classifier; sign of the decision function determines the class |
| $\mathbf{w}$ | Weight vector (normal to hyperplane) | Orientation of the decision boundary; learned during training to maximize margin |
| $\mathbf{x}$ | Input feature vector | The data point being classified |
| $b$ | Bias term (intercept) | Shifts the decision boundary away from origin; prevents constraint that boundary must pass through origin |
| $\text{sign}(\cdot)$ | Sign function: returns $+1$ if argument $> 0$, $-1$ otherwise | Converts real-valued decision score into discrete class prediction |
| $\mathbf{w}^T\mathbf{x} + b$ | Decision function; signed distance from $\mathbf{x}$ to the hyperplane | Magnitude indicates confidence (distance from boundary), sign indicates side (which class) |

* $\mathbf{w}^T\mathbf{x} + b > 0 \Rightarrow \hat{y} = +1$
* $\mathbf{w}^T\mathbf{x} + b < 0 \Rightarrow \hat{y} = -1$
* $\mathbf{w}^T\mathbf{x} + b = 0$ is the **decision boundary**

### Margin
The **functional margin** of a point $(\mathbf{x}_i, y_i)$ is:

$$
\hat{\gamma}_i = y_i(\mathbf{w}^T\mathbf{x}_i + b)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\hat{\gamma}_i$ | Functional margin for point $i$ | A positive value means correct classification; scale depends on $||\mathbf{w}||$, so it's not a true distance |
| $y_i$ | True class label ($+1$ or $-1$) for point $i$ | Encodes which side of the boundary the point should be on |
| $\mathbf{x}_i$ | The $i$-th training example | Each data point contributes to the margin constraints |
| $\mathbf{w}^T\mathbf{x}_i + b$ | Unscaled score at point $\mathbf{x}_i$ | Combined with $y_i$, this product must be $\ge 1$ for correct classification with margin |

This is positive if classified correctly, negative if misclassified.

### Geometric Margin
The actual perpendicular distance from point $\mathbf{x}_i$ to the hyperplane:

$$
\gamma_i = \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{||\mathbf{w}||}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\gamma_i$ | Geometric margin for point $i$ | The actual Euclidean distance from $\mathbf{x}_i$ to the hyperplane; invariant to scaling of $\mathbf{w}$ and $b$ |
| $||\mathbf{w}||$ | Euclidean norm (magnitude) of weight vector | Normalizing by $||\mathbf{w}||$ removes the scale dependence; this is what SVMs maximize |
| $\frac{1}{||\mathbf{w}||}$ | Normalization factor | Converts the unscaled functional margin into a true distance; key insight: minimizing $||\mathbf{w}||$ maximizes margin |

The factor $\frac{1}{||\mathbf{w}||}$ converts functional margin to geometric distance.

---

## 3. Why Maximum Margin?

### Statistical Learning Theory
VC dimension theory shows that the generalization error is bounded by:

$$
R(\mathbf{w}) \le R_{\text{emp}}(\mathbf{w}) + O\left(\sqrt{\frac{d_{\text{VC}} \log(n)}{n}}\right)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $R(\mathbf{w})$ | True (generalization) error of classifier with weights $\mathbf{w}$ | The actual risk on unseen data; what we ultimately care about |
| $R_{\text{emp}}(\mathbf{w})$ | Empirical (training) error | Error measured on the training set; always an overestimate of true performance |
| $d_{\text{VC}}$ | VC dimension — a measure of model capacity/complexity | Larger VC dimension means the model can fit more complex patterns but may overfit |
| $n$ | Number of training samples | More data reduces the gap between training and generalization error |
| $O(\sqrt{\cdot})$ | Big-O complexity bound | Describes how quickly the generalization gap shrinks as $n$ increases |

where $d_{\text{VC}}$ is the VC dimension. Maximizing the margin **reduces the effective VC dimension**, providing tighter generalization bounds.

### Robustness Intuition
A larger margin means the classifier is less sensitive to small perturbations in the data. Points far from the boundary don't affect the decision — only the support vectors matter.

---

## 4. Maximum Margin Classifier Formulation

### Primal Formulation
$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad &\frac{1}{2}||\mathbf{w}||^2 \\
\text{s.t.} \quad &y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 \quad \forall i
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\frac{1}{2}||\mathbf{w}||^2$ | Objective — half squared norm of weight vector | Minimizing this maximizes the margin $\frac{2}{||\mathbf{w}||}$; convex quadratic ensures unique global optimum |
| $\mathbf{w}$ | Weight vector (normal to hyperplane) | Learned to maximize margin; its norm $||\mathbf{w}||$ is minimized in the objective |
| $b$ | Bias term (intercept) | Shifts the decision boundary; ensures flexibility when data is not centered at origin |
| $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$ | Margin constraint for point $i$ | Enforces correct classification with functional margin $\ge 1$; active ($=1$) for support vectors |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $y_i(\mathbf{w}^T\mathbf{x}_i + b)$ | Functional margin of point $i$ | When $= 1$, the point lies exactly on the margin boundary — it is a support vector |
| $y_i$, $\mathbf{x}_i$, $\mathbf{w}$, $b$ | As defined above | Only points satisfying this equality determine the optimal hyperplane; all other points can be removed without changing the solution |

These are the only points that affect the optimal hyperplane. Removing any non-support-vector point does not change the solution.

**ML Connection:** This sparsity makes SVMs memory-efficient. Only support vectors need to be stored for prediction.

---

## 6. Geometric Interpretation

The margin is the region between the two parallel hyperplanes:

$$
\begin{aligned}
\mathbf{w}^T\mathbf{x} + b &= +1 \quad \text{(positive boundary)} \\
\mathbf{w}^T\mathbf{x} + b &= -1 \quad \text{(negative boundary)}
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathbf{w}^T\mathbf{x} + b = +1$ | Positive margin hyperplane | Boundary for class $+1$; points on this plane have functional margin $= 1$ |
| $\mathbf{w}^T\mathbf{x} + b = -1$ | Negative margin hyperplane | Boundary for class $-1$; points on this plane have functional margin $= -1$ |
| $\frac{2}{||\mathbf{w}||}$ | Margin width | Distance between the two margin boundaries; maximizing this is equivalent to minimizing $||\mathbf{w}||^2$ |

The distance between these boundaries is $\frac{2}{||\mathbf{w}||}$. Maximizing this distance is equivalent to minimizing $||\mathbf{w}||^2$.

> **Check your intuition:** If you multiply all weights and bias by 2, how does the margin change? *(Answer: The decision boundary doesn't change (same hyperplane), but the functional margin doubles while the geometric margin stays the same since $||\mathbf{w}||$ also doubles. This is why SVMs use geometric margin.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 53: Joint Probability Distributions](Lecture%2053%20Joint%20Probability%20Distributions.md) — Probability foundations for classification
- **Next:** [Lecture 55: Maximum Margin Classification](Lecture%2055%20Maximum%20Margin%20Classification.md) — Detailed optimization of SVM objective
- **Related:** [Lecture 55: Maximum Margin Classification](Lecture%2055%20Maximum%20Margin%20Classification.md) — Formal optimization of the maximum margin concept
- **Related:** [Lecture 56: Duality and the Dual Problem](Lecture%2056%20Duality%20and%20the%20Dual%20Problem.md) — Dual formulation enables kernel trick
- **Related:** [Lecture 26: Classification Metrics](Lecture%2026%20Classification%20Metrics.md) — Evaluating SVM performance
