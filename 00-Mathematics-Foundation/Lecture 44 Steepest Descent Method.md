## Steepest Descent Method

*Essential Mathematics for ML — Structured Notes*

---

## 1. The Workhorse of Deep Learning

### Motivation and Intuition
The **Steepest Descent Method** (also called Gradient Descent) is the single most important algorithm in machine learning. Every neural network, every language model, every image classifier is trained by some variant of steepest descent. The method follows the direction of negative gradient — the direction of steepest decrease — to iteratively minimize a function.

### The Update Rule

$$
\theta_{k+1} = \theta_k - \alpha_k \nabla_\theta \mathcal{L}(\theta_k)
$$

where:
* $\theta$: Model parameters
* $\alpha_k$: Learning rate at step $k$
* $\nabla_\theta \mathcal{L}$: Gradient of the loss

---

## 2. Batch vs Stochastic vs Mini-Batch

### Batch Gradient Descent
Computes the gradient over the **entire dataset**:

$$
\nabla_\theta \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(\theta; \mathbf{x}_i, y_i)
$$

* **Pros:** Exact gradient, smooth convergence.
* **Cons:** Extremely slow for large datasets ($N = 10^6$ → sum over 1M examples per step).

### Stochastic Gradient Descent (SGD)
Computes the gradient over a **single random sample**:

$$
\nabla_\theta \mathcal{L} \approx \nabla_\theta \ell(\theta; \mathbf{x}_i, y_i)
$$

* **Pros:** Extremely fast per iteration, introduces noise that helps escape local minima.
* **Cons:** Noisy updates, high variance, may never fully converge.

### Mini-Batch SGD (The Standard)
Computes the gradient over a **mini-batch** of $B$ samples:

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{B} \sum_{i \in \text{batch}} \nabla_\theta \ell(\theta; \mathbf{x}_i, y_i)
$$

* **Compromise:** $B = 32$ to $512$ is typical. GPU-friendly, low variance, fast.

```python
import numpy as np

def mini_batch_sgd(X, y, loss_grad, theta, lr=0.01, batch_size=32, epochs=100):
    """Mini-batch gradient descent."""
    N = X.shape[0]
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            grad = loss_grad(X_batch, y_batch, theta)
            theta -= lr * grad
    
    return theta
```

---

## 3. Learning Rate Strategies

### The Learning Rate Problem
* Too large → divergence, oscillation
* Too small → agonizingly slow convergence
* Just right → fast, stable convergence

### Constant Learning Rate
The simplest approach: $\alpha_k = \alpha$ for all $k$.

### Learning Rate Decay
Gradually reduce the learning rate:

| Schedule | Formula |
|:---|:---|
| **Step decay** | $\alpha_k = \alpha_0 \cdot \gamma^{\lfloor k/s \rfloor}$ |
| **Exponential decay** | $\alpha_k = \alpha_0 \cdot e^{-\lambda k}$ |
| **1/t decay** | $\alpha_k = \alpha_0 / (1 + \lambda k)$ |
| **Cosine annealing** | $\alpha_k = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\pi k / T))$ |

**Deep Learning Connection:** Cosine annealing is the default in most modern training pipelines. It starts with a high learning rate, smoothly decreases to near zero, and can be combined with warm restarts for better exploration.

```python
import numpy as np

def learning_rate_schedule(step, initial_lr=0.1, schedule='cosine', **kwargs):
    if schedule == 'constant':
        return initial_lr
    elif schedule == 'step':
        gamma = kwargs.get('gamma', 0.1)
        step_size = kwargs.get('step_size', 30)
        return initial_lr * gamma ** (step // step_size)
    elif schedule == 'exponential':
        decay = kwargs.get('decay', 0.01)
        return initial_lr * np.exp(-decay * step)
    elif schedule == 'cosine':
        T_max = kwargs.get('T_max', 1000)
        eta_min = kwargs.get('eta_min', 1e-6)
        return eta_min + 0.5 * (initial_lr - eta_min) * (1 + np.cos(np.pi * step / T_max))
```

---

## 4. Adaptive Learning Rate Methods

### The Problem
Not all parameters need the same learning rate. Features that appear frequently should have smaller updates; rare features should have larger updates.

### AdaGrad
Accumulates squared gradients and divides by them:

$$
\theta_{k+1} = \theta_k - \frac{\alpha}{\sqrt{\sum_{i=1}^k g_i^2 + \epsilon}} \odot g_k
$$

* **Pros:** Adapts per-parameter, good for sparse data.
* **Cons:** Learning rate monotonically decreases → eventually stops learning.

### RMSprop
Fixes AdaGrad's decay by using an exponential moving average:

$$
v_k = \beta v_{k-1} + (1-\beta) g_k^2
$$
$$
\theta_{k+1} = \theta_k - \frac{\alpha}{\sqrt{v_k + \epsilon}} \odot g_k
$$

### Adam (Adaptive Moment Estimation)
Combines momentum (first moment) with RMSprop (second moment):

$$
m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k \quad \text{(momentum)}
$$
$$
v_k = \beta_2 v_{k-1} + (1-\beta_2) g_k^2 \quad \text{(adaptive LR)}
$$
$$
\hat{m}_k = \frac{m_k}{1 - \beta_1^k}, \quad \hat{v}_k = \frac{v_k}{1 - \beta_2^k} \quad \text{(bias correction)}
$$
$$
\theta_{k+1} = \theta_k - \frac{\alpha}{\sqrt{\hat{v}_k} + \epsilon} \odot \hat{m}_k
$$

**Default hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\alpha = 0.001$.

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params
```

---

## 5. Convergence Theory

### For Convex Functions with Lipschitz Gradient

| Method | Convergence Rate |
|:---|:---|
| **Batch GD** (constant step) | $O(1/k)$ |
| **SGD** (decaying step) | $O(1/\sqrt{k})$ |
| **Momentum / Nesterov** | $O(1/k^2)$ |
| **Strongly convex + GD** | Linear: $O(c^k)$, $c < 1$ |

### The Noise Trade-off
* SGD's noise is **harmful** near the minimum (oscillation).
* SGD's noise is **helpful** far from the minimum (escaping saddle points and poor local minima).

This is why learning rate schedules start high (exploit noise for exploration) and decay low (reduce noise for precise convergence).

---

## 6. Practical Guidelines

| Guideline | Recommendation |
|:---|:---|
| **Default optimizer** | Adam ($\alpha = 0.001$) |
| **Batch size** | 32–512 (power of 2 for GPU efficiency) |
| **Learning rate** | Start with $10^{-3}$, reduce if loss oscillates |
| **Schedule** | Cosine annealing with warm restarts |
| **Gradient clipping** | Clip norm at 1.0 for RNNs/Transformers |
| **Weight decay** | $\lambda = 10^{-4}$ to $10^{-2}$ |

```python
# PyTorch training loop — the industry standard
import torch
import torch.optim as optim

model = MyModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
```

---

## 7. Summary

| Variant | Key Feature | When to Use |
|:---|:---|:---|
| **Batch GD** | Exact gradient | Small datasets |
| **SGD** | Noisy, fast | Online learning |
| **Mini-batch SGD** | Balanced | Default for deep learning |
| **Momentum** | Smooths oscillations | Ill-conditioned problems |
| **Adam** | Adaptive per-parameter LR | Almost everything |
| **AdamW** | Decoupled weight decay | Transformers, modern DL |

> **Check your intuition:** Why does SGD with a fixed learning rate never truly converge to the exact minimum? *(Answer: The stochastic gradient is noisy — even at the minimum, the gradient estimate from a mini-batch is not zero. The optimizer oscillates around the minimum with variance proportional to $\alpha^2 / B$. Decaying the learning rate reduces this oscillation.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 43: Constrained Optimization-II](Lecture%2043%20Constrained%20Optimization-II.md) — classifies optimization methods and introduces gradient descent as the primary first-order approach
- **Next:** [Lecture 45: Newton's and Penalty Function Method](Lecture%2045%20Newton%27s%20and%20Penalty%20Function%20Method.md) — extends to second-order methods and penalty-based approaches for constrained problems
- **Related:** [Lecture 35: Chain Rule](Lecture%2035%20Chain%20Rule.md) — backpropagation computes the gradients that steepest descent uses via the chain rule
- **Related:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — the theoretical framework that steepest descent algorithms implement
- **Related:** [Lecture 43: Constrained Optimization-II](Lecture%2043%20Constrained%20Optimization-II.md) — the optimization algorithm classification that places steepest descent in context
