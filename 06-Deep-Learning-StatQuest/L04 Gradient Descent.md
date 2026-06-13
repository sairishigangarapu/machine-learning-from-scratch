## L04 Gradient Descent

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What Gradient Descent Is

### Motivation and Intuition

After forward propagation, we have a prediction $\hat{y}$ and a loss $L$ that measures how wrong the prediction is. The question is: how do we adjust the weights and biases to make the loss smaller?

Gradient descent answers this question. It is an iterative optimization algorithm that finds the minimum of a function by taking small steps in the direction of steepest descent. Imagine standing on a hillside in thick fog. You cannot see the valley below, but you can feel the slope beneath your feet. You take a step in the steepest downhill direction. Then you feel the slope again and take another step. Repeat until you reach the bottom. That is gradient descent.

### The Core Idea

The gradient of the loss function tells us the direction of steepest **ascent** — the way that increases the loss the fastest. To decrease the loss, we move in the **opposite** direction (steepest descent). The size of each step is controlled by the **learning rate**.

---

## 2. The Loss Function

### Mean Squared Error (MSE)

For regression problems, a common loss function is the Mean Squared Error:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $L$ | Loss (or cost) | The total error we want to minimize — lower is better |
| $n$ | Number of training samples | We average over all samples to get a stable error estimate |
| $y_i$ | True value for sample $i$ | The ground truth we are trying to predict |
| $\hat{y}_i$ | Predicted value for sample $i$ | The network's output — depends on all weights and biases |
| $(y_i - \hat{y}_i)^2$ | Squared residual | Penalizes large errors much more than small ones (quadratic penalty) |

### Why Squared Error?

Squaring the residual does two things:
1. Removes the sign (positive and negative errors both increase the loss).
2. Disproportionately punishes large errors — a prediction that is off by 2 contributes 4x the loss of one that is off by 1.

---

## 3. The Gradient Descent Update Rule

### Parameter Update

Each parameter $\theta$ (which could be any weight $w$ or bias $b$) is updated as follows:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial \theta}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\theta_{\text{old}}$ | Current parameter value | Where we are now in parameter space |
| $\theta_{\text{new}}$ | Updated parameter value | The new value after this gradient descent step |
| $\alpha$ | Learning rate (hyperparameter) | Controls step size — too large and we overshoot the minimum, too small and convergence is painfully slow |
| $\frac{\partial L}{\partial \theta}$ | Partial derivative of loss w.r.t. the parameter | The gradient — tells us the direction and magnitude of steepest ascent at the current point |
| $\alpha \cdot \frac{\partial L}{\partial \theta}$ | The step | How much we actually change the parameter |

### The Minus Sign

We **subtract** the gradient because the gradient points uphill. Subtracting moves us downhill. To remember: gradient **descent** = move opposite to the gradient.

### Applying to Network Parameters

For a network with weights $w_1, w_2, \dots$ and biases $b_1, b_2, \dots$:

$$
w_1 \leftarrow w_1 - \alpha \cdot \frac{\partial L}{\partial w_1}
$$

$$
b_1 \leftarrow b_1 - \alpha \cdot \frac{\partial L}{\partial b_1}
$$

Every parameter gets its own update using its own gradient.

---

## 4. Finding the Minimum

### Convex vs. Non-Convex Surfaces

| Surface Type | Shape | Number of Minima | Typical Problem |
| :--- | :--- | :--- | :--- |
| **Convex** | Bowl-shaped | One global minimum | Linear regression |
| **Non-Convex** | Bumpy terrain | Many local minima + one global minimum | Neural networks |

For neural networks, the loss surface is non-convex — it has hills, valleys, and plateaus. Gradient descent can get stuck in a local minimum (a valley that is not the deepest point), but in practice, this rarely prevents good performance for large networks.

### Visualizing the Descent

Imagine a 2D slice of the loss surface. The x-axis is a single weight $w$, the y-axis is the loss $L$:

- At a random starting point, compute the slope (derivative) of the loss.
- If the slope is positive (loss increases as $w$ increases), subtract a fraction of the slope from $w$ to move left.
- If the slope is negative (loss decreases as $w$ increases), subtracting a negative means adding — so $w$ moves right.
- Repeat until the slope is close to zero.

---

## 5. The Learning Rate

### Step Size Matters

The learning rate $\alpha$ is the most important hyperparameter in gradient descent:

| Learning Rate | Behavior | Outcome |
| :--- | :--- | :--- |
| Too small | Tiny steps | Slow convergence — need many iterations, risk of getting stuck |
| Just right | Steady steps | Efficient convergence to the minimum |
| Too large | Huge steps | Overshoot the minimum, may diverge completely |

### Practical Tips

- Start with $\alpha = 0.01$ or $0.001$ and adjust based on the loss curve.
- If the loss oscillates or explodes, reduce the learning rate.
- If the loss decreases very slowly, increase the learning rate.
- Use **learning rate schedules** that reduce $\alpha$ over time for fine-tuning.

---

## 6. Gradient Descent Variants

| Algorithm | Full Name | How It Uses Data |
| :--- | :--- | :--- |
| **Batch GD** | Batch Gradient Descent | Uses all $n$ samples to compute the gradient — accurate but slow for large datasets |
| **Stochastic GD (SGD)** | Stochastic Gradient Descent | Uses one random sample per update — fast but noisy |
| **Mini-Batch GD** | Mini-Batch Gradient Descent | Uses a small random subset (e.g., 32 samples) — best of both worlds |

Neural networks almost always use mini-batch gradient descent. The noise from random batches actually helps escape local minima, and the computational efficiency makes training on large datasets feasible.

---

> **Check your intuition:** If the gradient $\frac{\partial L}{\partial w} = 0.5$ and the learning rate $\alpha = 0.1$, by how much does $w$ change in one update step? Which direction?

---

## Prerequisites and Further Reading

- **Previous:** L03 The Chain Rule (how we compute the gradients needed for gradient descent)
- **Next:** L05 Backpropagation Main Ideas (putting chain rule + gradient descent together)
- **Related:** Linear Regression Gradient Descent (simpler case with a convex surface)
