# Gradient Descent and Cost Function Optimization

## 1. The Objective: Cost Minimization

To train a linear regression model, we must find the line of best fit for a given training dataset. We achieve this by minimizing the error between the predicted values and the actual values.

### The Cost Function (Mean Squared Error)
The metric used to evaluate the error is the **Mean Squared Error (MSE)**.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (mx_i + b) \right)^2
$$

| Variable | Definition |
| :--- | :--- |
| $n$ | Number of data points. |
| $y_i$ | Actual (ground-truth) value for the $i$-th sample. |
| $mx_i + b$ | Predicted value (Hypothesis) for the $i$-th sample. |
| $(y_i - (mx_i + b))^2$ | Squared residual — penalizes large errors disproportionately. |

---

## 2. The Algorithm: Gradient Descent

Gradient descent is an iterative optimization algorithm used to find the minimum of a function.

### Algorithmic Mechanics

We can visualize the cost function $J(m, b)$ as a bowl-shaped surface (a convex surface for MSE). Each point on this surface represents a specific pair of $(m, b)$ values with a corresponding cost. Gradient descent starts at an arbitrary point on this surface and iteratively moves toward the bottom of the bowl.

As we compute the cost with different values of $m$ (slope) and $b$ (intercept), we progress in a direction that reduces error. This continues until we reach the global minimum, providing the optimal $m$ and $b$ for plotting the graph.

### The Descent Step

The gradient $\nabla J$ points in the direction of steepest **ascent**. By subtracting the gradient (multiplied by the learning rate), we move in the direction of steepest **descent** — directly toward the minimum.

To reach the minima, we calculate the slope of the cost function at the current point and move in the direction where the slope tends toward zero. This is done by taking the derivative at a specific point in the trajectory.

---

## 3. Mathematical Derivation

To update our parameters, we calculate the partial derivatives of the Cost Function with respect to the slope ($m$) and the intercept ($b$).

### Partial Derivative w.r.t $m$
$$
\frac{\partial J}{\partial m} = \frac{2}{n} \sum_{i=1}^{n} -x_i \left( y_i - (mx_i + b) \right)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial J}{\partial m}$ | Partial derivative of cost w.r.t. slope | How much the cost changes when we nudge the slope $m$ — this IS the gradient |
| $x_i$ | Input feature for sample $i$ | The variable we're predicting from — scaled by the residual |
| $y_i - (mx_i + b)$ | Residual (error) for sample $i$ | How far the prediction is from the true value |
| $-x_i (y_i - (mx_i + b))$ | Weighted residual | The error, weighted by the input — steeper slopes get larger corrections |
| $\frac{2}{n} \sum_{i=1}^{n}$ | Average over all samples | We average the gradients to get a stable estimate of the true direction |

This gradient tells us how the cost changes as we nudge the slope $m$. A negative gradient means increasing $m$ will decrease the cost.

### Partial Derivative w.r.t $b$
$$
\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} -\left( y_i - (mx_i + b) \right)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial J}{\partial b}$ | Partial derivative of cost w.r.t. intercept | How much the cost changes when we nudge the intercept $b$ |
| $y_i - (mx_i + b)$ | Residual for sample $i$ | Same error signal as above, but without the $x_i$ scaling |
| $-\left( y_i - (mx_i + b) \right)$ | Negative residual | If the prediction is too high (positive residual), we decrease $b$; if too low, we increase it |

This gradient tells us how the cost changes as we nudge the intercept $b$.

---

## 4. Hyperparameters

### Learning Rate ($\alpha$)
Once we have the direction (gradient), we need to decide the size of the step to take. This is determined by the **Learning Rate**.

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial \theta}$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\theta_{\text{old}}$ | Current parameter value | Where we are now in parameter space |
| $\theta_{\text{new}}$ | Updated parameter value | Where we'll be after this step |
| $\alpha$ | Learning rate (hyperparameter) | Controls step size — too small = slow convergence, too large = overshooting/divergence |
| $\frac{\partial J}{\partial \theta}$ | Gradient of cost w.r.t. parameter | The direction of steepest ascent — we move opposite to it (descent) |
| $\alpha \cdot \frac{\partial J}{\partial \theta}$ | Step vector | The actual change applied to the parameter |

* **Tuning:** It must be fine-tuned for accuracy; too small leads to slow convergence, while too large can cause divergence.

### Stopping Condition
Knowing when to stop is achieved by monitoring the iterations and the cost value. We stop when the cost reduction plateaus or after a fixed number of iterations.

---

## 5. Statistical Metrics

### Correlation Coefficient ($r$)
The correlation coefficient quantifies the strength and direction of the linear relationship between two variables.
* **Range:** -1 (Perfect Negative) to +1 (Perfect Positive).
* **Significance:** A high absolute value suggests a linear regression line is a good fit for the data.

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }{ \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \cdot \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $r$ | Pearson correlation coefficient | Measures linear relationship strength: $r = 1$ (perfect positive), $r = -1$ (perfect negative), $r = 0$ (no linear relationship) |
| $x_i, y_i$ | Individual data points | Paired observations from the dataset |
| $\bar{x}, \bar{y}$ | Sample means | The center of each variable's distribution |
| $(x_i - \bar{x})(y_i - \bar{y})$ | Cross-product of deviations | Positive when both deviate in same direction, negative when opposite — captures co-movement |
| $\sum_{i=1}^{n}$ | Sum over all $n$ samples | Aggregates the signal across the entire dataset |
| $\sqrt{\sum (x_i - \bar{x})^2}$ | Standard deviation of $x$ (up to $\sqrt{n}$) | Normalizes by the spread of $x$ — ensures $r$ is dimensionless |
| $\sqrt{\sum (y_i - \bar{y})^2}$ | Standard deviation of $y$ (up to $\sqrt{n}$) | Same normalization for $y$ |


---
**External Exercise:** [Codebasics Gradient Descent Exercise](https://github.com/codebasics/py/tree/master/ML/3_gradient_descent)
