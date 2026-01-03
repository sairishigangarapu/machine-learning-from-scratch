# Gradient Descent and Cost Function Optimization 📉

## 1. The Objective: Cost Minimization

To train a linear regression model, we must find the line of best fit for a given training dataset. We achieve this by minimizing the error between the predicted values and the actual values.

### The Cost Function (Mean Squared Error)
The metric used to evaluate the error is the **Mean Squared Error (MSE)**.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (mx_i + b) \right)^2
$$



* $n$: Number of data points.
* $y_i$: Actual value.
* $mx_i + b$: Predicted value (Hypothesis).

---

## 2. The Algorithm: Gradient Descent

Gradient descent is an iterative optimization algorithm used to find the minimum of a function.

### Algorithmic Mechanics


As we compute the cost with different values of $m$ (slope) and $b$ (intercept), we progress in a direction that reduces error. This continues until we reach the global minimum, providing the optimal $m$ and $b$ for plotting the graph.

### The Descent Step


To reach the minima, we calculate the slope of the cost function at the current point and move in the direction where the slope tends toward zero. This is done by taking the derivative at a specific point in the trajectory.

---

## 3. Mathematical Derivation

To update our parameters, we calculate the partial derivatives of the Cost Function with respect to the slope ($m$) and the intercept ($b$).

### Partial Derivative w.r.t $m$
$$
\frac{\partial}{\partial m} = \frac{2}{n} \sum_{i=1}^{n} -x_i \left( y_i - (mx_i + b) \right)
$$


### Partial Derivative w.r.t $b$
$$
\frac{\partial}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} -\left( y_i - (mx_i + b) \right)
$$


---

## 4. Hyperparameters

### Learning Rate ($\alpha$)
Once we have the direction (gradient), we need to decide the size of the step to take. This is determined by the **Learning Rate**.



* **Definition:** A user-defined hyperparameter that controls how much we adjust the weights with respect to the loss gradient.
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


---
**External Exercise:** [Codebasics Gradient Descent Exercise](https://github.com/codebasics/py/tree/master/ML/3_gradient_descent)
