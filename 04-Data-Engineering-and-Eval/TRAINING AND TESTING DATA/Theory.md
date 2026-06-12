# Training & Testing Splits ✂️

## 1. Concept Overview: Generalization
The goal of Machine Learning is not to memorize data, but to **generalize** to new, unseen examples.

If we train our model on *all* the available data, we have no way of knowing if it has actually learned the underlying patterns or just memorized the specific answers (Overfitting).

### The Strategy
We split our dataset into two distinct sets:
1.  **Training Set (e.g., 80%):** Used to teach the model (calculate weights/biases). The model is allowed to see these answers.
2.  **Test Set (e.g., 20%):** Used to evaluate the model. The model *never* sees these answers during training. We use this to simulate "future" data.

---

## 2. The Code: `train_test_split`

We use Scikit-Learn's utility function to randomize and split the data.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Parameter Breakdown

| Parameter | Definition | Why use it? |
| :--- | :--- | :--- |
| **`X`** | Feature Matrix | The inputs (e.g., Mileage, Age). |
| **`y`** | Target Vector | The output (e.g., Price). |
| **`test_size`** | Float (0.0 to 1.0) | Defines the size of the test set (e.g., `0.2` = 20%). |
| **`random_state`** | Integer (Seed) | Ensures **Reproducibility**. Without this, the split changes every time you run the code, making debugging impossible. |

---

## 3. Visualizing the Split
When we split the data, we are randomly sampling points from the domain.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.uniform(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1.5, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(8, 4))
plt.scatter(X_train, y_train, color='blue', s=30, alpha=0.7, label=f'Training ({len(X_train)} pts)')
plt.scatter(X_test, y_test, color='red', s=60, marker='x', label=f'Testing ({len(X_test)} pts)')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Train/Test Split Visualization')
plt.legend()
plt.grid(True)
plt.show()
```

* 🔵 **Blue Points:** Training Data (Model learns from these).
* 🔴 **Red Points:** Testing Data (Model is tested on these).
