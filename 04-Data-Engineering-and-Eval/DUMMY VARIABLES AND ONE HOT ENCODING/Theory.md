# Categorical Data Encoding & The Dummy Variable Trap 🔢

## 1. The Problem: Text vs. Math
Machine Learning algorithms rely on mathematical equations (e.g., $y = mx + c$). They cannot process text data like "New York" or "Male." We must convert these categorical features into numerical representations.

---

## 2. Types of Categorical Variables

### A. Nominal Variables (No Order)
Categories with no inherent ranking.
* **Examples:** Town Names (Monroe, West Windsor), Colors (Red, Blue), Gender.
* **Strategy:** Use **One-Hot Encoding**. 

### B. Ordinal Variables (Ordered)
Categories with a clear rank or hierarchy.
* **Examples:** Education (High School < Bachelors < Masters), Satisfaction (Low < Medium < High).
* **Strategy:** Use **Label Encoding** (assign 1, 2, 3...) to preserve the order.

---

## 3. One-Hot Encoding & The Dummy Variable Trap

One-Hot Encoding creates a new binary column for each category.

| Town | $\rightarrow$ | Monroe | West Windsor | Robbinsville |
| :--- | :--- | :--- | :--- | :--- |
| Monroe | $\rightarrow$ | 1 | 0 | 0 |
| West Windsor | $\rightarrow$ | 0 | 1 | 0 |
| Robbinsville | $\rightarrow$ | 0 | 0 | 1 |

### 🚧 The Dummy Variable Trap
The "Trap" occurs when variables are highly correlated (**Multicollinearity**). If we include **all** dummy variables, one variable can be predicted from the others:

$$
\text{Monroe} = 1 - (\text{West Windsor} + \text{Robbinsville})
$$



This linear dependency breaks the mathematics of Linear Regression (Matrix Inversion becomes unstable).

### ✅ The Solution: Drop One Column
We must drop one dummy variable to act as the **reference baseline**.
* If you have $N$ categories, you need $N-1$ dummy columns.
* *Example:* Drop "West Windsor". If Monroe=0 and Robbinsville=0, the model implicitly knows it must be West Windsor.

---

**External Exercise:** [Codebasics One Hot Encoding Lab](https://github.com/codebasics/py/blob/master/ML/5_one_hot_encoding/one_hot_encoding.ipynb)
