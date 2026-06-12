# XGBoost & Gradient Boosting: The Tabular ML King

## 1. Why Gradient Boosting?

For **tabular data** (spreadsheets, databases, CSVs), gradient boosting consistently outperforms deep learning. XGBoost has won **most Kaggle tabular competitions** since 2015 and is the most used algorithm in industry for structured data.

> **StatQuest Rule of Thumb:** If your data fits in a spreadsheet, use gradient boosting. If it's images/text/audio, use deep learning.

---

## 2. Boosting: Learning from Mistakes

**Boosting** builds an ensemble of **weak learners** (typically shallow decision trees) **sequentially** — each new tree corrects the errors of the previous ones.

### The Key Insight
Instead of training one deep tree, train many tiny trees (depth 3–6). Each tree focuses on the examples the previous trees got wrong.

### Boosting vs Bagging

| Feature | Bagging (Random Forest) | Boosting (XGBoost) |
| :--- | :--- | :--- |
| Trees trained | **Parallel** (independent) | **Sequential** (dependent) |
| Error reduction | Reduces **variance** | Reduces **bias** (and variance) |
| Each tree trained on | Bootstrap sample | Residuals (errors) of previous trees |
| Overfitting risk | Lower (averages out) | Higher (can memorize errors) |
| Speed | Parallelizable | Sequential (slower to train) |

---

## 3. Gradient Boosting Algorithm

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(\mathbf{x}_i)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\hat{y}_i^{(t)}$ | Prediction for sample $i$ after adding tree $t$ | The current ensemble prediction — updated incrementally with each new tree |
| $\hat{y}_i^{(t-1)}$ | Prediction for sample $i$ after tree $t-1$ | The previous ensemble prediction before adding the new tree |
| $\eta$ | Learning rate (shrinkage, e.g., 0.1) | Scales down each tree's contribution — lower $\eta$ = more regularization, requires more trees |
| $f_t(\mathbf{x}_i)$ | Prediction of the $t$-th tree for sample $i$ | The weak learner's output — typically a shallow decision tree predicting residuals |
| $r_i = y_i - \hat{y}_i^{(t-1)}$ | Residual for sample $i$ | What the current ensemble gets wrong — the new tree learns to predict these residuals |

### Step by Step
1. Start with a simple prediction (mean for regression, log-odds for classification).
2. Compute residuals (how wrong we are).
3. Fit a small tree to predict those residuals.
4. Add the tree's prediction (scaled by learning rate) to the ensemble.
5. Repeat.

---

## 4. XGBoost: eXtreme Gradient Boosting

XGBoost (Chen & Guestrin, 2016) is an optimized implementation of gradient boosting with:

| Innovation | Benefit |
| :--- | :--- |
| **Regularized objective** | Adds L1/L2 penalties on tree weights to prevent overfitting |
| **Newton-Raphson updates** | Uses second-order gradients (Hessian) for faster convergence |
| **Sparsity-aware splitting** | Handles missing values natively |
| **Column subsampling** | Random feature selection per tree (like Random Forest) |
| **Parallel tree construction** | Parallelizes feature evaluation (not tree building) |
| **Cache-aware access** | Optimized for CPU cache efficiency |
| **Histogram-based splitting** | Approximate split finding for large datasets |

### XGBoost Objective

$$
\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

where $\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2$ penalizes tree complexity ($T$ = number of leaves, $w_j$ = leaf weights).

---

## 5. Key Hyperparameters

| Parameter | Typical Range | What It Controls |
| :--- | :--- | :--- |
| `n_estimators` | 100–1000 | Number of boosting rounds |
| `max_depth` | 3–10 | Tree depth (smaller = less overfitting) |
| `learning_rate` ($\eta$) | 0.01–0.3 | Step size (smaller needs more trees) |
| `subsample` | 0.6–1.0 | Row sampling per tree |
| `colsample_bytree` | 0.6–1.0 | Feature sampling per tree |
| `reg_alpha` (L1) | 0–1 | L1 regularization on leaf weights |
| `reg_lambda` (L2) | 1–10 | L2 regularization on leaf weights |
| `min_child_weight` | 1–10 | Minimum sum of instance weight in a leaf |

> **Golden rule:** `learning_rate` × `n_estimators` trade-off. Lower LR needs more trees but generalizes better.

---

## 6. Code Example

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
 random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
xgb_model = xgb.XGBClassifier(
 n_estimators=200, max_depth=5, learning_rate=0.1,
 subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# Random Forest baseline
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"XGBoost: {xgb_acc:.4f}")
print(f"Random Forest: {rf_acc:.4f}")
```

---

## 7. Feature Importance

XGBoost provides three types of feature importance:

| Type | What It Measures |
| :--- | :--- |
| `weight` | Number of times a feature is used for splitting |
| `gain` | Average improvement in accuracy from splits using that feature |
| `cover` | Average number of samples affected by splits using that feature |

```python
import pandas as pd
fi = pd.Series(xgb_model.feature_importances_)
fi.sort_values(ascending=False).head(10).plot(kind='bar', title='XGBoost Feature Importance')
```

---

## 8. XGBoost vs Other Algorithms

| Scenario | Best Choice |
| :--- | :--- |
| Tabular data < 10K rows | XGBoost or LightGBM |
| Tabular data > 1M rows | LightGBM (faster) or CatBoost |
| Categorical features many | CatBoost (native handling) |
| Images / Text / Audio | Deep Learning |
| Need interpretability | Decision Tree or Logistic Regression |
| Need maximum accuracy | XGBoost + hyperparameter tuning |

---

## 9. The Gradient Boosting Family (2026)

| Algorithm | Key Innovation |
| :--- | :--- |
| **XGBoost** | Regularized, Newton-Raphson, sparse-aware |
| **LightGBM** | Histogram-based, leaf-wise growth (faster, less memory) |
| **CatBoost** | Native categorical feature handling, ordered boosting |

> **StatQuest Tip:** Start with XGBoost. If speed matters, try LightGBM. If you have many categorical features, try CatBoost.

---

## 10. Advantages & Disadvantages

### Pros
* **State-of-the-art on tabular data** — beats deep learning on most structured datasets.
* Handles missing values natively.
* Built-in regularization (L1, L2).
* Feature importance built in.
* Works with mixed feature types (numeric + categorical).

### Cons
* **Black box** — harder to interpret than a single tree.
* Sequential training — can't parallelize across trees.
* Sensitive to **noisy data** and **outliers** (can memorize noise).
* Requires careful hyperparameter tuning.
* Not suitable for unstructured data (images, text).

---

**Previous:** [Random Forest](../RANDOM%20FOREST/Theory.md) | **Next:** [XGBoost Lab](xgboost_lab.py) | **Related:** [Bias-Variance Tradeoff](../../01-Core-Concepts/Bias-Variance-Tradeoff.md)
