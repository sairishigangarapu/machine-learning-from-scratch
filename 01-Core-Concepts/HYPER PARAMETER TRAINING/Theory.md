# Hyperparameter Optimization Strategies 🎛️

## 1. The Challenge of Tuning
Machine Learning models are governed by **Hyperparameters**—settings that are not learned from data but must be defined *before* training (e.g., `learning_rate` in Gradient Descent, `k` in KNN).

Finding the optimal combination manually using nested loops is inefficient and computationally expensive. We use specialized algorithms to automate this search.

---

## 2. Strategy A: Grid Search (`GridSearchCV`)
Grid Search performs an **exhaustive search** over a specified parameter grid.

### ✅ When to use
* The search space is **small and finite**.
* You need to guarantee finding the absolute best combination within the grid.
* Computational resources are not a constraint.

| Feature | Description |
| :--- | :--- |
| **Search Method** | Brute-force checking of every single combination. |
| **Complexity** | $O(n \times m \times ...)$, grows exponentially with dimensions. |
| **Code** | `sklearn.model_selection.GridSearchCV` |

---

## 3. Strategy B: Randomized Search (`RandomizedSearchCV`)
Randomized Search samples a fixed number of parameter settings from specified probability distributions.

### ✅ When to use
* The search space is **large or continuous**.
* You want to find a "good enough" model quickly.
* Some parameters are more important than others (sparsity of the search space).

| Feature | Description |
| :--- | :--- |
| **Search Method** | Stochastic sampling from distributions (Uniform, LogUniform). |
| **Efficiency** | Can find optimal models in a fraction of the time compared to Grid Search. |
| **Code** | `sklearn.model_selection.RandomizedSearchCV` |

---

## 4. Hyperparameter Reference Guide 📖
## 5. Algorithm-Specific Parameter Grids 🎛️

Use these `param_dist` dictionaries when configuring your `RandomizedSearchCV`.

### 🤖 Logistic Regression
```python
param_dist = {
    'C': loguniform(1e-4, 1e4),   # Inverse regularization (Lower = Stronger Reg)
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga'], # 'saga' handles all penalties
    'max_iter': [100, 200, 500]
}
```

### 🧠 Support Vector Machine (SVM) 
```python
param_dist = {
    'C': uniform(0.1, 100),       # Penalty parameter of the error term
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],   # Kernel coefficient
    'degree': [2, 3, 4]           # Only for 'poly' kernel
}
```

### 🌳 Random Forest
```python
param_dist = {
    'n_estimators': randint(50, 300),    # Number of trees
    'max_depth': randint(3, 20),         # Pruning depth
    'min_samples_split': randint(2, 10), # Prevent overfitting
    'min_samples_leaf': randint(1, 5),   # Smooth predictions
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}
```

### 📈 K-Nearest Neighbors (KNN)
```python
param_dist = {
    'n_neighbors': randint(1, 30),
    'weights': ['uniform', 'distance'],  # 'distance' gives weight to closer neighbors
    'metric': ['euclidean', 'manhattan']
}
```

### 🚀 Gradient Boosting (XGBoost / GBM)
```python
param_dist = {
    'learning_rate': uniform(0.01, 0.3), # Step size shrinkage
    'n_estimators': randint(100, 1000),  # Boosting rounds
    'max_depth': randint(3, 10),         # Tree complexity
    'subsample': uniform(0.6, 0.4)       # Stochastic boosting
}
```
