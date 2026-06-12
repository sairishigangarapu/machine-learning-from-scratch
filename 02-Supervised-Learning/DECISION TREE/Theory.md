# Decision Tree: The "White Box" Model

## 1. Concept Overview
A **Decision Tree** is a supervised learning algorithm that makes predictions by asking a sequence of "Yes/No" questions. It splits the data into smaller, more homogeneous subsets until it reaches a final conclusion.

Unlike "Black Box" models (like Neural Networks), Decision Trees are easy to interpret and visualize.

### Key Terminology
| Term | Definition |
| :--- | :--- |
| **Root Node** | The starting point containing the entire dataset. |
| **Decision Node** | An internal node that splits the data based on a specific feature condition. |
| **Leaf Node** | The terminal node that gives the final prediction (Class or Value). |
| **Purity** | A measure of how "mixed" the data is in a specific node. |

---

## 2. Choosing the Split: Gini vs. Entropy

The algorithm must decide **which feature** to split on at every step. It chooses the feature that results in the highest "Information Gain" (or creates the most "Pure" child nodes).

### A. Gini Impurity (CART Default)
Used by `scikit-learn`'s Classification And Regression Tree (CART) algorithm. It measures the probability of misclassifying a randomly chosen element.

$$
Gini = 1 - \sum_{i=1}^{n} (p_i)^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Gini$ | Gini impurity score | Ranges from 0 (pure) to 0.5 (maximally impure for binary) — lower is better |
| $p_i$ | Proportion of samples belonging to class $i$ | The fraction of data points in this node that are class $i$ |
| $(p_i)^2$ | Squared proportion | Penalizes large classes more — a node dominated by one class has low Gini |
| $1 - \sum$ | Complement of sum of squares | 1 minus the probability of correct classification = probability of misclassification |

* **Gini = 0:** Pure node (all samples belong to one class).
* **Gini = 0.5:** Maximum impurity (50/50 split in binary classification).

### B. Entropy (Information Gain)
A measure of disorder derived from thermodynamics/information theory.

$$
Entropy = - \sum_{i=1}^{n} p_i \log_2(p_i)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Entropy$ | Shannon entropy of the node | Ranges from 0 (pure) to $\log_2 K$ (maximally impure) — lower is better |
| $p_i$ | Proportion of samples in class $i$ | Same as Gini — the fraction of the node belonging to class $i$ |
| $-\sum$ | Negative sum | Ensures entropy is non-negative (since $\log_2(p_i) \leq 0$ for $p_i \leq 1$) |
| $\log_2$ | Base-2 logarithm | Measures information in bits — entropy is the average bits needed to encode a class label |

$$
\text{Info Gain} = \text{Entropy(Parent)} - \text{Weighted Avg Entropy(Children)}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{Entropy(Parent)}$ | Entropy before the split | How mixed the node was before splitting |
| $\text{Weighted Avg Entropy(Children)}$ | Entropy after the split, weighted by child sizes | How mixed the children are — weighted by how many points went to each child |
| $\text{Info Gain}$ | Reduction in entropy | Higher = more informative split — the algorithm chooses the feature with maximum Info Gain |

---

## 3. Advantages & Disadvantages

### Pros
* **Interpretability:** You can draw the tree and explain exactly *why* a decision was made.
* **No Scaling:** Requires no normalization or scaling of data.
* **Versatile:** Handles both Numerical and Categorical data.

### Cons
* **Overfitting:** Trees tend to grow very deep and memorize the noise (High Variance).
* **Instability:** Small changes in data can result in a completely different tree structure.
* **Greedy Behavior:** Each split is locally optimal — no guarantee of globally optimal tree.

---

## 4. Pruning: Controlling Overfitting

**Pre-pruning** (early stopping) limits tree growth during training:
* `max_depth`: Maximum depth of the tree (e.g., 5–10).
* `min_samples_split`: Minimum samples required to split a node (e.g., 10–20).
* `min_samples_leaf`: Minimum samples at a leaf node (e.g., 5–10).

**Post-pruning** grows the full tree first, then removes branches that don't improve validation performance (e.g., `ccp_alpha` in sklearn's Cost-Complexity Pruning).

---

## 5. Code Example

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
 iris.data, iris.target, test_size=0.2, random_state=42
)

# Unpruned tree (will overfit)
dt_unpruned = DecisionTreeClassifier(random_state=42)
dt_unpruned.fit(X_train, y_train)
print(f"Unpruned — Train: {dt_unpruned.score(X_train, y_train):.3f}, "
 f"Test: {dt_unpruned.score(X_test, y_test):.3f}")

# Pruned tree (generalizes better)
dt_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_pruned.fit(X_train, y_train)
print(f"Pruned — Train: {dt_pruned.score(X_train, y_train):.3f}, "
 f"Test: {dt_pruned.score(X_test, y_test):.3f}")

# Visualize the pruned tree
plt.figure(figsize=(12, 6))
plot_tree(dt_pruned, feature_names=iris.feature_names,
 class_names=iris.target_names, filled=True)
plt.title("Pruned Decision Tree (max_depth=3)")
plt.show()
```

---

## 6. Feature Importance

Decision Trees rank features by how much they reduce impurity across all splits:

```python
import pandas as pd
fi = pd.Series(dt_pruned.feature_importances_, index=iris.feature_names)
fi.sort_values().plot(kind='barh', title='Feature Importance')
plt.show()
```

> Features that never appear in any split have importance = 0.

---

**External Exercise:** [Codebasics Decision Tree Lab](https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/9_decision_tree_exercise.ipynb)
