# Decision Tree: The "White Box" Model 🌳

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

* **Gini = 0:** Pure node (all samples belong to one class).
* **Gini = 0.5:** Maximum impurity (50/50 split in binary classification).

### B. Entropy (Information Gain)
A measure of disorder derived from thermodynamics/information theory.

$$
Entropy = - \sum_{i=1}^{n} p_i \log_2(p_i)
$$

$$
\text{Info Gain} = \text{Entropy(Parent)} - \text{Weighted Avg Entropy(Children)}
$$

---

## 3. Advantages & Disadvantages

### ✅ Pros
* **Interpretability:** You can draw the tree and explain exactly *why* a decision was made.
* **No Scaling:** Requires no normalization or scaling of data.
* **Versatile:** Handles both Numerical and Categorical data.

### ❌ Cons
* **Overfitting:** Trees tend to grow very deep and memorize the noise (High Variance).
* **Instability:** Small changes in data can result in a completely different tree structure.

---
**External Exercise:** [Codebasics Decision Tree Lab](https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/9_decision_tree_exercise.ipynb)
