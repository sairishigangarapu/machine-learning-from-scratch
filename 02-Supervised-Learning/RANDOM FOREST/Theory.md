# Random Forest: Ensemble Learning

## 1. Concept Overview
**Random Forest** is a supervised learning algorithm that utilizes **Ensemble Learning**. It operates by constructing a multitude of Decision Trees during training and outputting the class that is the **mode** (majority vote) of the classes (classification) or mean prediction (regression) of the individual trees.

### The Core Mechanism: Bagging
Random Forest relies on **Bagging** (Bootstrap Aggregation).
1. **Bootstrapping:** Randomly sampling the dataset with replacement (some rows are repeated, some are ignored).
2. **Feature Randomness:** At each split in the tree, the algorithm considers only a random subset of features (not all).
3. **Aggregation:** Combining the results of all trees to reduce variance and prevent overfitting.

> **"The Wisdom of the Crowd"**: A single decision tree is prone to noise (High Variance). A forest of decorrelated trees averages out the noise, resulting in a robust model.

---

## 2. Key Hyperparameters (`RandomForestClassifier`)

| Parameter | Description | Recommended Tuning |
| :--- | :--- | :--- |
| `n_estimators` | Number of trees in the forest. | Higher is better (stable), but slower. Start with 100. |
| `max_depth` | Max depth of each tree. | Limit this to prevent overfitting (e.g., 10-20). |
| `max_features` | Features considered per split. | Default is `sqrt(n_features)` for classification. |
| `min_samples_leaf` | Min samples required at a leaf node. | Increase this (e.g., 5 or 10) to smooth predictions. |
| `n_jobs` | Number of CPU cores to use. | Set to `-1` to use all processors (faster training). |

---

## 3. Advantages vs. Disadvantages

### Pros
* **Robustness:** Handles outliers and noise much better than single Decision Trees.
* **Feature Importance:** Can automatically identify which features are most predictive.
* **No Scaling:** Like Decision Trees, it requires no feature scaling/normalization.

### Cons
* **Black Box:** Harder to interpret than a single tree (you can't easily draw the forest).
* **Latency:** Slower predictions because every tree has to calculate an output.
* **Memory:** Stores all trees in memory, which can be significant for very large forests.

---

## 4. Out-of-Bag (OOB) Evaluation

Because each tree is trained on a bootstrap sample, roughly **36.8%** of the original data is left out of each tree. These "out-of-bag" samples can serve as a built-in validation set — no explicit `train_test_split` needed.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(iris.data, iris.target)

print(f"OOB Score: {rf.oob_score_:.4f}")
```

> The OOB score approximates cross-validation performance at zero extra computational cost.

---

## 5. Code Example: Full Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
 iris.data, iris.target, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 6. Feature Importance

Random Forests aggregate impurity reduction across all trees, giving a robust feature importance ranking:

```python
fi = pd.Series(rf.feature_importances_, index=iris.feature_names)
fi.sort_values().plot(kind='barh', title='Random Forest Feature Importance')
plt.xlabel('Mean Decrease in Impurity')
plt.show()
```

> Unlike a single Decision Tree, RF importance is averaged over many decorrelated trees, making it more stable and reliable.

---

**External Exercise:** [Codebasics Random Forest Lab](https://github.com/codebasics/py/blob/master/ML/11_random_forest/11_random_forest.ipynb)
