# Ensemble Learning: Bagging & Validation 🧺

## 1. Concept Overview: The Wisdom of the Crowd
**Ensemble Learning** is a meta-approach to machine learning that combines predictions from multiple models to achieve better predictive performance than any single constituent model.

### The Problem: Variance
Single models, especially Decision Trees, suffer from **High Variance**.
* A small change in the training data can result in a completely different tree structure.
* This leads to **Overfitting** (memorizing the specific noise of the training set).

### The Solution: Bagging (Bootstrap Aggregating)
Bagging reduces variance by training multiple models in parallel on different subsets of data and averaging their outputs.



1.  **Bootstrapping (Resampling with Replacement):** We create $N$ new datasets by sampling from the original data. Some rows are repeated, and some are omitted.
2.  **Parallel Training:** We train an independent model (e.g., a Decision Tree) on each bootstrap sample.
3.  **Aggregation:**
    * **Classification:** Majority Vote ("Hard Voting").
    * **Regression:** Average of predictions ("Soft Voting").

---

## 2. Boosting vs. Bagging
While Bagging reduces *Variance* (Overfitting), Boosting reduces *Bias* (Underfitting).

| Feature | Bagging (e.g., Random Forest) | Boosting (e.g., XGBoost) |
| :--- | :--- | :--- |
| **Goal** | Reduce Variance (Stability) | Reduce Bias (Accuracy) |
| **Training** | Parallel (Independent models) | Sequential (Each model fixes the previous) |
| **Method** | Random Resampling | Reweighting hard-to-predict examples |

---

## 3. Validation Strategies

To ensure our model generalizes well, we need robust validation techniques.

### A. K-Fold Cross-Validation
Systematically splits the data into $K$ "folds."
* **Process:** Train on $K-1$ folds, test on the remaining 1 fold. Repeat $K$ times.
* **Result:** The average score represents the model's true performance.



### B. Out-of-Bag (OOB) Score
Specific to Bagging. Since bootstrapping omits ~37% of data in each sample, these "unused" (Out-of-Bag) data points can be used as a built-in validation set.
* **Advantage:** No need to set aside a separate validation set.

---

**External Exercise:** [Codebasics Bagging Lab](https://github.com/codebasics/py/blob/master/ML/18_bagging/bagging.ipynb)
