# Cross-Validation Strategies 🔄

## 1. The Problem with Simple Splits
When using a simple `train_test_split`, the accuracy of your model depends heavily on **which** data points end up in the training set vs. the test set.
* **Lucky Split:** You might get easy test examples $\rightarrow$ High Accuracy (Misleading).
* **Unlucky Split:** You might get all the outliers in the test set $\rightarrow$ Low Accuracy.

## 2. K-Fold Cross-Validation (The Gold Standard)
To get a stable measure of model performance, we split the data into **$K$ folds** (subsets).

### How It Works
1.  Divide data into $K$ equal parts (e.g., $K=5$).
2.  **Iteration 1:** Train on Folds 2-5, Test on Fold 1.
3.  **Iteration 2:** Train on Folds 1, 3-5, Test on Fold 2.
4.  ...Repeat $K$ times.
5.  **Final Score:** Average of all $K$ scores.

$$
\text{Accuracy} = \frac{1}{K} \sum_{i=1}^{K} \text{Score}_i
$$

---

## 3. Variations of Cross-Validation

### A. Stratified K-Fold (Crucial for Classification) 📊
Standard K-Fold splits purely randomly. If your dataset is imbalanced (e.g., 90% Cat A, 10% Cat B), a random split might result in a fold having **zero** examples of Cat B.
* **Stratified K-Fold** ensures that each fold preserves the **same percentage of samples for each class** as the original dataset.
* *Note: Scikit-Learn's `cross_val_score` uses this by default for classifiers.*

### B. Leave-One-Out (LOOCV) 🐢
* **Process:** Use $N-1$ samples to train, and 1 sample to test. Repeat $N$ times.
* **Pros:** Uses maximum data for training.
* **Cons:** Extremely slow/expensive for large datasets.

### C. Shuffle Split 🔀
* **Process:** Randomly samples training and test sets $X$ times. No guarantee that every point is tested exactly once.
* **Use Case:** When you want to control the exact size of the test set independent of the number of iterations.

---

## 4. Summary Table

| Technique | Preserves Class Balance? | Speed | Use Case |
| :--- | :--- | :--- | :--- |
| **Train/Test Split** | ❌ No | ⚡ Very Fast | Prototyping, Huge Datasets |
| **K-Fold** | ❌ No | 🐢 Medium | Regression, Balanced Data |
| **Stratified K-Fold** | ✅ Yes | 🐢 Medium | Classification (Standard) |
| **LOOCV** | ✅ Yes | 🐌 Very Slow | Very Small Datasets |

---
**External Exercise:** [Codebasics Cross Validation Lab](https://github.com/codebasics/py/blob/master/ML/12_KFold/12_k_fold.ipynb)
