# Model Persistence: Pickle vs. Joblib 💾

## 1. The Concept: Serialization
In Machine Learning, training a model can take hours or even days. **Model Persistence** allows you to save the trained model to a file (Serialization) and load it back later (Deserialization) to make predictions without retraining.

---

## 2. Comparison: Pickle vs. Joblib

Both libraries serialize Python objects, but they have distinct use cases.

| Feature | `pickle` 🥒 | `joblib` ⚡ |
| :--- | :--- | :--- |
| **Primary Use** | General-purpose Python object serialization. | Efficient serialization for large NumPy arrays. |
| **Best For** | Small, simple dictionaries or lists. | **Scikit-Learn models** (which rely heavily on NumPy). |
| **Performance** | Slower with large numerical data. | Optimized for large arrays (uses memory mapping). |
| **File Format** | Standard Python byte stream. | Optimized binary format. |
| **Usage** | Built-in (no install needed). | Requires `pip install joblib`. |

### ⚠️ Security Warning
**NEVER** load a pickle/joblib file from an untrusted source. Malicious code inside the file can execute immediately upon loading.

---

## 3. Workflow

### Pickle Example
```python
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression().fit(X, y)

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print(f"Prediction: {loaded_model.predict([[6]])[0]:.2f}")  # ~12.00
```

### Joblib Example
```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# Save (optimized for NumPy arrays)
joblib.dump(model, 'rf_model.joblib')

# Load
loaded_rf = joblib.load('rf_model.joblib')
print(f"Accuracy: {loaded_rf.score(iris.data, iris.target):.4f}")
```

---

**External Exercise:** [Codebasics Model Saving Lab](https://github.com/codebasics/py/blob/master/ML/4_save_model/4_save_model.ipynb)
