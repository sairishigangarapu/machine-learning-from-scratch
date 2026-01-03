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

1.  **Train** your model.
2.  **Serialize (Dump)** the model object to a file (e.g., `model.pkl`).
3.  **Deserialize (Load)** the file back into a variable.
4.  **Predict** using the loaded object.

---
**External Exercise:** [Codebasics Model Saving Lab](https://github.com/codebasics/py/blob/master/ML/4_save_model/4_save_model.ipynb)
