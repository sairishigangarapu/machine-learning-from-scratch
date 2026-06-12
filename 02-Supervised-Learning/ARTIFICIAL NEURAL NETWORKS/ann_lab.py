import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_moons, make_regression
from sklearn.preprocessing import StandardScaler

def run_ann_classification():
 print("=" * 50)
 print("PART 1: Classification — XOR + Moons")
 print("=" * 50)

 # ---------------------------------------------------------
 # 1. XOR Problem (Perceptron fails, MLP succeeds)
 # ---------------------------------------------------------
 X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
 y_xor = np.array([0, 1, 1, 0])

 from sklearn.linear_model import Perceptron
 perc = Perceptron()
 perc.fit(X_xor, y_xor)
 print(f"\n XOR — Perceptron accuracy: {perc.score(X_xor, y_xor):.2f} (fails)")

 mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000, random_state=42)
 mlp_xor.fit(X_xor, y_xor)
 print(f" XOR — MLP accuracy: {mlp_xor.score(X_xor, y_xor):.2f} (succeeds)")
 print(f" Predictions: {mlp_xor.predict(X_xor)}")

 # ---------------------------------------------------------
 # 2. Non-Linear Decision Boundary (Moons Dataset)
 # ---------------------------------------------------------
 X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 scaler = StandardScaler()
 X_train_s = scaler.fit_transform(X_train)
 X_test_s = scaler.transform(X_test)

 # Logistic Regression (linear boundary)
 from sklearn.linear_model import LogisticRegression
 lr = LogisticRegression()
 lr.fit(X_train_s, y_train)
 print(f"\n Moons — Logistic Regression accuracy: {lr.score(X_test_s, y_test):.3f}")

 # MLP (non-linear boundary)
 mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, random_state=42)
 mlp.fit(X_train_s, y_train)
 print(f" Moons — MLP (10,10) accuracy: {mlp.score(X_test_s, y_test):.3f}")

 # ---------------------------------------------------------
 # 3. Effect of Architecture
 # ---------------------------------------------------------
 print("\n Architecture Comparison:")
 for hidden in [(5,), (10, 10), (20, 20, 20)]:
 mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=2000, random_state=42)
 mlp.fit(X_train_s, y_train)
 acc = mlp.score(X_test_s, y_test)
 params = sum(w.size for w in mlp.coefs_)
 print(f" Hidden {str(hidden):12s} → Accuracy: {acc:.3f}, Parameters: {params}")

def run_ann_regression():
 print("\n" + "=" * 50)
 print("PART 2: Regression — Sine Wave")
 print("=" * 50)

 np.random.seed(42)
 X = np.sort(np.random.uniform(0, 10, 200)).reshape(-1, 1)
 y = np.sin(X).ravel() + np.random.normal(0, 0.2, 200)

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Linear baseline
 from sklearn.linear_model import LinearRegression
 lr = LinearRegression()
 lr.fit(X_train, y_train)
 mse_lr = mean_squared_error(y_test, lr.predict(X_test))

 # MLP Regressor
 mlp = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=2000, random_state=42)
 mlp.fit(X_train, y_train)
 mse_mlp = mean_squared_error(y_test, mlp.predict(X_test))

 print(f"\n Linear Regression MSE: {mse_lr:.4f}")
 print(f" MLP (20,20) MSE: {mse_mlp:.4f}")
 print(f" Improvement: {(1 - mse_mlp/mse_lr)*100:.1f}%")

if __name__ == "__main__":
 run_ann_classification()
 run_ann_regression()
