import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import make_regression

def run_regularization_lab():
    # ---------------------------------------------------------
    # 1. Data Generation (Simulating a Complex Housing Dataset)
    # ---------------------------------------------------------
    # We generate a dataset with 50 features, but only 10 are actually useful.
    # This simulates "Noise" that leads to Overfitting.
    print("📊 Generating Synthetic Data (50 Features, only 10 informative)...")
    X, y = make_regression(n_samples=200, n_features=50, n_informative=10, noise=10, random_state=42)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---------------------------------------------------------
    # 2. Linear Regression (The Baseline)
    # ---------------------------------------------------------
    # Without regularization, this model will likely overfit by using all 50 features.
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    
    print(f"\n📉 Linear Regression Test Score: {lr_score:.4f}")
    print(f"   -> Features used: {np.sum(lr.coef_ != 0)}/50")

    # ---------------------------------------------------------
    # 3. Lasso Regression (L1 - The Feature Selector)
    # ---------------------------------------------------------
    # Lasso should drive the coefficients of the 40 useless features to Zero.
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_score = lasso.score(X_test, y_test)
    
    print(f"\n🛡️ Lasso (L1) Test Score:        {lasso_score:.4f}")
    print(f"   -> Features used: {np.sum(lasso.coef_ != 0)}/50 (Notice the drop!)")

    # ---------------------------------------------------------
    # 4. Ridge Regression (L2 - The Stabilizer)
    # ---------------------------------------------------------
    # Ridge will keep all features but shrink their weights to prevent extreme values.
    ridge = Ridge(alpha=0.1, max_iter=10000)
    ridge.fit(X_train, y_train)
    ridge_score = ridge.score(X_test, y_test)
    
    print(f"\n⚖️ Ridge (L2) Test Score:        {ridge_score:.4f}")
    print(f"   -> Features used: {np.sum(ridge.coef_ != 0)}/50")

    # ---------------------------------------------------------
    # 5. Visualizing Coefficients (The Proof)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(lr.coef_, marker='o', label='Linear Regression', alpha=0.5)
    plt.plot(ridge.coef_, marker='s', label='Ridge (L2)', alpha=0.7)
    plt.plot(lasso.coef_, marker='^', label='Lasso (L1)', alpha=0.9)
    
    plt.title("Comparison of Model Coefficients")
    plt.xlabel("Feature Index (0-49)")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n💡 Observation: Notice how Lasso (Green triangles) sets many weights exactly to zero.")

if __name__ == "__main__":
    run_regularization_lab()
