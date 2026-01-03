import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def run_pca_lab():
    # ---------------------------------------------------------
    # 1. Load Data (Handwritten Digits)
    # ---------------------------------------------------------
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    print("📊 Dataset Loaded: Handwritten Digits")
    print(f"Original Shape: {X.shape} (64 pixels/features per image)")
    
    # Visualization of the first digit
    plt.gray()
    plt.matshow(digits.data[0].reshape(8, 8))
    plt.title(f"Target Label: {digits.target[0]}")
    plt.show()

    # ---------------------------------------------------------
    # 2. Preprocessing & Split
    # ---------------------------------------------------------
    # Step A: Standard Scaling (Mandatory for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step B: Split Data
    # Important: Split BEFORE PCA to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

    # ---------------------------------------------------------
    # 3. Benchmark: Model WITHOUT PCA
    # ---------------------------------------------------------
    print("\n🐢 Training Logistic Regression on Full Data (64 Features)...")
    model_base = LogisticRegression(max_iter=1000)
    model_base.fit(X_train, y_train)
    score_base = model_base.score(X_test, y_test)
    print(f"✅ Baseline Accuracy: {score_base:.4f}")

    # ---------------------------------------------------------
    # 4. Applying PCA
    # ---------------------------------------------------------
    # We ask PCA to retain 95% of the useful information (variance)
    pca = PCA(n_components=0.95)
    
    # Fit only on training data
    X_train_pca = pca.fit_transform(X_train)
    # Transform test data using the learned rotation
    X_test_pca = pca.transform(X_test)
    
    print(f"\n📉 PCA Complete. Variance Retained: 95%")
    print(f"New Shape: {X_train_pca.shape}")
    print(f"Features reduced from 64 -> {pca.n_components_}")

    # ---------------------------------------------------------
    # 5. Model WITH PCA
    # ---------------------------------------------------------
    print("\n🐇 Training Logistic Regression on PCA Data...")
    model_pca = LogisticRegression(max_iter=1000)
    model_pca.fit(X_train_pca, y_train)
    score_pca = model_pca.score(X_test_pca, y_test)
    
    print(f"✅ PCA Accuracy: {score_pca:.4f}")
    
    # ---------------------------------------------------------
    # 6. Conclusion
    # ---------------------------------------------------------
    print("-" * 30)
    print(f"Speed/Efficiency Gain: Features reduced by {(64 - pca.n_components_)/64:.0%}")
    print(f"Accuracy Change: {score_pca - score_base:.4f}")
    if abs(score_pca - score_base) < 0.05:
        print("👉 Conclusion: PCA maintained accuracy while significantly reducing complexity!")

if __name__ == "__main__":
    run_pca_lab()
