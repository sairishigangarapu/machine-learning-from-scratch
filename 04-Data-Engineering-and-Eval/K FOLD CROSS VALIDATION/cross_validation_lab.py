import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_score(model, X_train, X_test, y_train, y_test):
    """Helper function to fit and score a model."""
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def run_cv_lab():
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(f"📊 Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # ---------------------------------------------------------
    # 2. The "Simple Split" (Baseline)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n🔹 Method 1: Single Train-Test Split (Risky)")
    print(f"Logistic Regression: {get_score(LogisticRegression(max_iter=5000), X_train, X_test, y_train, y_test):.4f}")
    print(f"SVM:                 {get_score(SVC(), X_train, X_test, y_train, y_test):.4f}")
    print(f"Random Forest:       {get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test):.4f}")

    # ---------------------------------------------------------
    # 3. Stratified K-Fold (Under the Hood)
    # ---------------------------------------------------------
    # This shows what cross_val_score does internally
    print("\n🔹 Method 2: Manual Stratified K-Fold (Understanding the Logic)")
    skf = StratifiedKFold(n_splits=3)
    
    scores_rf = []
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        score = get_score(RandomForestClassifier(n_estimators=40), X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        scores_rf.append(score)
        print(f"   Fold {i+1} Score: {score:.4f}")
        
    print(f"   => Average Score: {np.mean(scores_rf):.4f}")

    # ---------------------------------------------------------
    # 4. cross_val_score (The Professional Way)
    # ---------------------------------------------------------
    print("\n🔹 Method 3: cross_val_score (Production Ready)")
    # Note: cross_val_score automatically uses StratifiedKFold for classification models
    
    # Logistic Regression
    lr_scores = cross_val_score(LogisticRegression(max_iter=5000), X, y, cv=5)
    print(f"Logistic Regression (Avg of 5): {np.mean(lr_scores):.4f}")

    # SVM
    svm_scores = cross_val_score(SVC(), X, y, cv=5)
    print(f"SVM (Avg of 5):                 {np.mean(svm_scores):.4f}")

    # Random Forest
    rf_scores = cross_val_score(RandomForestClassifier(n_estimators=40), X, y, cv=5)
    print(f"Random Forest (Avg of 5):       {np.mean(rf_scores):.4f}")

if __name__ == "__main__":
    run_cv_lab()
