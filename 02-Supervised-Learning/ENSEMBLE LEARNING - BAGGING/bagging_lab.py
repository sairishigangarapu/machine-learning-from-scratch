import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

def run_bagging_lab():
    # ---------------------------------------------------------
    # 1. Load Data (Breast Cancer Dataset - Binary Classification)
    # ---------------------------------------------------------
    # Note: This is a standard proxy for medical diagnosis tasks (like Diabetes)
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    print("📊 Data Loaded: Breast Cancer Diagnostic")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Preprocessing (Scaling)
    # ---------------------------------------------------------
    # Scaling is good practice for most ML pipelines, though Trees handle unscaled data well.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 3. Baseline: Single Decision Tree
    # ---------------------------------------------------------
    # We expect this to have higher variance (scores fluctuate more between folds)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_scores = cross_val_score(dt_model, X_scaled, y, cv=5)
    
    print("\n🌲 Single Decision Tree Performance:")
    print(f"Mean CV Accuracy: {dt_scores.mean():.4f}")
    print(f"Variance (Std Dev): {dt_scores.std():.4f}")

    # ---------------------------------------------------------
    # 4. Solution: Bagging Classifier
    # ---------------------------------------------------------
    # We use the Decision Tree as the "Base Estimator"
    print("\n🧺 Training Bagging Classifier (100 Trees)...")
    
    bag_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(), 
        n_estimators=100,     # Train 100 independent trees
        max_samples=0.8,      # Use 80% of data per tree (Bootstrap)
        oob_score=True,       # Use "leftover" data for validation
        random_state=42,
        n_jobs=-1             # Parallel processing
    )
    
    bag_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Evaluation & Comparison
    # ---------------------------------------------------------
    test_score = bag_model.score(X_test, y_test)
    oob_score = bag_model.oob_score_
    
    print("-" * 30)
    print(f"✅ Bagging Test Accuracy: {test_score:.4f}")
    print(f"✅ Bagging OOB Score:     {oob_score:.4f} (Estimated Generalization)")
    
    # Cross Validation on Bagging to confirm stability
    bag_cv_scores = cross_val_score(bag_model, X_scaled, y, cv=5)
    print(f"✅ Bagging Mean CV Score: {bag_cv_scores.mean():.4f}")
    
    print("\n🚀 Conclusion:")
    print(f"Improvement over Single Tree: +{(bag_cv_scores.mean() - dt_scores.mean())*100:.2f}%")

if __name__ == "__main__":
    run_bagging_lab()
