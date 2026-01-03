```python
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint, loguniform

def run_tuning_lab():
    # 1. Load Data
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define Configuration Dictionary
    # This allows us to loop through completely different models dynamically
    model_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000),
            "params": {
                "C": loguniform(1e-4, 1e4),
                "solver": ['liblinear', 'saga']
            }
        },
        "SVM": {
            "model": SVC(),
            "params": {
                "C": uniform(0.1, 100),
                "kernel": ['linear', 'rbf'],
                "gamma": ['scale', 'auto']
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 15),
                "min_samples_split": randint(2, 10)
            }
        }
    }

    # 4. The Tuning Engine
    results = []
    print("🚀 Starting Hyperparameter Tuning Process...\n")

    for name, config in model_config.items():
        print(f"🔍 Tuning {name}...")
        
        # Initialize RandomizedSearchCV
        rs = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=20,          # Try 20 random combinations
            cv=5,               # 5-fold Cross Validation
            random_state=42,
            n_jobs=-1,          # Use all CPU cores
            verbose=0
        )
        
        # Fit
        rs.fit(X_train, y_train)
        
        # Evaluate on Test Set (Unseen Data)
        y_pred = rs.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log Results
        results.append({
            "Model": name,
            "Best CV Score": rs.best_score_,
            "Test Accuracy": acc,
            "Best Params": rs.best_params_
        })

    # 5. Display Leaderboard
    results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)

    print("\n🏆 Model Leaderboard:")
    print(results_df[["Model", "Test Accuracy", "Best CV Score"]])

    print("\n🥇 Best Configuration:")
    best_model_row = results_df.iloc[0]
    print(f"Model: {best_model_row['Model']}")
    print(f"Params: {best_model_row['Best Params']}")

if __name__ == "__main__":
    run_tuning_lab()
