import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def run_xgboost_lab():
    # ---------------------------------------------------------
    # 1. Compare Algorithms on Synthetic Data
    # ---------------------------------------------------------
    print("=" * 50)
    print("PART 1: Algorithm Comparison (Synthetic Data)")
    print("=" * 50)

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                         learning_rate=0.1, random_state=42),
    }

    # Try XGBoost if installed
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=42
        )
    except ImportError:
        print("⚠️  xgboost not installed — pip install xgboost")

    print(f"\n{'Model':<25s} {'Train Acc':>10s} {'Test Acc':>10s} {'Parameters':>12s}")
    print("-" * 60)
    for name, model in models.items():
        Xtr = X_train_s if 'Logistic' in name else X_train
        Xte = X_test_s if 'Logistic' in name else X_test
        model.fit(Xtr, y_train)
        train_acc = accuracy_score(y_train, model.predict(Xtr))
        test_acc = accuracy_score(y_test, model.predict(Xte))
        n_params = sum(getattr(tree, 'n_nodes', 0) if hasattr(tree, 'n_nodes')
                       else getattr(tree, 'tree_', {}).get('node_count', 0)
                       for tree in getattr(model, 'estimators_', [model]))
        print(f"{name:<25s} {train_acc:>10.4f} {test_acc:>10.4f}")

    # ---------------------------------------------------------
    # 2. Learning Rate vs n_estimators Trade-off
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 2: Learning Rate vs Trees Trade-off")
    print("=" * 50)

    try:
        import xgboost as xgb
        best_acc = 0
        best_cfg = ""
        print(f"\n{'LR':>6s} {'Trees':>6s} {'Test Acc':>10s}")
        print("-" * 26)
        for lr in [0.01, 0.05, 0.1, 0.3]:
            for n_est in [50, 100, 200, 500]:
                m = xgb.XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=lr,
                                       eval_metric='logloss', random_state=42)
                m.fit(X_train, y_train)
                acc = accuracy_score(y_test, m.predict(X_test))
                print(f"{lr:>6.2f} {n_est:>6d} {acc:>10.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = f"lr={lr}, n_est={n_est}"
        print(f"\n🏆 Best: {best_cfg} → {best_acc:.4f}")
    except ImportError:
        pass

    # ---------------------------------------------------------
    # 3. Real Dataset: Breast Cancer
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 3: Real Dataset (Breast Cancer)")
    print("=" * 50)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                   eval_metric='logloss', random_state=42)
    except ImportError:
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.1, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=data.feature_names)
    fi.nlargest(10).plot(kind='barh', title='Top 10 Features (Breast Cancer)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

import pandas as pd
if __name__ == "__main__":
    run_xgboost_lab()
