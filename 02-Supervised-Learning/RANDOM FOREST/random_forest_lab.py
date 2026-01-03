import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def run_random_forest_lab():
    # ---------------------------------------------------------
    # 1. Load Data (Digits Dataset)
    # ---------------------------------------------------------
    digits = load_digits()
    
    # Convert to DataFrame for cleaner handling
    df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])])
    df['target'] = digits.target
    
    print("📊 Input Data Shape:", df.shape)
    
    # ---------------------------------------------------------
    # 2. Split Data
    # ---------------------------------------------------------
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 3. Train Model (The Ensemble)
    # ---------------------------------------------------------
    # n_estimators=100 means we are building 100 decision trees
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 4. Evaluation
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc:.4f}")

    # ---------------------------------------------------------
    # 5. Visualization: Confusion Matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: Random Forest on Digits")
    plt.show()

    # ---------------------------------------------------------
    # 6. Bonus: Feature Importance (The "Senior Engineer" Check)
    # ---------------------------------------------------------
    # Random Forest can tell us WHICH pixels were most important for the decision
    # We plot the top 10 most important features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    
    plt.figure(figsize=(10, 6))
    feature_importance.nlargest(10).plot(kind='barh', color='green')
    plt.title("Top 10 Most Important Pixels (Features)")
    plt.xlabel("Importance Score")
    plt.show()

if __name__ == "__main__":
    run_random_forest_lab()
