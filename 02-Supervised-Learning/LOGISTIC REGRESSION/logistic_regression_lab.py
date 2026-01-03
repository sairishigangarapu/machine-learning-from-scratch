import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def run_binary_classification():
    print("\n🔹 PART 1: Binary Classification (Insurance Data)")
    
    # 1. Dummy Data Generation
    # (Simulating Age vs. Buying Insurance)
    data = {
        'age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49],
        'have_insurance': [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.have_insurance, test_size=0.1, random_state=42)
    
    # 3. Train Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 4. Visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(df.age, df.have_insurance, marker='+', color='blue', label='Actual Data')
    
    # Plotting the Sigmoid Curve
    ages = np.linspace(df.age.min(), df.age.max(), 100).reshape(-1, 1)
    probs = model.predict_proba(ages)[:, 1] # Probability of Class 1
    plt.plot(ages, probs, color='red', label='Sigmoid Curve')
    
    plt.title("Binary Logistic Regression: Age vs Insurance")
    plt.xlabel("Age")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    print("✅ Binary classification plot generated.")


def run_multiclass_classification():
    print("\n🔹 PART 2: Multiclass Classification (Handwritten Digits)")
    
    # 1. Load Standard Dataset (Digits 0-9)
    digits = load_digits()
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    
    # 3. Train Model (max_iter increased for convergence)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    score = model.score(X_test, y_test)
    print(f"Model Accuracy: {score:.4f}")
    
    # 5. Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualizing the Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix: Multiclass Digits")
    plt.show()

if __name__ == "__main__":
    run_binary_classification()
    run_multiclass_classification()
