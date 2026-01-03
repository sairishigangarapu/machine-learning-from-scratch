import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def run_knn_lab():
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    print("📊 Data Loaded. Shape:", df.shape)

    # Visualization (Context)
    df0 = df[df.target == 0]
    df1 = df[df.target == 1]
    df2 = df[df.target == 2]

    plt.figure(figsize=(8, 6))
    plt.scatter(df0.iloc[:, 0], df0.iloc[:, 1], color='red', label='Setosa')
    plt.scatter(df1.iloc[:, 0], df1.iloc[:, 1], color='blue', label='Versicolor')
    plt.scatter(df2.iloc[:, 0], df2.iloc[:, 1], color='green', label='Virginica')
    plt.title("Iris Dataset: Sepal Length vs Width")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()
    plt.show()

    # ---------------------------------------------------------
    # 2. Pipeline Construction (Scaling + KNN)
    # ---------------------------------------------------------
    X = df.drop('target', axis=1)
    y = df['target']

    # CRITICAL STEP: KNN requires scaling. 
    # We use a Pipeline to ensure scaling happens correct inside CV.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Step 1: Scale features
        ('knn', KNeighborsClassifier(n_neighbors=5)) # Step 2: KNN
    ])

    # ---------------------------------------------------------
    # 3. K-Fold Cross Validation
    # ---------------------------------------------------------
    print("\n🔄 Running 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf)

    print(f"CV Scores: {cv_scores}")
    print(f"✅ Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ---------------------------------------------------------
    # 4. Detailed Evaluation (Train/Test Split)
    # ---------------------------------------------------------
    print("\n🔍 Detailed Classification Report (Hold-out Test Set)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit pipeline on training data
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Report
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    run_knn_lab()
