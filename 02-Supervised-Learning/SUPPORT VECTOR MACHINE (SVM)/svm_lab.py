import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def plot_decision_boundary(X, y, model, title):
    """
    Helper function to visualize the SVM Boundary.
    We use a meshgrid to predict across the entire 2D plane.
    """
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict class for every point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contours and data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(title)
    plt.show()

def run_svm_lab():
    # 1. Load Data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # 2. Split Data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Standard Model (Use all 4 features)
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    
    print(f"✅ Full Model Accuracy (4 Features): {model.score(X_test, y_test):.4f}")

    # ---------------------------------------------------------
    # 4. Advanced Visualization: 2D Slice
    # ---------------------------------------------------------
    print("\n🎨 Generating Decision Boundaries (2D Slice)...")
    
    # We take only the first two features (Sepal Length & Width) to visualize in 2D
    X_2d = iris.data[:, :2]  
    y_2d = iris.target

    # Scenario A: Linear Kernel
    model_linear = SVC(kernel='linear', C=1.0)
    model_linear.fit(X_2d, y_2d)
    plot_decision_boundary(X_2d, y_2d, model_linear, "SVM: Linear Kernel")

    # Scenario B: RBF Kernel (Non-Linear)
    model_rbf = SVC(kernel='rbf', C=1.0, gamma=0.7)
    model_rbf.fit(X_2d, y_2d)
    plot_decision_boundary(X_2d, y_2d, model_rbf, "SVM: RBF Kernel (C=1, Gamma=0.7)")

if __name__ == "__main__":
    run_svm_lab()
