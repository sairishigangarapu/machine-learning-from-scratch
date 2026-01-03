import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    """
    Performs gradient descent to minimize MSE for linear regression
    and visualizes the process dynamically.
    """
    # 🔧 Step 1: Initialize model parameters
    m_curr = 0
    b_curr = 0

    # 📈 Step 2: Set hyperparameters
    learning_rate = 0.01 # Tuning this is key for convergence
    iterations = 100     # Defining the stopping condition
    n = len(x)           # Number of data points

    # 📊 Step 3: Set up interactive plotting
    plt.ion()
    fig, ax = plt.subplots()

    # 🔁 Step 4: Optimization Loop
    for i in range(iterations):
        # 🧮 4.1: Prediction
        y_pred = m_curr * x + b_curr

        # 🧾 4.2: Cost Calculation (MSE)
        cost = (1 / n) * np.sum((y - y_pred) ** 2)

        # 🧠 4.3: Compute Gradients (Partial Derivatives)
        # Derivative w.r.t m
        md = -(2 / n) * np.sum(x * (y - y_pred))
        # Derivative w.r.t b
        bd = -(2 / n) * np.sum(y - y_pred)

        # 🔄 4.4: Update Parameters
        m_curr -= learning_rate * md
        b_curr -= learning_rate * bd

        # 🎨 4.5: Visualization
        ax.clear()
        ax.scatter(x, y, color='blue', label='Data Points')
        ax.plot(x, y_pred, color='red', label='Regression Line')
        ax.set_title(f"Iteration {i+1}/{iterations} | Cost: {cost:.4f}")
        ax.set_xlabel("Feature (X)")
        ax.set_ylabel("Target (Y)")
        ax.legend()
        ax.grid(True)
        
        plt.pause(0.1) # Pause to render animation

    # 🛑 Step 5: Final Output
    print(f"Final Parameters: m = {m_curr:.4f}, b = {b_curr:.4f}, Cost = {cost:.4f}")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # 📥 Input Data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    # 🚀 Run
    gradient_descent(x, y)
