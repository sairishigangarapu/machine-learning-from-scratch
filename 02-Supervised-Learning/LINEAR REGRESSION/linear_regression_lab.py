import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def run_linear_regression():
    # ---------------------------------------------------------
    # 1. Data Preparation (Creating Dummy Data for Demo)
    # ---------------------------------------------------------
    data = {
        'area': [2600, 3000, 3200, 3600, 4000],
        'price': [550000, 565000, 610000, 680000, 725000]
    }
    df = pd.DataFrame(data)
    
    print("📊 Training Data:")
    print(df.head())
    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Visualization (Before Training)
    # ---------------------------------------------------------
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price (USD)')
    plt.scatter(df.area, df.price, color='red', marker='+', label='Actual Data')

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    # Create the Linear Regression Object
    reg = linear_model.LinearRegression()
    
    # Fit the model (Training)
    # Note: sklearn expects a 2D array for inputs, hence df[['area']]
    reg.fit(df[['area']], df.price)

    # ---------------------------------------------------------
    # 4. Model Inspection
    # ---------------------------------------------------------
    m = reg.coef_[0]      # Slope (Gradient)
    c = reg.intercept_    # Intercept (Bias)
    
    print(f"✅ Model Trained!")
    print(f"Slope (m): {m:.2f}")
    print(f"Intercept (c): {c:.2f}")
    print(f"Equation: y = {m:.2f}x + {c:.2f}")

    # ---------------------------------------------------------
    # 5. Prediction & Visualization
    # ---------------------------------------------------------
    # Predict prices for the existing areas to plot the line
    df['predicted_price'] = reg.predict(df[['area']])
    
    # Plot the Line of Best Fit
    plt.plot(df.area, df.predicted_price, color='blue', label='Line of Best Fit')
    plt.legend()
    plt.title("Linear Regression: Area vs Price")
    plt.show()

    # ---------------------------------------------------------
    # 6. Single Prediction Example
    # ---------------------------------------------------------
    new_area = 3300
    predicted_value = reg.predict([[new_area]])
    print("-" * 30)
    print(f"🔮 Prediction for {new_area} sq ft: ${predicted_value[0]:,.2f}")

if __name__ == "__main__":
    run_linear_regression()
