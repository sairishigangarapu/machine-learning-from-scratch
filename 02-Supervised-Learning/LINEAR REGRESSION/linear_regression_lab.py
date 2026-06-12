import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def run_linear_regression():
 # ---------------------------------------------------------
 # 1. Data Preparation (Creating Dummy Data for Demo)
 # ---------------------------------------------------------
 data = {
 'area': [2600, 3000, 3200, 3600, 4000, 4200, 4400, 4800, 5000, 5200],
 'price': [550000, 565000, 610000, 680000, 725000, 750000, 780000, 830000, 870000, 910000]
 }
 df = pd.DataFrame(data)

 print(" Full Dataset:")
 print(df)
 print("-" * 40)

 # ---------------------------------------------------------
 # 2. Train/Test Split (80/20)
 # ---------------------------------------------------------
 X_train, X_test, y_train, y_test = train_test_split(
 df[['area']], df['price'], test_size=0.2, random_state=42
 )
 print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

 # ---------------------------------------------------------
 # 3. Model Training
 # ---------------------------------------------------------
 reg = linear_model.LinearRegression()
 reg.fit(X_train, y_train)

 m = reg.coef_[0]
 c = reg.intercept_

 print(f"\n Model Trained!")
 print(f"Slope (m): {m:.2f}")
 print(f"Intercept (c): {c:.2f}")
 print(f"Equation: y = {m:.2f}x + {c:.2f}")

 # ---------------------------------------------------------
 # 4. Evaluation (R² and MSE)
 # ---------------------------------------------------------
 y_pred = reg.predict(X_test)
 r2 = r2_score(y_test, y_pred)
 mse = mean_squared_error(y_test, y_pred)
 rmse = np.sqrt(mse)

 print(f"\n Test Set Evaluation:")
 print(f" R² Score: {r2:.4f} (1.0 = perfect)")
 print(f" MSE: {mse:.2f}")
 print(f" RMSE: ${rmse:,.2f}")

 # ---------------------------------------------------------
 # 5. Visualization
 # ---------------------------------------------------------
 plt.scatter(df['area'], df['price'], color='blue', label='All Data')
 plt.scatter(X_test, y_test, color='red', marker='x', s=100, label='Test Set')

 line_x = np.linspace(df['area'].min(), df['area'].max(), 100).reshape(-1, 1)
 plt.plot(line_x, reg.predict(line_x), color='green', linewidth=2, label='Line of Best Fit')

 plt.xlabel('Area (sq ft)')
 plt.ylabel('Price (USD)')
 plt.title("Linear Regression: Train/Test Split + Best Fit")
 plt.legend()
 plt.show()

 # ---------------------------------------------------------
 # 6. Single Prediction
 # ---------------------------------------------------------
 new_area = 3300
 predicted_value = reg.predict([[new_area]])
 print("-" * 40)
 print(f" Prediction for {new_area} sq ft: ${predicted_value[0]:,.2f}")

if __name__ == "__main__":
 run_linear_regression()
