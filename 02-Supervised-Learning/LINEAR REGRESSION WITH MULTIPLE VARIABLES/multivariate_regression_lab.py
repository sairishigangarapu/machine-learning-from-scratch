import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

def run_multivariate_lab():
    # ---------------------------------------------------------
    # 1. Data Preparation (With Missing Values)
    # ---------------------------------------------------------
    # Creating a dataset where one house is missing 'bedrooms' info
    data = {
        'area': [2600, 3000, 3200, 3600, 4000, 4100],
        'bedrooms': [3, 4, np.nan, 3, 5, 6],  # NaN represents missing data
        'age': [20, 15, 18, 30, 8, 8],
        'price': [550000, 565000, 610000, 595000, 760000, 810000]
    }
    df = pd.DataFrame(data)

    print("📊 Raw Data (Note the NaN value):")
    print(df)
    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Data Cleaning (Imputation)
    # ---------------------------------------------------------
    # Logic: We use the median because it's integer-safe and robust
    median_bedrooms = math.floor(df['bedrooms'].median())
    
    # Fill the NaN values
    df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)
    
    print(f"🧹 Data Cleaned! Replaced NaN with Median ({median_bedrooms}):")
    print(df)
    print("-" * 30)

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    reg = LinearRegression()
    
    # Features: Area, Bedrooms, Age
    # Target: Price
    reg.fit(df[['area', 'bedrooms', 'age']], df.price)

    # ---------------------------------------------------------
    # 4. Model Inspection
    # ---------------------------------------------------------
    print("✅ Model Trained!")
    print(f"Coefficients (Weights): {reg.coef_}")
    print(f"Intercept (Bias): {reg.intercept_:.2f}")
    
    # Mathematical explanation of the learned model
    print(f"\nFormula: Price = ({reg.coef_[0]:.2f} * Area) + "
          f"({reg.coef_[1]:.2f} * Bedrooms) + "
          f"({reg.coef_[2]:.2f} * Age) + {reg.intercept_:.2f}")

    # ---------------------------------------------------------
    # 5. Prediction Logic
    # ---------------------------------------------------------
    # Example: 3000 sq ft, 3 bedrooms, 40 years old
    new_home = [[3000, 3, 40]]
    prediction = reg.predict(new_home)
    
    print("-" * 30)
    print(f"🔮 Prediction for 3000sqft, 3 Bed, 40yo home: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    run_multivariate_lab()
