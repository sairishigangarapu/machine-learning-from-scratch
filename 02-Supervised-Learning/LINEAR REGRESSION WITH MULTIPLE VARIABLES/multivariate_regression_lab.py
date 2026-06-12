import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def run_multivariate_lab():
    # ---------------------------------------------------------
    # 1. Data Preparation (With Missing Values)
    # ---------------------------------------------------------
    data = {
        'area': [2600, 3000, 3200, 3600, 4000, 4100, 4500, 4800, 5000, 5200],
        'bedrooms': [3, 4, np.nan, 3, 5, 6, 4, 5, 6, 4],
        'age': [20, 15, 18, 30, 8, 8, 12, 6, 5, 10],
        'price': [550000, 565000, 610000, 595000, 760000, 810000, 830000, 890000, 920000, 870000]
    }
    df = pd.DataFrame(data)

    print("📊 Raw Data (Note the NaN value):")
    print(df)
    print("-" * 40)

    # ---------------------------------------------------------
    # 2. Data Cleaning (Imputation)
    # ---------------------------------------------------------
    median_bedrooms = math.floor(df['bedrooms'].median())
    df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)
    
    print(f"🧹 Data Cleaned! Replaced NaN with Median ({median_bedrooms}):")
    print(df)
    print("-" * 40)

    # ---------------------------------------------------------
    # 3. Train/Test Split
    # ---------------------------------------------------------
    features = ['area', 'bedrooms', 'age']
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['price'], test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    print("\n✅ Model Trained!")
    print(f"Coefficients (Weights): {dict(zip(features, reg.coef_))}")
    print(f"Intercept (Bias): {reg.intercept_:.2f}")
    
    terms = " + ".join(f"({reg.coef_[i]:.2f} * {features[i]})" for i in range(len(features)))
    print(f"\nFormula: Price = {terms} + {reg.intercept_:.2f}")

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n📈 Test Set Evaluation:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MSE:       {mse:.2f}")
    print(f"  RMSE:      ${rmse:,.2f}")

    # ---------------------------------------------------------
    # 6. Prediction
    # ---------------------------------------------------------
    new_home = [[3000, 3, 40]]
    prediction = reg.predict(new_home)
    
    print("-" * 40)
    print(f"🔮 Prediction for 3000sqft, 3 Bed, 40yo home: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    run_multivariate_lab()
