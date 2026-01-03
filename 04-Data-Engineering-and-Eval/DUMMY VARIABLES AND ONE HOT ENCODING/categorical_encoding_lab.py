import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def run_encoding_lab():
    # ---------------------------------------------------------
    # 0. Data Generation (Dummy Data)
    # ---------------------------------------------------------
    data = {
        'town': ['monroe township', 'monroe township', 'monroe township', 
                 'west windsor', 'west windsor', 'west windsor', 
                 'robbinsville', 'robbinsville', 'robbinsville'],
        'area': [2600, 3000, 3200, 2600, 2800, 3300, 2600, 2900, 3100],
        'price': [550000, 565000, 610000, 585000, 615000, 650000, 575000, 600000, 620000]
    }
    df = pd.DataFrame(data)
    print("📊 Raw Dataset:")
    print(df)
    print("-" * 30)

    # ---------------------------------------------------------
    # Method 1: Pandas Get Dummies (The Quick Way)
    # ---------------------------------------------------------
    print("\n🔹 Method 1: Pandas get_dummies")
    
    # Step 1: Create Dummies
    dummies = pd.get_dummies(df['town'])
    
    # Step 2: Concatenate and Drop 'town' + Drop First Dummy (Trap Evasion)
    # We drop 'west windsor' to avoid the Dummy Variable Trap
    df_pandas = pd.concat([df, dummies], axis='columns')
    df_pandas = df_pandas.drop(['town', 'west windsor'], axis='columns')
    
    print("Processed DataFrame (Traps Removed):")
    print(df_pandas.head())
    
    # Train Model
    X = df_pandas.drop('price', axis='columns')
    y = df_pandas['price']
    
    model_pd = LinearRegression()
    model_pd.fit(X, y)
    
    # Predict: 2800 sqft in Robbinsville
    # Input format: [area, monroe, robbinsville]
    pred = model_pd.predict([[2800, 0, 1]]) 
    print(f"Prediction (2800sqft, Robbinsville): ${pred[0]:,.2f}")


    # ---------------------------------------------------------
    # Method 2: Scikit-Learn OneHotEncoder (The Pipeline Way)
    # ---------------------------------------------------------
    print("\n🔹 Method 2: Sklearn ColumnTransformer (Production Ready)")
    
    # We want to transform 'town' (index 0) and keep 'area' (index 1)
    # drop='first' handles the Dummy Variable Trap automatically
    ct = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(drop='first'), ['town'])
        ],
        remainder='passthrough' # Leave 'area' alone
    )
    
    X_raw = df[['town', 'area']]
    y_raw = df['price']
    
    X_transformed = ct.fit_transform(X_raw)
    
    # Train Model
    model_sk = LinearRegression()
    model_sk.fit(X_transformed, y_raw)
    
    print(f"Model Score (R^2): {model_sk.score(X_transformed, y_raw):.4f}")


    # ---------------------------------------------------------
    # Method 3: Label Encoding (For Ordinal Data Only!)
    # ---------------------------------------------------------
    print("\n🔹 Method 3: Label Encoding (Warning: Implies Order)")
    
    df_le = df.copy()
    le = LabelEncoder()
    
    # Assigns 0, 1, 2 to towns. 
    # WARNING: Model might think Town 2 > Town 1, which is wrong for nominal data.
    df_le['town_encoded'] = le.fit_transform(df_le['town'])
    print(df_le[['town', 'town_encoded']].drop_duplicates())

if __name__ == "__main__":
    run_encoding_lab()
