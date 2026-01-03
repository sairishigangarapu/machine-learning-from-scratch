import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.linear_model import LinearRegression

def run_persistence_lab():
    # ---------------------------------------------------------
    # 1. Data Generation & Training
    # ---------------------------------------------------------
    # Dummy data: Area (sq ft) vs Price
    data = {
        'area': [2600, 3000, 3200, 3600, 4000],
        'price': [550000, 565000, 610000, 680000, 725000]
    }
    df = pd.DataFrame(data)
    
    # Train Model
    model = LinearRegression()
    model.fit(df[['area']], df['price'])
    
    print("🏗️ Model Trained!")
    print(f"   Slope: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")

    # Test Prediction (Area = 3300)
    test_input = [[3300]]
    original_pred = model.predict(test_input)[0]
    print(f"   Original Prediction (3300 sqft): ${original_pred:,.2f}")
    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Method A: Pickle 🥒
    # ---------------------------------------------------------
    print("\n📦 Method A: Pickle")
    
    # Save (Dump) - 'wb' means Write Binary
    with open('model_pickle.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   ✅ Model saved to 'model_pickle.pkl'")
    
    # Load - 'rb' means Read Binary
    with open('model_pickle.pkl', 'rb') as f:
        loaded_pickle = pickle.load(f)
        
    pickle_pred = loaded_pickle.predict(test_input)[0]
    print(f"   ♻️ Loaded Prediction: ${pickle_pred:,.2f}")

    # ---------------------------------------------------------
    # 3. Method B: Joblib ⚡ (Recommended for sklearn)
    # ---------------------------------------------------------
    print("\n⚡ Method B: Joblib")
    
    # Save
    joblib.dump(model, 'model_joblib.pkl')
    print("   ✅ Model saved to 'model_joblib.pkl'")
    
    # Load
    loaded_joblib = joblib.load('model_joblib.pkl')
    
    joblib_pred = loaded_joblib.predict(test_input)[0]
    print(f"   ♻️ Loaded Prediction: ${joblib_pred:,.2f}")

    # ---------------------------------------------------------
    # 4. Verification
    # ---------------------------------------------------------
    if original_pred == pickle_pred == joblib_pred:
        print("\n🎉 Success! All models returned identical predictions.")
    else:
        print("\n❌ Error: Predictions do not match!")

if __name__ == "__main__":
    run_persistence_lab()
