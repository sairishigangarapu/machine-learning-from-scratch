import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def run_split_lab():
    # ---------------------------------------------------------
    # 1. Data Generation
    # ---------------------------------------------------------
    data = {
        'Mileage': [69000, 35000, 57000, 22500, 46000, 59000, 52000, 72000, 91000, 67000, 83000, 79000, 55000, 48000, 30000],
        'Age': [6, 3, 5, 2, 4, 5, 5, 6, 8, 6, 7, 7, 5, 4, 3],
        'Price': [18000, 34000, 26100, 40000, 31500, 26750, 32000, 19300, 12000, 22000, 18000, 21000, 28000, 30500, 36000]
    }
    df = pd.DataFrame(data)

    print("📊 Raw Dataset:")
    print(df.head())
    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Splitting the Data
    # ---------------------------------------------------------
    X = df[['Mileage', 'Age']]
    y = df['Price']

    # 80% Train, 20% Test, Fixed Random Seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"✅ Data Split Complete.")
    print(f"Training Set Size: {len(X_train)} samples")
    print(f"Test Set Size:     {len(X_test)} samples")

    # ---------------------------------------------------------
    # 3. Visualization (The "Aha!" Moment)
    # ---------------------------------------------------------
    # We visualize the split to show that it is a random sample
    plt.figure(figsize=(8, 6))
    
    # Plot Training Data
    plt.scatter(X_train['Mileage'], y_train, color='blue', label='Training Data (80%)', marker='o')
    
    # Plot Test Data
    plt.scatter(X_test['Mileage'], y_test, color='red', label='Test Data (20%)', marker='x', s=100)
    
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Visualizing Train/Test Split (Random Sampling)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------------
    # 4. Model Training & Evaluation
    # ---------------------------------------------------------
    model = LinearRegression()
    
    # Train ONLY on Training Data
    model.fit(X_train, y_train)
    
    # Predict on Test Data (Simulating the future)
    predictions = model.predict(X_test)
    
    print("-" * 30)
    print("🔮 Predictions on Test Data:")
    for i in range(len(predictions)):
        actual = y_test.iloc[i]
        pred = predictions[i]
        print(f" - Actual: ${actual} | Predicted: ${pred:.0f} | Diff: ${pred-actual:.0f}")

    score = model.score(X_test, y_test)
    print(f"\n📈 Model R^2 Score on Test Set: {score:.4f}")

if __name__ == "__main__":
    run_split_lab()
