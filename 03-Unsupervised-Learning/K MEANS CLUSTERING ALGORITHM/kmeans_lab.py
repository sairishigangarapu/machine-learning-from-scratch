import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def run_kmeans_lab():
    # ---------------------------------------------------------
    # 1. Data Generation (Income vs Age)
    # ---------------------------------------------------------
    data = {
        'Name': ['Rob', 'Michael', 'Mohan', 'Ismail', 'Kory', 'Gautam', 'David', 'Andrea', 'Brad', 'Angelina', 'Donald', 'Tom', 'Arnold', 'Jared', 'Stark', 'Ranbir', 'Dipika', 'Piyanka', 'Nick', 'Alia', 'Sid', 'Abdul'],
        'Age': [27, 29, 29, 28, 42, 39, 41, 38, 36, 35, 37, 26, 27, 28, 29, 32, 40, 41, 43, 30, 41, 39],
        'Income': [70000, 90000, 61000, 60000, 150000, 155000, 160000, 162000, 156000, 130000, 137000, 45000, 48000, 51000, 49500, 53000, 65000, 63000, 64000, 80000, 82000, 58000]
    }
    df = pd.DataFrame(data)

    print("📊 Raw Data Preview:")
    print(df.head())
    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Preprocessing (Scaling is CRITICAL for K-Means)
    # ---------------------------------------------------------
    # Income (150,000) is huge compared to Age (40). 
    # Without scaling, Income dominates the Euclidean distance.
    scaler = MinMaxScaler()
    
    df_scaled = df.copy()
    df_scaled[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])
    
    print("✅ Data Scaled (0-1 Range)")

    # ---------------------------------------------------------
    # 3. The Elbow Method (Finding Optimal K)
    # ---------------------------------------------------------
    print("\n📉 Calculating SSE for Elbow Plot...")
    sse = []
    k_range = range(1, 10)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(df_scaled[['Age', 'Income']])
        sse.append(km.inertia_) # Inertia = Sum of Squared Errors

    # Plot Elbow
    plt.figure(figsize=(6, 4))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Error (SSE)')
    plt.title('The Elbow Method')
    plt.grid()
    plt.show()
    
    print("👉 Look at the plot. The 'Elbow' is at K=3.")

    # ---------------------------------------------------------
    # 4. Final Clustering (K=3)
    # ---------------------------------------------------------
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_predicted = km.fit_predict(df_scaled[['Age', 'Income']])
    
    df_scaled['cluster'] = y_predicted
    
    # Separate clusters for plotting
    df1 = df_scaled[df_scaled.cluster == 0]
    df2 = df_scaled[df_scaled.cluster == 1]
    df3 = df_scaled[df_scaled.cluster == 2]

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(df1.Age, df1.Income, color='green', label='Cluster 1')
    plt.scatter(df2.Age, df2.Income, color='red', label='Cluster 2')
    plt.scatter(df3.Age, df3.Income, color='black', label='Cluster 3')
    
    # Plot Centroids
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
                color='purple', marker='*', s=200, label='Centroids')
    
    plt.xlabel('Age (Scaled)')
    plt.ylabel('Income (Scaled)')
    plt.title('Customer Segments')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_kmeans_lab()
