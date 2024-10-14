

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset with the corrected file path
file_path = r'D:\ML data sets\Mall_Customers.csv'  # Corrected file path
data = pd.read_csv(file_path)

# Step 2: Explore the data (Optional)
print("First few rows of the dataset:")
print(data.head())

# Step 3: Select the features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Step 6: Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Step 7: Visualize the Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()