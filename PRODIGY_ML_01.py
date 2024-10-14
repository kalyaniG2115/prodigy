import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = r'D:/ML data sets/train.csv'  # Adjust this path if needed
data = pd.read_csv(file_path)

# Step 2: Explore the data (Optional)
print("First few rows of the dataset:")
print(data.head())

# Step 3: Select features for clustering
X = data[['LotArea', 'SalePrice']]

# Step 4: Handle missing values (if any)
X = X.fillna(X.mean())  # Fill missing values with column means

# Step 5: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Step 7: Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Step 8: Visualize the Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('Housing Clusters based on Lot Area and Sale Price')
plt.xlabel('Lot Area (scaled)')
plt.ylabel('Sale Price (scaled)')
plt.show()