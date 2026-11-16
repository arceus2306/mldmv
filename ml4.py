
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = pd.read_csv('Iris.csv')

# 2. Data understanding
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# 3. Prepare features
X = data.drop(['Id', 'Species'], axis=1)

# 4. Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5. Elbow method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 6. Plot elbow graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 7. Apply K-Means with K=3
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# 8. Visualize clusters
plt.scatter(X[y_kmeans == 0, 0],
            X[y_kmeans == 0, 1],
            s=100, c='red', label='Cluster 1')

plt.scatter(X[y_kmeans == 1, 0],
            X[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')

plt.scatter(X[y_kmeans == 2, 0],
            X[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='yellow',
            marker='X', label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Sepal Length (Scaled)')
plt.ylabel('Sepal Width (Scaled)')
plt.legend()
plt.show()

