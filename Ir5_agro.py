
import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


# Step 1: Download and Load Dataset

path = kagglehub.dataset_download("arjunbhasin2013/ccdata")

print("Path to dataset files:", path)

X = pd.read_csv(os.path.join(path, "CC GENERAL.csv"))

print(X.head())
print("Dataset Shape:", X.shape)


# Step 2: Cleaning the Data

# Drop customer ID column
X = X.drop('CUST_ID', axis=1)

# Handle missing values
X.fillna(method='ffill', inplace=True)

print("Missing Values:\n", X.isnull().sum())


# Step 3: Preprocessing the Data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalize the data
X_normalized = normalize(X_scaled)

# Convert numpy array into DataFrame
X_normalized = pd.DataFrame(X_normalized)


# Step 4: Dimensionality Reduction using PCA

pca = PCA(n_components=2)

X_principal = pca.fit_transform(X_normalized)

X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

print("Principal Components:")
print(X_principal.head())


# Step 5: Visualizing the Dendrogram

plt.figure(figsize=(8, 8))
plt.title('Visualising the data')

dendrogram = shc.dendrogram(
    shc.linkage(X_principal, method='ward'),
    truncate_mode='lastp',
    p=30
)

plt.xlabel('Data Points / Clusters')
plt.ylabel('Euclidean Distance')
plt.show()


# Step 6: Building and Visualizing Clustering Models

# k = 2
ac2 = AgglomerativeClustering(n_clusters=2)

plt.figure(figsize=(6, 6))
plt.scatter(
    X_principal['P1'],
    X_principal['P2'],
    c=ac2.fit_predict(X_principal),
    cmap='rainbow'
)
plt.title('Agglomerative Clustering (k=2)')
plt.xlabel('P1')
plt.ylabel('P2')
plt.show()


# k = 3
ac3 = AgglomerativeClustering(n_clusters=3)

plt.figure(figsize=(6, 6))
plt.scatter(
    X_principal['P1'],
    X_principal['P2'],
    c=ac3.fit_predict(X_principal),
    cmap='rainbow'
)
plt.title('Agglomerative Clustering (k=3)')
plt.xlabel('P1')
plt.ylabel('P2')
plt.show()


# k = 4
ac4 = AgglomerativeClustering(n_clusters=4)

plt.figure(figsize=(6, 6))
plt.scatter(
    X_principal['P1'],
    X_principal['P2'],
    c=ac4.fit_predict(X_principal),
    cmap='rainbow'
)
plt.title('Agglomerative Clustering (k=4)')
plt.xlabel('P1')
plt.ylabel('P2')
plt.show()


# k = 5
ac5 = AgglomerativeClustering(n_clusters=5)

plt.figure(figsize=(6, 6))
plt.scatter(
    X_principal['P1'],
    X_principal['P2'],
    c=ac5.fit_predict(X_principal),
    cmap='rainbow'
)
plt.title('Agglomerative Clustering (k=5)')
plt.xlabel('P1')
plt.ylabel('P2')
plt.show()


# k = 6
ac6 = AgglomerativeClustering(n_clusters=6)

plt.figure(figsize=(6, 6))
plt.scatter(
    X_principal['P1'],
    X_principal['P2'],
    c=ac6.fit_predict(X_principal),
    cmap='rainbow'
)
plt.title('Agglomerative Clustering (k=6)')
plt.xlabel('P1')
plt.ylabel('P2')
plt.show()


# Step 7: Evaluating Models using Silhouette Score

k = [2, 3, 4, 5, 6]

silhouette_scores = []

silhouette_scores.append(
    silhouette_score(X_principal, ac2.labels_)
)

silhouette_scores.append(
    silhouette_score(X_principal, ac3.labels_)
)

silhouette_scores.append(
    silhouette_score(X_principal, ac4.labels_)
)

silhouette_scores.append(
    silhouette_score(X_principal, ac5.labels_)
)

silhouette_scores.append(
    silhouette_score(X_principal, ac6.labels_)
)


# Display Silhouette Scores
for i in range(len(k)):
    print("k =", k[i], "Silhouette Score =", silhouette_scores[i])


# Plotting Bar Graph
plt.figure(figsize=(8, 5))

plt.bar(k, silhouette_scores)

plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Comparison of Silhouette Scores')

plt.show()


# Finding Optimal Number of Clusters
optimal_k = k[np.argmax(silhouette_scores)]

print("Optimal number of clusters:", optimal_k)
