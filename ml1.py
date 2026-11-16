
# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the dataset
df = pd.read_csv("Wine.csv")

# Separate features and target
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 2. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=3)
x_pca = pca.fit_transform(X_scaled)

# 4. Display explained variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTransformed Dataset:")
print(x_pca)

# 5. Visualize first two principal components
plt.figure(figsize=(8, 6))

plt.scatter(
    x_pca[:, 0],
    x_pca[:, 1],
    c=y,
    cmap='viridis'
)

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA on Wine Dataset")

plt.colorbar(label="Wine Class")
plt.show()
