
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import the dataset
df = pd.read_csv("RealEstate_Prices.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:", df.shape)

# 2. Clean column names
# Convert column names to lowercase and replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\nCleaned Column Names:")
print(df.columns)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Replace missing numerical values with their column median
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 3. Data Merging
# Create an additional lookup dataset for ocean proximity
location_data = pd.DataFrame({
    "ocean_proximity": ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
    "location_type": ["Bay Area", "Near Ocean", "Inland Area", "Coastal Area", "Island Area"]
})

# Merge both datasets using the common column
df = pd.merge(df, location_data, on="ocean_proximity", how="left")

print("\nMerged Dataset:")
print(df.head())

# 4. Filter and subset the data
# Select houses with median income greater than 5
high_income = df[df["median_income"] > 5]

print("\nProperties with Median Income > 5:")
print(high_income.head())

# Select properties near the ocean
ocean_houses = df[df["ocean_proximity"] == "NEAR OCEAN"]

print("\nProperties Near Ocean:")
print(ocean_houses.head())

# Select only required columns
subset = df[["longitude", "latitude", "median_income", "median_house_value"]]

print("\nSelected Columns:")
print(subset.head())

# 5. Handle categorical variables using One-Hot Encoding
df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

print("\nDataset After One-Hot Encoding:")
print(df.head())

# 6. Data Aggregation
# Calculate average house value for each location type
# Use the original location information
df["location_type"] = df["location_type"].fillna("Unknown")

average_price = df.groupby("location_type")["median_house_value"].mean()

print("\nAverage House Price by Location:")
print(average_price)

# Create new features
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

print("\nDataset After Feature Engineering:")
print(df[["rooms_per_household", "bedrooms_per_room"]].head())

# 7. Identify and Handle Outliers using IQR method

# Calculate Q1 and Q3 for median house value
Q1 = df["median_house_value"].quantile(0.25)
Q3 = df["median_house_value"].quantile(0.75)

IQR = Q3 - Q1

# Calculate lower and upper limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Count outliers
outliers = df[
    (df["median_house_value"] < lower_limit) |
    (df["median_house_value"] > upper_limit)
]

print("\nNumber of Outliers:", len(outliers))

# Replace extreme values with the calculated limits (Capping)
df["median_house_value"] = df["median_house_value"].clip(
    lower=lower_limit,
    upper=upper_limit
)

print("\nOutliers Handled Successfully!")

# Data Visualization
plt.figure(figsize=(8, 5))
sns.histplot(df["median_house_value"], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Median House Value")
plt.show()

# Boxplot to visualize outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["median_house_value"])
plt.title("Boxplot of House Prices")
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Save the cleaned dataset
df.to_csv("Cleaned_RealEstate_Prices.csv", index=False)

print("\nFinal Dataset Shape:", df.shape)
print("\nCleaned Dataset Saved Successfully!")
