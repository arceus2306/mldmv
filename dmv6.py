
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("Retail_Sales_Data.csv")

# 2. Explore the dataset
print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Select relevant columns
print("\nRelevant Columns:")
print(df[['Region', 'Sales_Amount', 'Product_Category']].head())

# 4. Group data by Region and calculate total sales
region_sales = df.groupby('Region')['Sales_Amount'].sum()

print("\nTotal Sales by Region:")
print(region_sales)

# 5. Visualize sales distribution by Region
plt.figure(figsize=(8, 5))
region_sales.plot(kind='bar', color='skyblue')
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart showing sales distribution
plt.figure(figsize=(7, 7))
region_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Sales Distribution by Region")
plt.ylabel("")
plt.show()

# 6. Identify top-performing regions
top_regions = region_sales.sort_values(ascending=False)

print("\nTop Performing Regions:")
print(top_regions)

print("\nBest Performing Region:", top_regions.index[0])
print("Highest Sales:", top_regions.iloc[0])

# 7. Group sales by Region and Product Category
category_sales = df.groupby(
    ['Region', 'Product_Category']
)['Sales_Amount'].sum().reset_index()

print("\nSales by Region and Product Category:")
print(category_sales)

# 8. Visualize sales using a stacked bar chart
pivot_data = category_sales.pivot(
    index='Region',
    columns='Product_Category',
    values='Sales_Amount'
)

pivot_data.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title("Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.legend(title="Product Category")
plt.tight_layout()
plt.show()

# Grouped bar chart
pivot_data.plot(kind='bar', stacked=False, figsize=(10, 6))

plt.title("Comparison of Product Sales Across Regions")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.legend(title="Product Category")
plt.tight_layout()
plt.show()
