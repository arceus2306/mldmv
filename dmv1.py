import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# ==========================================
# 1. CREATE SAMPLE SALES DATA
# ==========================================

os.makedirs("sales_data", exist_ok=True)
os.makedirs("sales_output", exist_ok=True)

data = [
    [1001, "2026-01-05", "Laptop", "Electronics", "West", 2, 55000],
    [1002, "2026-01-08", "Mouse", "Accessories", "North", 5, 500],
    [1003, "2026-01-12", "Chair", "Furniture", "South", 3, 4500],
    [1004, "2026-02-02", "Laptop", "Electronics", "East", 1, 55000],
    [1005, "2026-02-10", "Keyboard", "Accessories", "West", 4, 1200],
    [1006, "2026-02-15", "Desk", "Furniture", "North", 2, 8000],
    [1007, "2026-03-01", "Phone", "Electronics", "South", 2, 30000],
    [1008, "2026-03-09", "Mouse", "Accessories", "East", 3, 500],
    [1009, "2026-03-18", "Chair", "Furniture", "West", 1, 4500]
]

columns = [
    "order_id", "order_date", "product",
    "category", "region", "quantity", "unit_price"
]

df = pd.DataFrame(data, columns=columns)

# Add a few data quality issues for demonstration
df.loc[1, "region"] = " north "
df.loc[4, "category"] = "accessories"
df.loc[6, "quantity"] = None

# ==========================================
# 2. SAVE DATA IN DIFFERENT FILE FORMATS
# ==========================================

# CSV file
df.iloc[:3].to_csv(
    "sales_data/sales1.csv", index=False
)

# Excel file
df.iloc[3:6].to_excel(
    "sales_data/sales2.xlsx", index=False
)

# JSON file
df.iloc[6:].to_json(
    "sales_data/sales3.json",
    orient="records",
    indent=4
)

print("Sample datasets created successfully!")

# ==========================================
# 3. LOAD DATA FROM DIFFERENT FORMATS
# ==========================================

df1 = pd.read_csv("sales_data/sales1.csv")

df2 = pd.read_excel("sales_data/sales2.xlsx")

with open("sales_data/sales3.json", "r") as file:
    data_json = json.load(file)

df3 = pd.DataFrame(data_json)

print("\nCSV DATA:")
print(df1)

print("\nEXCEL DATA:")
print(df2)

print("\nJSON DATA:")
print(df3)

# ==========================================
# 4. EXPLORE THE DATA
# ==========================================

print("\nCSV DATA INFORMATION:")
print(df1.info())

print("\nSTATISTICAL SUMMARY:")
print(df1.describe())

print("\nMISSING VALUES:")
print(df1.isnull().sum())

print("\nDUPLICATE VALUES:")
print(df1.duplicated().sum())

# ==========================================
# 5. COMBINE ALL DATASETS
# ==========================================

df = pd.concat(
    [df1, df2, df3],
    ignore_index=True
)

print("\nCOMBINED DATA:")
print(df)

# ==========================================
# 6. DATA CLEANING
# ==========================================

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Remove extra spaces and standardize capitalization
for col in ["product", "category", "region"]:
    df[col] = df[col].astype("string").str.strip().str.title()

# Convert date into datetime format
df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)

# Convert numeric columns
df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
)

df["unit_price"] = pd.to_numeric(
    df["unit_price"], errors="coerce"
)

# Remove rows with missing essential information
df = df.dropna(subset=["order_id", "order_date"])

# Handle missing numerical values using median
df["quantity"] = df["quantity"].fillna(
    df["quantity"].median()
)

df["unit_price"] = df["unit_price"].fillna(
    df["unit_price"].median()
)

# Handle missing text values
for col in ["product", "category", "region"]:
    df[col] = df[col].fillna("Unknown")

# Remove duplicate orders
df = df.drop_duplicates(
    subset=["order_id"], keep="first"
)

# Remove invalid negative values
df = df[
    (df["quantity"] >= 0) &
    (df["unit_price"] >= 0)
].copy()

print("\nCLEANED DATA:")
print(df)

# ==========================================
# 7. DATA TRANSFORMATION
# ==========================================

# Calculate total sales
df["total_sales"] = (
    df["quantity"] * df["unit_price"]
)

# Extract month from date
df["month"] = df["order_date"].dt.month

print("\nTRANSFORMED DATA:")
print(df)

# ==========================================
# 8. DATA ANALYSIS
# ==========================================

# Total sales
total_sales = df["total_sales"].sum()

print("\nTOTAL SALES:", total_sales)

# Number of orders
total_orders = df["order_id"].nunique()

print("TOTAL ORDERS:", total_orders)

# Average order value
average_order = total_sales / total_orders

print("AVERAGE ORDER VALUE:", average_order)

# Category-wise sales
category_sales = df.groupby(
    "category"
)["total_sales"].sum()

print("\nCATEGORY-WISE SALES:")
print(category_sales)

# Region-wise sales
region_sales = df.groupby(
    "region"
)["total_sales"].sum()

print("\nREGION-WISE SALES:")
print(region_sales)

# Monthly sales
monthly_sales = df.groupby(
    "month"
)["total_sales"].sum()

print("\nMONTHLY SALES:")
print(monthly_sales)

# Descriptive statistics
print("\nDESCRIPTIVE STATISTICS:")
print(df["total_sales"].describe())

# ==========================================
# 9. BAR CHART
# ==========================================

category_sales.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_output/bar_chart.png")
plt.show()

# ==========================================
# 10. PIE CHART
# ==========================================

category_sales.plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(7, 7)
)

plt.title("Sales Distribution by Category")
plt.ylabel("")
plt.tight_layout()
plt.savefig("sales_output/pie_chart.png")
plt.show()

# ==========================================
# 11. LINE CHART
# ==========================================

monthly_sales.plot(
    kind="line",
    marker="o",
    figsize=(8, 5)
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_output/line_chart.png")
plt.show()

# ==========================================
# 12. BOX PLOT
# ==========================================

df.boxplot(
    column="total_sales",
    by="category",
    figsize=(8, 5)
)

plt.title("Sales Distribution by Category")
plt.suptitle("")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_output/box_plot.png")
plt.show()

# ==========================================
# 13. SIMPLE SALES FORECASTING
# ==========================================

x = np.arange(len(monthly_sales))
y = monthly_sales.values

if len(monthly_sales) >= 2:

    slope, intercept = np.polyfit(x, y, 1)

    next_month_sales = (
        slope * len(monthly_sales) + intercept
    )

    print(
        "\nPREDICTED NEXT MONTH SALES:",
        max(0, next_month_sales)
    )

else:
    print("\nNot enough monthly data for forecasting.")

# ==========================================
# 14. SAVE FINAL CLEANED DATA
# ==========================================

df.to_csv(
    "sales_output/cleaned_sales.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print("\nALL OPERATIONS COMPLETED SUCCESSFULLY!")
