
# EXPERIMENT: DATA CLEANING AND PREPARATION
# Dataset: Telecom Customer Churn

# --------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# --------------------------------------------------

import kagglehub
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 2. DOWNLOAD AND IMPORT DATASET
# --------------------------------------------------

path = kagglehub.dataset_download("blastchar/telco-customer-churn")

csv_files = glob.glob(os.path.join(path, "*.csv"))

df = pd.read_csv(csv_files[0])

print("Dataset loaded successfully!")
print(df.head())


# --------------------------------------------------
# 3. EXPLORE THE DATASET
# --------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Fill missing TotalCharges with 0 for customers
# having zero tenure
df.loc[df["tenure"] == 0, "TotalCharges"] = 0

# Fill remaining numerical missing values with median
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# 5. REMOVE DUPLICATE RECORDS
# --------------------------------------------------

df.drop_duplicates(inplace=True)

print("\nDataset shape after removing duplicates:")
print(df.shape)


# --------------------------------------------------
# 6. HANDLE INCONSISTENT DATA
# --------------------------------------------------

# Remove unwanted spaces from categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Standardize Yes/No values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace({
        "yes": "Yes",
        "no": "No",
        "YES": "Yes",
        "NO": "No"
    })

print("\nInconsistent values standardized.")


# --------------------------------------------------
# 7. CONVERT DATA TYPES
# --------------------------------------------------

df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
df["tenure"] = df["tenure"].astype(int)

df["MonthlyCharges"] = pd.to_numeric(
    df["MonthlyCharges"]
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"]
)

print("\nUpdated Data Types:")
print(df.dtypes)


# --------------------------------------------------
# 8. IDENTIFY AND HANDLE OUTLIERS
# --------------------------------------------------

# Boxplot before handling outliers
plt.figure(figsize=(10, 5))
df[["MonthlyCharges", "TotalCharges"]].boxplot()
plt.title("Boxplot Before Outlier Handling")
plt.show()

# IQR method
for col in ["MonthlyCharges", "TotalCharges"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    print(f"\nOutliers in {col}: {outliers}")

    # Cap outliers using IQR limits
    df[col] = df[col].clip(lower, upper)


# Boxplot after handling outliers
plt.figure(figsize=(10, 5))
df[["MonthlyCharges", "TotalCharges"]].boxplot()
plt.title("Boxplot After Outlier Handling")
plt.show()


# --------------------------------------------------
# 9. FEATURE ENGINEERING
# --------------------------------------------------

# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Medium", "Long", "Very Long"]
)

# Calculate average monthly spending
df["AverageMonthlySpend"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)

print("\nNew Features:")
print(df[["TenureGroup", "AverageMonthlySpend"]].head())


# --------------------------------------------------
# 10. ENCODE TARGET VARIABLE
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nChurn Values:")
print(df["Churn"].value_counts())


# --------------------------------------------------
# 11. SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

# CustomerID is not useful for prediction
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# --------------------------------------------------
# 12. NORMALIZE AND ENCODE DATA
# --------------------------------------------------

# Identify numerical and categorical columns
num_features = X_train.select_dtypes(
    include=np.number
).columns

cat_features = X_train.select_dtypes(
    exclude=np.number
).columns

# Numerical preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Fit on training data and transform both sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data Shape:",
      X_train_processed.shape)

print("Processed Testing Data Shape:",
      X_test_processed.shape)


# --------------------------------------------------
# 13. EXPORT CLEANED DATASET
# --------------------------------------------------

df.to_csv("Telecom_Customer_Churn_Cleaned.csv",
          index=False)

print("\nCleaned dataset exported successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nData Cleaning and Preparation Completed!")
