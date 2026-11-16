
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("uber.csv")

print("First five rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# 2. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())


# 3. Handle Missing Values
df = df.dropna(subset=[
    'fare_amount',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'pickup_datetime'
])


# 4. Convert Datetime
df['pickup_datetime'] = pd.to_datetime(
    df['pickup_datetime'], errors='coerce'
)

df = df.dropna(subset=['pickup_datetime'])

df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year


# 5. Remove Unnecessary Columns
df = df.drop(
    ['key', 'pickup_datetime', 'Unnamed: 0'],
    axis=1,
    errors='ignore'
)


# 6. Remove Invalid Values
df = df[
    (df['fare_amount'] > 0) &
    (df['passenger_count'] > 0) &
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90)) &
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180))
].copy()


# 7. Detect Outliers using IQR
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("\nLower Bound:", lower)
print("Upper Bound:", upper)

# Boxplot before removing outliers
sns.boxplot(x=df['fare_amount'])
plt.title("Fare Amount Outliers")
plt.show()

# Remove Outliers
df = df[
    (df['fare_amount'] >= lower) &
    (df['fare_amount'] <= upper)
].copy()


# 8. Calculate Distance using Haversine Formula
def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * \
        np.sin(dlon / 2)**2

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


df['distance'] = haversine(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)

# Remove geographical coordinate columns
df = df.drop([
    'pickup_latitude',
    'pickup_longitude',
    'dropoff_latitude',
    'dropoff_longitude'
], axis=1)

# Remove zero-distance rides
df = df[df['distance'] > 0].copy()


# 9. Correlation
corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# 10. Separate Independent and Dependent Variables
X = df.drop('fare_amount', axis=1)
Y = df['fare_amount']


# 11. Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# 12. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 13. Linear Regression
lr = LinearRegression()

lr.fit(X_train, Y_train)

Y_pred_lr = lr.predict(X_test)


# 14. Ridge Regression
ridge = Ridge(alpha=1.0)

ridge.fit(X_train, Y_train)

Y_pred_ridge = ridge.predict(X_test)


# 15. Lasso Regression
lasso = Lasso(alpha=0.1, max_iter=10000)

lasso.fit(X_train, Y_train)

Y_pred_lasso = lasso.predict(X_test)


# 16. Model Evaluation Function
def evaluate_model(Y_test, Y_pred):

    mae = mean_absolute_error(Y_test, Y_pred)

    mse = mean_squared_error(Y_test, Y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(Y_test, Y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# 17. Evaluate All Models
print("\n------ Linear Regression ------")
evaluate_model(Y_test, Y_pred_lr)

print("\n------ Ridge Regression ------")
evaluate_model(Y_test, Y_pred_ridge)

print("\n------ Lasso Regression ------")
evaluate_model(Y_test, Y_pred_lasso)


# 18. Compare All Models
results = pd.DataFrame({

    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],

    'MAE': [
        mean_absolute_error(Y_test, Y_pred_lr),
        mean_absolute_error(Y_test, Y_pred_ridge),
        mean_absolute_error(Y_test, Y_pred_lasso)
    ],

    'RMSE': [
        np.sqrt(mean_squared_error(Y_test, Y_pred_lr)),
        np.sqrt(mean_squared_error(Y_test, Y_pred_ridge)),
        np.sqrt(mean_squared_error(Y_test, Y_pred_lasso))
    ],

    'R2 Score': [
        r2_score(Y_test, Y_pred_lr),
        r2_score(Y_test, Y_pred_ridge),
        r2_score(Y_test, Y_pred_lasso)
    ]
})

print("\n------ Model Comparison ------")
print(results)


# 19. Actual vs Predicted Visualization
plt.figure(figsize=(8, 5))

plt.scatter(Y_test, Y_pred_lr, alpha=0.3)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")

plt.title("Actual vs Predicted Fare - Linear Regression")

plt.show()
