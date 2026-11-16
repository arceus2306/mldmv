
import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load the dataset
data = pd.read_csv("City_Air_Quality.csv")

# Task 2: Explore the dataset
print("First 5 rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nStatistical Description:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

# Task 3: Identify relevant variables
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

print("\nAvailable Columns:")
print(data.columns)

# Task 4: Line plot for overall AQI trend
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['AQI'], color='blue', marker='o', label='AQI')
plt.xlabel("Date")
plt.ylabel("AQI Value")
plt.title("Overall AQI Trend Over Time")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 5: Plot individual pollutant levels
pollutants = ['PM2.5', 'PM10', 'CO']

for pollutant in pollutants:
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data[pollutant], marker='o', label=pollutant)
    plt.xlabel("Date")
    plt.ylabel("Pollutant Level")
    plt.title(pollutant + " Levels Over Time")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Task 6: Bar plot to compare AQI values
plt.figure(figsize=(10, 5))
plt.bar(data['Date'].astype(str), data['AQI'], color='orange')
plt.xlabel("Date")
plt.ylabel("AQI Value")
plt.title("AQI Comparison Across Dates")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 7: Box plot to analyze AQI distribution
plt.figure(figsize=(8, 5))
plt.boxplot(data['AQI'].dropna())
plt.ylabel("AQI Value")
plt.title("Distribution of AQI Values")
plt.grid(True)
plt.show()

# Task 8: Scatter plots to explore relationships
for pollutant in pollutants:
    plt.figure(figsize=(8, 5))
    plt.scatter(data[pollutant], data['AQI'], color='green')
    plt.xlabel(pollutant + " Level")
    plt.ylabel("AQI Value")
    plt.title("AQI vs " + pollutant)
    plt.grid(True)
    plt.show()

# Task 9: Customize visualizations
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['AQI'], color='red',
         marker='o', linestyle='-', linewidth=2,
         label='Air Quality Index')

plt.xlabel("Date", fontsize=12)
plt.ylabel("AQI", fontsize=12)
plt.title("Customized AQI Trend Analysis", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
