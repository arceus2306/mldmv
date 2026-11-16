import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# ---------------------------------------
# 1. API CONFIGURATION
# ---------------------------------------

API_KEY = "6a58b56efe7693d81951f8b3f9cde44c"

CITY = "Pune"
COUNTRY = "IN"

URL = "https://api.openweathermap.org/data/2.5/forecast"

params = {
    "q": f"{CITY},{COUNTRY}",
    "appid": API_KEY,
    "units": "metric"
}

# ---------------------------------------
# 2. FETCH WEATHER DATA
# ---------------------------------------

response = requests.get(URL, params=params)

if response.status_code != 200:
    print("Error fetching data:", response.json())
    raise SystemExit

data = response.json()

print("Weather data retrieved successfully!")

# ---------------------------------------
# 3. EXTRACT RELEVANT ATTRIBUTES
# ---------------------------------------

weather_list = []

for item in data["list"]:

    weather_list.append({
        "Date": item["dt_txt"],
        "Temperature": item["main"]["temp"],
        "Humidity": item["main"]["humidity"],
        "Wind Speed": item["wind"]["speed"],
        "Precipitation": (
            item.get("rain", {}).get("3h", 0) +
            item.get("snow", {}).get("3h", 0)
        ),
        "Weather": item["weather"][0]["description"],
        "Cloudiness": item["clouds"]["all"]
    })

df = pd.DataFrame(weather_list)

print("\nOriginal Weather Data:")
print(df.head())

# ---------------------------------------
# 4. DATA CLEANING AND PREPROCESSING
# ---------------------------------------

# Convert date column into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing numerical values
numeric_cols = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation",
    "Cloudiness"
]

df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].mean()
)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Sort data by date
df.sort_values(by="Date", inplace=True)

print("\nCleaned Weather Data:")
print(df.head())

# ---------------------------------------
# 5. DATA MODELING AND STATISTICAL ANALYSIS
# ---------------------------------------

print("\nWeather Statistics:")

print("Average Temperature:",
      df["Temperature"].mean())

print("Maximum Temperature:",
      df["Temperature"].max())

print("Minimum Temperature:",
      df["Temperature"].min())

print("Average Humidity:",
      df["Humidity"].mean())

print("Average Wind Speed:",
      df["Wind Speed"].mean())

print("Total Precipitation:",
      df["Precipitation"].sum())

# ---------------------------------------
# 6. DATA VISUALIZATION
# ---------------------------------------

# Temperature over time
plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Temperature"],
    marker="o"
)

plt.title("Temperature Changes Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Humidity over time
plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Humidity"],
    color="green",
    marker="o"
)

plt.title("Humidity Changes Over Time")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Wind speed visualization
plt.figure(figsize=(10, 5))

plt.bar(
    df["Date"],
    df["Wind Speed"]
)

plt.title("Wind Speed Variation")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Precipitation visualization
plt.figure(figsize=(10, 5))

plt.bar(
    df["Date"],
    df["Precipitation"],
    color="skyblue"
)

plt.title("Precipitation Levels")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------
# 7. DAILY DATA AGGREGATION
# ---------------------------------------

df["Day"] = df["Date"].dt.date

daily_data = df.groupby("Day").agg({
    "Temperature": ["mean", "max", "min"],
    "Humidity": "mean",
    "Wind Speed": "mean",
    "Precipitation": "sum"
})

print("\nDaily Weather Statistics:")
print(daily_data)

# Daily average temperature
plt.figure(figsize=(10, 5))

daily_temp = df.groupby("Day")["Temperature"].mean()

plt.bar(
    daily_temp.index.astype(str),
    daily_temp.values
)

plt.title("Daily Average Temperature")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------
# 8. GEOGRAPHICAL VISUALIZATION
# ---------------------------------------

latitude = data["city"]["coord"]["lat"]
longitude = data["city"]["coord"]["lon"]

weather_map = folium.Map(
    location=[latitude, longitude],
    zoom_start=10
)

folium.Marker(
    [latitude, longitude],
    popup=f"Weather Location: {CITY}"
).add_to(weather_map)

weather_map.save("weather_map.html")

print("\nGeographical map saved as weather_map.html")

# ---------------------------------------
# 9. CORRELATION ANALYSIS
# ---------------------------------------

correlation = df[numeric_cols].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Weather Attributes Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------------------------------------
# 10. SAVE PROCESSED DATA
# ---------------------------------------

df.to_csv("weather_data.csv", index=False)

print("\nProcessed data saved successfully!")

print("\nWeather Data Analysis Completed!")
