
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
columns = [
    'buying', 'maint', 'doors',
    'persons', 'lug_boot', 'safety', 'class'
]

df = pd.read_csv('car.data', names=columns)

# 2. Explore dataset
print(df.head())
print(df.info())
print(df.describe(include='all'))
print(df.isnull().sum())

# 3. Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# 4. Encode categorical data
le = LabelEncoder()

for col in X.columns:
    X[col] = le.fit_transform(X[col])

y = le.fit_transform(y)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 7. Train model
model.fit(X_train, y_train)

# 8. Predict test data
y_pred = model.predict(X_test)

# 9. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 11. Feature Importance
importance = model.feature_importances_

plt.bar(X.columns, importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()
