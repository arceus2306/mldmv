
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
digits = datasets.load_digits()

X = digits.data
y = digits.target

# 2. Visualize images
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(digits.target[i])
    plt.axis('off')

plt.show()

# 3. Convert into DataFrame
numbers = pd.DataFrame(X)
numbers['label'] = y

# 4. Data exploration
print(numbers.info())
print(numbers.describe())

print("Null values:")
print(numbers.isnull().sum())

print("Unique labels:")
print(np.unique(numbers['label']))

print("Label counts:")
print(numbers['label'].value_counts())

# 5. Separate features and labels
X = numbers.drop('label', axis=1)
y = numbers['label']

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Build SVM model
model = SVC(kernel='linear')

# 9. Train model
model.fit(X_train_scaled, y_train)

# 10. Predict test data
y_pred = model.predict(X_test_scaled)

# 11. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
