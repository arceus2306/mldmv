
import os
import kagglehub
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Step 1: Download the dataset using KaggleHub
path = kagglehub.dataset_download("purusinghvi/email-spam-classification-dataset")
print("Path to dataset files:", path)

# Locate and load the CSV file
csv_file = None
for file in os.listdir(path):
    if file.endswith('.csv'):
        csv_file = os.path.join(path, file)
        break

df = pd.read_csv(csv_file)

print("\nDataset:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nClass Distribution:")
print(df['label'].value_counts())

# Step 2: Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)

    processed_words = []

    for word in tokens:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            processed_words.append(stemmed_word)

    return ' '.join(processed_words)

df['processed_text'] = df['text'].apply(preprocess_text)

print("\nPreprocessed Emails:")
print(df[['text', 'processed_text']].head())

# Step 3: Feature extraction using TF-IDF
X = df['processed_text']
y = df['label']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

print("\nTF-IDF Feature Shape:", X.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Step 4: KNN classification
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean',
    algorithm='brute'
)

knn.fit(X_train, y_train)

# Predict the test emails
y_pred = knn.predict(X_test)

print("\nPredicted Labels:")
print(y_pred[:10])

# Step 5: Performance evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam', zero_division=0)
recall = recall_score(y_test, y_pred, pos_label='spam', zero_division=0)

cm = confusion_matrix(y_test, y_pred, labels=['spam', 'ham'])

TP, FN, FP, TN = cm.ravel()

specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

print("\nPerformance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall / Sensitivity:", recall)
print("Specificity:", specificity)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Display confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Spam', 'Ham'],
    yticklabels=['Spam', 'Ham']
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - KNN Spam Classification")
plt.show()

# Test with a new email
new_email = ["Congratulations! You have won a free prize. Claim your reward now!"]

processed_email = [preprocess_text(new_email[0])]
email_vector = vectorizer.transform(processed_email)

prediction = knn.predict(email_vector)

print("\nNew Email:", new_email[0])
print("Predicted Category:", prediction[0])
