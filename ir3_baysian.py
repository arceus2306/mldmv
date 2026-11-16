
import os
import glob
import kagglehub
import numpy as np
import pandas as pd

from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

# Download Heart Disease Dataset
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")

print("Path to dataset files:", path)

# Find and read CSV file
csv_files = glob.glob(os.path.join(path, "*.csv"), recursive=False)

if not csv_files:
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

heartDisease = pd.read_csv(csv_files[0])

# Rename target column to match the lab manual
heartDisease.rename(columns={'target': 'heartdisease'}, inplace=True)

# Replace missing values
heartDisease.replace('?', np.nan, inplace=True)

# Select attributes used in the Bayesian Network
columns = ['age', 'sex', 'cp', 'exang', 'heartdisease', 'restecg', 'chol']
heartDisease = heartDisease[columns].copy()

# Convert columns to numeric values
for col in columns:
    heartDisease[col] = pd.to_numeric(heartDisease[col], errors='coerce')

# Remove rows containing missing values
heartDisease.dropna(inplace=True)

# Convert continuous attributes into discrete categories
heartDisease['age'] = pd.cut(
    heartDisease['age'],
    bins=[0, 40, 55, 100],
    labels=['Young', 'Middle', 'Senior']
)

heartDisease['chol'] = pd.cut(
    heartDisease['chol'],
    bins=[0, 200, 240, 300, 600],
    labels=['Low', 'Normal', 'High', 'Very High']
)

print('\nSample instances from the dataset:')
print(heartDisease.head())

print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# Construct Bayesian Network
model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

# Learn CPDs using Maximum Likelihood Estimator
print('\nLearning CPDs using Maximum Likelihood Estimator')

model.fit(
    heartDisease,
    estimator=DiscreteMLE()
)

# Check whether the model is valid
print('\nModel valid:', model.check_model())

# Perform inference using Bayesian Network
print('\nInferencing with Bayesian Network:')

HeartDisease_infer = VariableElimination(model)

# Query 1: Probability of HeartDisease given restecg = 1
print('\n1. Probability of HeartDisease given evidence = restecg')

q1 = HeartDisease_infer.query(
    variables=['heartdisease'],
    evidence={'restecg': 1}
)

print(q1)

# Query 2: Probability of HeartDisease given cp = 2
print('\n2. Probability of HeartDisease given evidence = cp')

q2 = HeartDisease_infer.query(
    variables=['heartdisease'],
    evidence={'cp': 2}
)

print(q2)
