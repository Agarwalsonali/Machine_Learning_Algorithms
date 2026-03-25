# q-Load any suitable dataset using scikit-learn. Explore and print basic statistics of the data.

# Import required libraries
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()

# Convert to DataFrame for better understanding
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column
df['target'] = iris.target

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Shape of dataset
print("\nShape of dataset:", df.shape)

# Column names
print("\nColumn names:")
print(df.columns)

# Data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# Count of each class
print("\nClass distribution:")
print(df['target'].value_counts())