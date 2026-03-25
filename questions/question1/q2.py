# q-Split the dataset into training and testing sets(80% train, 20% test).

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()

X = iris.data      # Features
y = iris.target    # Target

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shapes
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)