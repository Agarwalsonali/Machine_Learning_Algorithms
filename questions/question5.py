# q-Implement stochastic gradient descent algorithm.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD model
sgd = SGDClassifier(max_iter=1000, tol=1e-3)

# Train model
sgd.fit(X_train, y_train)

# Predictions
y_pred = sgd.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("SGD Model Accuracy:", accuracy)