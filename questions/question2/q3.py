# q-Use cross_val_score for k-fold cross-validation and discuss the results.

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create model
knn = KNeighborsClassifier(n_neighbors=3)

# Apply 5-fold cross-validation
scores = cross_val_score(knn, X, y, cv=5)

# Print results
print("Accuracy for each fold:", scores)
print("Mean accuracy:", scores.mean())