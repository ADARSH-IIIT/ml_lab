import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# -----------------------------------------
# Load MNIST dataset
# -----------------------------------------
digits = datasets.load_digits()

X = digits.data     # 64 pixel values
y = digits.target   # 0–9 labels

# -----------------------------------------
# Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------
# Train SVM classifier
# -----------------------------------------
svm_clf = SVC(kernel='rbf', C=10, gamma=0.001)
svm_clf.fit(X_train, y_train)

# -----------------------------------------
# Predictions
# -----------------------------------------
y_pred = svm_clf.predict(X_test)

# -----------------------------------------
# Classification accuracy
# -----------------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------------------
# Regression-style metrics
# -----------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\nMAE :", mae)
print("MSE :", mse)
print("R²  :", r2)
