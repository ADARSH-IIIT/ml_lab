import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Load dataset
# -------------------------------
data = fetch_california_housing()
X = data.data
y = data.target

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train SVM Regressor
# -------------------------------
svm_reg = SVR(kernel='rbf', C=1.0, gamma=2)
svm_reg.fit(X_train, y_train)


# ⭐ Larger gamma → model becomes very wiggly
# ⭐ Smaller gamma → smoother model
# param of rbf
# K(x,x′)=e−γ∥x−x′∥2

# -------------------------------
# Predict
# -------------------------------
y_pred = svm_reg.predict(X_test)

# -------------------------------
# Performance metrics
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)
