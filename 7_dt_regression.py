# ===============================
# Decision Tree Regressor on Diabetes Dataset
# ===============================

# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------------------
# 2. Load Diabetes Dataset
# ------------------------------
diabetes = load_diabetes()

# Convert to DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:", df.shape)

# ------------------------------
# 3. Split into features (X) and labels (y)
# ------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# ------------------------------
# 4. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 5. Train Decision Tree Regressor (Without Pruning)
# ------------------------------
dtr_no_pruning = DecisionTreeRegressor(random_state=42)
dtr_no_pruning.fit(X_train, y_train)

y_pred_no = dtr_no_pruning.predict(X_test)

# ------------------------------
# 6. Train Decision Tree (Pre-Pruning)
#    Pre-pruning limits tree growth (max_depth, min_samples_split etc.)
# ------------------------------
dtr_pre = DecisionTreeRegressor(
    max_depth=5,            # limit depth
    min_samples_split=10,   # splits allowed only if ≥ 10 samples
    random_state=42
)
dtr_pre.fit(X_train, y_train)

y_pred_pre = dtr_pre.predict(X_test)

# ------------------------------
# 7. Train Decision Tree (Post-Pruning)
#    Post-pruning uses cost complexity pruning (ccp_alpha)
# ------------------------------
path = dtr_no_pruning.cost_complexity_pruning_path(X_train, y_train)
ccp_values = path.ccp_alphas


# it returna list of alpha from unpruned normla tre  

# pick min one 
# Choose a middle alpha for pruning
alpha_best = ccp_values[len(ccp_values) // 2]

dtr_post = DecisionTreeRegressor(ccp_alpha=alpha_best, random_state=42)
dtr_post.fit(X_train, y_train)

y_pred_post = dtr_post.predict(X_test)

# ------------------------------
# 8. Compare Metrics
# ------------------------------
results = pd.DataFrame({
    "Model": ["No Pruning", "Pre-Pruning", "Post-Pruning"],
    "MSE": [
        mean_squared_error(y_test, y_pred_no),
        mean_squared_error(y_test, y_pred_pre),
        mean_squared_error(y_test, y_pred_post)
    ],
    "R2 Score": [
        r2_score(y_test, y_pred_no),
        r2_score(y_test, y_pred_pre),
        r2_score(y_test, y_pred_post)
    ]
})

print("\nModel Performance Comparison:")
print(results)

# ------------------------------
# 9. Plot trees (optional)
# ------------------------------

plt.figure(figsize=(12, 6))
plt.title("Decision Tree (Pre-Pruning)")
plot_tree(dtr_pre, filled=True, feature_names=X.columns)
plt.show()

plt.figure(figsize=(12, 6))
plt.title("Decision Tree (Post-Pruning)")
plot_tree(dtr_post, filled=True, feature_names=X.columns)
plt.show()
