# ===============================
# Decision Tree Classifier on Iris Dataset
# ===============================



# 1. Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ------------------------------
# 2. Load the Iris dataset
# ------------------------------
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("First 5 rows of dataset:")
print(df.head())
print("\nShape of dataset:", df.shape)

# ------------------------------
# 3. Split into features and labels
# ------------------------------
X = df.iloc[:, :-1]   # all columns except target
y = df["target"]      # target column

# ------------------------------
# 4. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# ------------------------------
# 5. Import & train Decision Tree Classifier
# ------------------------------
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# ------------------------------
# 6. Evaluate the model
# ------------------------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy on Test Set:", accuracy)

# ------------------------------
# 7. Visualize the Decision Tree
# ------------------------------
plt.figure(figsize=(16, 10))
plot_tree(
    clf,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True
)
plt.show()
