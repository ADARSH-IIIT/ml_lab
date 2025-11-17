import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# Load Iris dataset
# -----------------------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# -----------------------------------------
# Keep only Setosa (0) and Versicolor (1)
# -----------------------------------------
binary_df = df[df['target'] != 2]

X = binary_df.iloc[:, :-1].values
y = binary_df['target'].values

# -----------------------------------------
# Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------
# Compare different kernels (NO PLOTS)
# -----------------------------------------
kernels = ["linear", "poly", "rbf"]

for k in kernels:
    print(f"\n==============================")
    print(f"      Kernel: {k}")
    print("==============================")

    model = SVC(kernel=k, C=1, gamma="scale")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
