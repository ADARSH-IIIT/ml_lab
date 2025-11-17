import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------------
# Load the Iris dataset
# -----------------------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for checking size
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("\nDataFrame Size:", df.shape)
print(df.head())

# -----------------------------------------
# Keep only Setosa (0) and Versicolor (1)
# -----------------------------------------
binary_df = df[df['target'] != 2]     # Remove Virginica
X_binary = binary_df.iloc[:, :-1]
y_binary = binary_df['target']

# -----------------------------------------
# Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# -----------------------------------------
# Train SVM with linear kernel
# -----------------------------------------
svm_clf = SVC(kernel='linear', C=1)
# c large means no error allowed 

svm_clf.fit(X_train, y_train)

# -----------------------------------------
# Predictions and evaluation
# -----------------------------------------
y_pred = svm_clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ------------------------------------------------
# Plot accuracy for different C values
# ------------------------------------------------
C_values = [0.01, 1, 100]
accuracies = []

for C in C_values:
    model = SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    y_pred_c = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_c))

plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel("C Value (log scale)")
plt.ylabel("Accuracy")
plt.title("Model Performance for Different C Values")
plt.grid(True)
plt.show()
