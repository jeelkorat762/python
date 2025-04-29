import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report



# Load data
df = pd.read_csv('iris.csv')

# Binary classification: virginica vs others
df['label'] = (df['species'] == 'Iris-virginica').astype(int)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# Print metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}, "
      f"Precision: {precision_score(y_test, y_pred):.2f}, "
      f"Recall: {recall_score(y_test, y_pred):.2f}")

# KNN error analysis
error = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    error.append(np.mean(y_pred_k != y_test))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 20), error, marker='o', markersize=5)
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.title('KNN Error Rate for k = 1 to 19')
plt.grid(True)
plt.show()

# Confusion matrix
print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred).ravel())

# ADDITIONAL: Scatter plot of Sepal dimensions colored by species
species = df['species'].unique()
colors = ['red', 'green', 'blue']

plt.figure(figsize=(8, 5))
for i, sp in enumerate(species):
    subset = df[df['species'] == sp]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], label=sp, color=colors[i])


# SVC classification
svcclassifier = SVC(kernel='linear')
svcclassifier.fit(X_train_scaled, y_train)  # Use scaled data for SVC too
y_pred_svc = svcclassifier.predict(X_test_scaled)

# Metrics for SVC
print(f"\nSVC Accuracy: {accuracy_score(y_test, y_pred_svc):.2f}")

# SVC - Confusion Matrix and Classification Report
print("\nSVC Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svc))
print("\nSVC Classification Report:")
print(classification_report(y_test, y_pred_svc))



plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Width by Species")
plt.legend()
plt.grid(True)
plt.show()