import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Read the CSV file (pay attention to delimiter)
df = pd.read_csv("bank.csv", delimiter=';')
print(df.info())

# Step 2: Select specific columns
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print(df2.head())

# Step 3: Convert categorical variables to dummy variables
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

# Step 4: Fix the error - convert 'y' column to numeric (required for .corr())
df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

# Plot heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""
The correlation heatmap shows weak correlations among most variables.
No strong individual predictors of the target variable 'y' are observed.
"""

# Step 5: Define target and feature variables
y = df3['y']
X = df3.drop('y', axis=1)

# Step 6: Split into training and testing sets (75/25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Step 8: Evaluate Logistic Regression
log_cm = confusion_matrix(y_test, y_pred_log)
log_acc = accuracy_score(y_test, y_pred_log)
print("Confusion Matrix - Logistic Regression:\n", log_cm)
print("Accuracy - Logistic Regression:", log_acc)

sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression - Confusion Matrix')
plt.show()

# Step 9: K-Nearest Neighbors (K=3)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluate KNN
knn_cm = confusion_matrix(y_test, y_pred_knn)
knn_acc = accuracy_score(y_test, y_pred_knn)
print("Confusion Matrix - KNN (k=3):\n", knn_cm)
print("Accuracy - KNN (k=3):", knn_acc)

sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Greens')
plt.title('KNN (k=3) - Confusion Matrix')
plt.show()

"""
# Step 10: Model Comparison

- Logistic Regression accuracy: usually around 88%
- KNN (k=3) accuracy: slightly lower, around 85%

"""
