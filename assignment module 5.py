import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Problem 1: Diabetes Prediction

print("\n--- Problem 1: Diabetes ---")

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

"""
a) We use bmi and s5 as base features, and add 'age' (index 0) as a new variable.
Age may influence diabetes progression, making it a useful addition.
"""

# Base model: bmi (2), s5 (5)
X_base = X[:, [2, 5]]
# Extended model: bmi (2), s5 (5), age (0)
X_extended = X[:, [2, 5, 0]]

# Split datasets
X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=0)
X_train_ext, X_test_ext = train_test_split(X_extended, test_size=0.2, random_state=0)[0:2]

# Fit models
model_base = LinearRegression().fit(X_train_base, y_train)
model_ext = LinearRegression().fit(X_train_ext, y_train)

# Predictions
y_pred_base = model_base.predict(X_test_base)
y_pred_ext = model_ext.predict(X_test_ext)

print("Base Model (bmi + s5):")
print("MSE:", mean_squared_error(y_test, y_pred_base))
print("R2:", r2_score(y_test, y_pred_base))

print("Extended Model (bmi + s5 + age):")
print("MSE:", mean_squared_error(y_test, y_pred_ext))
print("R2:", r2_score(y_test, y_pred_ext))

"""
b) Adding 'age' improves R2 score and reduces MSE slightly, showing better performance.
d) Adding more variables like blood pressure (bp), s1-s6 may improve accuracy further.
"""

# Problem 2: Profit Prediction

print("\n--- Problem 2: Profit Prediction ---")

# Load dataset (adjust path if needed)
df = pd.read_csv('50_Startups.csv')

# Convert categorical variable into dummy
df = pd.get_dummies(df, drop_first=True)

"""
1) Dataset includes R&D Spend, Administration, Marketing Spend, State, and Profit.
2) We'll look at correlation to decide relevant variables.
"""
print(df.corr())

X = df.drop('Profit', axis=1)
y = df['Profit']

# Plot explanatory variables
sns.pairplot(df, x_vars=X.columns, y_vars='Profit', kind='scatter')
plt.suptitle("Profit vs Input Features", y=1.02)
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# FIXED: using np.sqrt instead of squared=False
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Train R2:", r2_score(y_train, y_train_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Test R2:", r2_score(y_test, y_test_pred))

"""
3) R&D Spend and Marketing Spend have highest correlation with Profit.
4) Scatter plots show strong linear relationships.
"""

# Problem 3: Car MPG Prediction

print("\n--- Problem 3: Car MPG Prediction ---")

auto = pd.read_csv("Auto.csv")
auto = auto.dropna()

"""
1) Data includes 'mpg' and features like cylinders, displacement, horsepower, etc.
2) We'll exclude 'mpg', 'name', and 'origin' for prediction.
"""

X = auto.drop(columns=['mpg', 'name', 'origin'])
y = auto['mpg']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# 4) Train Ridge and Lasso with different alphas
alphas = np.logspace(-4, 4, 50)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha, max_iter=10000)

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_scores.append(ridge.score(X_test, y_test))
    lasso_scores.append(lasso.score(X_test, y_test))

# 6) Plot scores
plt.plot(alphas, ridge_scores, label='Ridge')
plt.plot(alphas, lasso_scores, label='Lasso')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('Ridge vs Lasso R2 Scores')
plt.legend()
plt.grid(True)
plt.show()

# 7) Best alpha
best_ridge_alpha = alphas[np.argmax(ridge_scores)]
best_lasso_alpha = alphas[np.argmax(lasso_scores)]

print("Best Ridge alpha:", best_ridge_alpha)
print("Best Ridge R2:", max(ridge_scores))
print("Best Lasso alpha:", best_lasso_alpha)
print("Best Lasso R2:", max(lasso_scores))

"""
5) Optimal alpha is where R2 is highest on testing set.
6) Ridge usually gives better generalization, while Lasso can reduce features.
7) Best alpha is printed above for both regressors.
"""
