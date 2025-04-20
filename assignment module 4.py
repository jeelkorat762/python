import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Exercise 1: Dice Histogram
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, size=n)
    dice2 = np.random.randint(1, 7, size=n)
    dice_sum = dice1 + dice2

    h, h2 = np.histogram(dice_sum, bins=range(2, 14))
    plt.bar(h2[:-1], h / n, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of Dice Sums (n = {n})")
    plt.xlabel("Sum of Two Dice")
    plt.ylabel("Frequency")
    plt.xticks(range(2, 13))
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()

# Exercise 2: Linear Regression
df = pd.read_csv('weight-height.csv')
X = df['Height'].values.reshape(-1, 1)
y = df['Weight'].values

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# Sort X and y_pred for smoother plot line
sorted_indices = X.flatten().argsort()
X_sorted = X[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.scatter(X, y, alpha=0.4, label='Actual data')
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression line')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.title('Linear Regression: Height vs Weight')
plt.legend()
plt.grid(True)
plt.show()

print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
print(f"R²: {r2_score(y, y_pred):.4f}")
