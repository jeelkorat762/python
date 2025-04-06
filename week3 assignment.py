# 1. Exercise
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.linspace(-5, 5, 100)

y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3

plt.figure(figsize=(8, 6))
plt.plot(x, y1, 'r--', label='y = 2x + 1')
plt.plot(x, y2, 'g-', label='y = 2x + 2')
plt.plot(x, y3, 'b:', label='y = 2x + 3')

plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Graphs of y = 2x + c for different values of c')
plt.legend()
plt.grid(True)
plt.show()


# 2. Exercise
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.scatter(x, y, marker='+', color='green')
plt.title("Scatter plot of points (x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# 3. Exercise

data = pd.read_csv('weight-height.csv')

# Extract height and weight
length = data['Height'].values  # in inches
weight = data['Weight'].values  # in pounds

# Convert to metric units
length_cm = length * 2.54
weight_kg = weight * 0.453592

# Calculate means
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean Length (cm): {mean_length:.2f}")
print(f"Mean Weight (kg): {mean_weight:.2f}")

plt.figure(figsize=(7, 5))
plt.hist(length_cm, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Student Heights (cm)")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

#4 exercise
# Define matrix A
A = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
])

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

# Print the inverse
print("Inverse of A:")
print(A_inv)

# Check A * A_inv
identity_1 = np.dot(A, A_inv)
print("\nA * A_inv (should be identity):")
print(identity_1)

# Check A_inv * A
identity_2 = np.dot(A_inv, A)
print("\nA_inv * A (should be identity):")
print(identity_2)