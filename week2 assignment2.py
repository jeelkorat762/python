import matplotlib.pyplot as plt
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.hstack((a, b))
result1 = np.vstack((a, b))
print(result)
print(result1)

result2 = np.vstack((a, b))

for i in range(result2.shape[0]):
    for j in range(result2.shape[1]):
        print(f"Element ({i}, {j}): {result2[i, j]}")



x = np.array([2020,2021,2022,2023,2024,2025])
y = np.array([100,200,400,600,800,1000])

plt.bar(x,y, color='lightblue')
plt.title('sales in millions')
plt.xlabel("years")
plt.ylabel("sales")
plt.show()


a = np.array([1, 2, 3,4,5,6,7,8,9,10,11,12])
a = a.reshape(4,3)
n,m = np.shape(a)
print(a)
for i in range(n):
    for j in range(m):
        print(f"Element ({i}, {j}): {a[i, j]}")

a2 = np.delete(a,[0],1)
print(a2)

a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
for i in np.nditer(a):
    print(i)

