import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

add = a + b
print(add)

substraction = a - b
print(substraction)

divide = a/b
print(divide)

mul = a*b
print(mul)

matmul = a@b
print(matmul)

Y = a >= 1
print(Y)

Y = a <= 1
print(Y)

#sin
print(np.sin(a))
#cos
print(np.cos(a))
#tan
print(np.tan(a))
#sqrt
print(np.sqrt(a))

added = a + 2
print(added)

deduct = a - 1
print(deduct)

n = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
n1 = np.reshape(n,(3,4))
rows,cols = n1.shape
print(rows,cols)

