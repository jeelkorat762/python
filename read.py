import numpy as np

a = np.array([[1,2,3],[4,5,6]])
print(a)
print(a[1,1])

x = np.array([[[1,2,3],[3,4,5]],[[5,6,7],[9,10,11]]])
print(x[1,1,[1]])
Z = np.zeros(5)
print(Z)
np.shape(Z)
Z2 = np.zeros((4,5))
print(Z2)
np.shape(Z2)
Y = np.ones((2,3))
print(Y)
F = np.full((7,8),11)

X = np.linspace(0,5,10)
print(X)
X2 = np.arange(0,5,0.2)
print(X2)

a = 1
b = 6
amount = 50
nopat = np.random.randint(a,b+1,amount)
print(nopat)
x = np.random.randn(100)
print(x)




