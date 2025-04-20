import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import metrics

df = pd.read_csv('linreg_data.csv',skiprows=0,names=['x','y'])

print(df.head())
print(df.columns)

xp = df['x']
yp = df['y']

xm = np.mean(xp)
ym = np.mean(yp)
n = len(xp)

b = (np.sum(xp * yp) - n * xm * ym) / (np.sum(xp ** 2) - n * xm ** 2)
a = ym - b * xm

print(a)
print(n)
print(b)

x = np.linspace(0,2,100)
y= a+b*x
plt.plot(x,y,color="green")
plt.scatter(xp,yp,color="green")
plt.scatter(xm,ym,color="yellow")
plt.show()
print("so the x is :",x)
print("so the y is :",y)

plt.scatter(xp, yp, color='blue')
plt.plot(xp, a + b * xp, color='black')
plt.scatter(xm, ym, color='red', s=100, marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.title('regression line')
plt.show()


#prediction with regration line

xval = 0.2
yval = a+b*xval

print("yval :",yval)

xval = np.array([0.5,0.75,0.99,1])
yval = a+b*xval

print("yval agin :",yval)


yhat = a*xval+b

print('Mean Absolute Error:', metrics.mean_absolute_error(yval, yhat))
print('Mean Squared Error:', metrics.mean_squared_error(yval, yhat))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yval, yhat)))

my_data = np.genfromtxt("linreg_data.csv",delimiter=",")
print(my_data)
xp = my_data[:,0]
yp = my_data[:,1]
print(xp)
xp = xp.reshape(-1,1)
yp = yp.reshape(-1,1)
print("xp =",xp)
regr = linear_model.LinearRegression()

regr.fit(xp,yp)
print("slope b =",regr.coef_)
print("y-intercept =",regr.intercept_)

xval = np.full((1,1),0.5)
yval = regr.predict(xval)
print("yval =",yval)
yhat = regr.predict(xp)
print("yhat =",yhat)

#evalution

