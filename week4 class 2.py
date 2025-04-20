import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

my_data = np.genfromtxt('linreg_data.csv', delimiter=',')
print(my_data)
xp = my_data[:,0]
yp = my_data[:,1]
print(xp)
xp = xp.reshape(-1,1)
yp = yp.reshape(-1,1)
print("xp =", xp)
regr = linear_model.LinearRegression()

#Training/fitting model with training  data
regr.fit(xp,yp)
print("slope b :",regr.coef_)
print("y-intercept a :",regr.intercept_)

#Calculating predictions

xval = np.full((1,1),0.5)
yval = regr.predict(xval)
print("yval :",yval)

yhat = regr.predict(xp)
print("yhat :",yhat)
print('Mean Absolute Error:', metrics.mean_absolute_error(yp, yhat))
print('Mean Squared Error:', metrics.mean_squared_error(yp, yhat))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yp, yhat)))


data_pd = pd.read_csv('linreg_data.csv',skiprows=0,names=['x','y'])

print(data_pd)
xpd = np.array(data_pd['x'])
ypd = np.array(data_pd['y'])
xpd = xpd.reshape(-1,1)
ypd = ypd.reshape(-1,1)
print(xpd)

pol_reg = PolynomialFeatures(degree=2)
X_poly = pol_reg.fit_transform(xpd)
print(X_poly)

poly_reg = LinearRegression()
poly_reg.fit(X_poly,ypd)
plt.scatter(xpd,ypd,color='blue')

xval = np.linspace(-1,50).reshape(-1,1)
yhat = poly_reg.predict(pol_reg.fit_transform(xval))
plt.plot(xval,yhat)
plt.show()

print("intercept =", poly_reg.intercept_)
print("coef =", poly_reg.coef_)
print("xpd =",xpd)
yhat = poly_reg.predict(pol_reg.fit_transform(xpd))
print("yhat =",yhat)
print('R2 value =', metrics.r2_score(ypd, yhat))