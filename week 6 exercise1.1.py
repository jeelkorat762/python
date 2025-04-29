import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

df = pd.read_csv('Admission_Predict.csv')

X = df[["CGPA", "GRE Score"]]
y = df["Chance of Admit "]  # Make sure there's a space after "Admit" in the CSV


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


scaler_norm = MinMaxScaler()
X_train_norm = scaler_norm.fit_transform(X_train)
X_test_norm = scaler_norm.transform(X_test)

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

model = neighbors.KNeighborsRegressor(n_neighbors=5)


model.fit(X_train, y_train)
print("R2 =", model.score(X_test, y_test))


model.fit(X_train_norm, y_train)
print("R2 (norm) =", model.score(X_test_norm, y_test))


model.fit(X_train_std, y_train)
print("R2 (std) =", model.score(X_test_std, y_test))