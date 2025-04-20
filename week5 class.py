
import pandas as pd


data_pd = pd.read_csv('weight-height.csv',names=['weight','height'],skiprows=1)

print(data_pd.corr())