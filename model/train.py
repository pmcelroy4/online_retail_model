# model/train.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# import data
data_path = "data/Online Retail.xlsx"
df = pd.read_excel(data_path)

# get general info on data
print(df.info())
print(df.describe())
print(df.head())