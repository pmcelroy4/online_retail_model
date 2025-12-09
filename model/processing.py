# model/processing.py

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
data_path = "data/Online Retail.csv"
df = pd.read_csv(data_path)

# get general info on data
print(df.info())
print(df.describe())
print(df.head())

# numeric features
numeric_features = [
    "Quantity",
    "UnitPrice",
    "CustomerID"
]

# distribution of numeric features
plt.figure(figsize=(12,8))
for i, feature in enumerate(numeric_features):
    plt.subplot(3, 1, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distrubution of {feature}")
plt.tight_layout()
plt.show()

# correlation plot of numeric features
plt.figure(figsize=(8,6))
corr = df[numeric_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

def clean_data(df):
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df.dropna(subset='CustomerID')
    df['Total Price'] = df['Quantity']*df['UnitPrice']
    data = pd.DataFrame()
    data['Transactions'] = df.groupby('CustomerID')['InvoiceNo'].sum()
    data['Total Items'] = df.groupby('CustomerID')['Quantity'].sum()
    data['Unique Items'] = df.groupby('CustomerID')['StockCode'].count()
    data['Avg Items'] = df.groupby('CustomerID')['Quantity'].mean()
    data['Avg Unit Price'] = df.groupby('CustomerID')['UnitPrice'].mean()