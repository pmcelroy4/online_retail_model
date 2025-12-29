# model/train.py

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, MeanShift, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
from processing import clean_data

# import data

data_path = 'data/Online Retail.csv'
df = pd.read_csv(data_path)

data = clean_data(df)

# Scale Data for Customer Segmentation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Train different clustering algorithms - DBScan, KMeans, MeanShift, Spectral


