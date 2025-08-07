# Customer Segmentation Model

This project uses an unsupervised machine learning model to segment customers based on various features such as product description, .

## 🚀 Project Overview

Segment customers into distinct groups based on purchasing behavior and demographics so that businesses can tailor marketing strategies, improve retention, and increase sales:

- Data preprocessing and feature engineering
- Model training and evaluation
- Prediction on unseen data
- Visualizations for interpretation

## 📁 Project Structure

Online Retail/
├── data/ # (Optional) Raw or processed datasets
├── model/
│ ├── train.py # Model training script
│ └── predict.py # Prediction script
├── tests/
│ └── test_model.py # Unit tests
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Files and folders to exclude from Git


## 🧠 Model Details

- **Model Type:** Clustering
- **Evaluation Metric:** Silhouette Score, Davies-Bouldin Index
- **Frameworks Used:** scikit-learn, pandas, numpy, matplotlib/seaborn


## 📦 Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/pmcelroy4/online_retail_model.git
cd Online Retail
