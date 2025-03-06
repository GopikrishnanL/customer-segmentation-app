import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Load trained K-Means model and scaler
kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# GitHub raw file URL
url = "https://raw.githubusercontent.com/your-username/your-repo/main/cust_data.csv"

# Read CSV directly from GitHub
df = pd.read_csv(url)

# **Handle missing values by generating synthetic data**
if "Age" not in df.columns:
    df['Age'] = np.random.randint(18, 70, df.shape[0])  # Age between 18-70
if "Annual Income (k$)" not in df.columns:
    df['Annual Income (k$)'] = np.random.randint(20, 150, df.shape[0])  # Income between 20k-150k
if "Spending Score (1-100)" not in df.columns:
    df['Spending Score (1-100)'] = np.random.randint(1, 100, df.shape[0])  # Score between 1-100

# Streamlit App UI
st.title("Customer Segmentation App")
st.write("Enter customer details to predict their segment.")

# Sidebar Inputs
age = st.slider("Age", 18, 70, 30)
income = st.slider("Annual Income (k$)", 20, 150, 50)
spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)

# Predict Customer Cluster
if st.button("Predict Segment"):
    input_data = np.array([[age, income, spending_score]])
    input_data_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_data_scaled)
    st.success(f"Predicted Customer Cluster: {cluster[0]}")

    # Cluster Analysis Visualization
    st.subheader("Customer Segmentation Insights")

    # Predict clusters for the entire dataset
    df["Cluster"] = kmeans.predict(scaler.transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]))

    # Scatter plot for clusters
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette="viridis", s=50)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments")
    st.pyplot(plt)
