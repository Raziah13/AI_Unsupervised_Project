import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
algorithm = st.sidebar.selectbox("Choose Clustering Algorithm", ["K-Means", "DBSCAN"])
k_value = st.sidebar.slider("Select k for K-Means", 2, 10, 5)
eps_value = st.sidebar.slider("eps for DBSCAN", 0.1, 5.0, 0.8)
min_samples_value = st.sidebar.slider("min_samples for DBSCAN", 2, 20, 5)

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Print column names for debugging
    st.write("Column names in your CSV:", df.columns.tolist())

    # Rename columns to match what the code expects
    df_clean = df.copy()

    # Check if columns exist and rename them
    if 'Annual Income (k$)' in df_clean.columns:
        df_clean.rename(columns={'Annual Income (k$)': 'Annual_Income'}, inplace=True)
    if 'Spending Score (1-100)' in df_clean.columns:
        df_clean.rename(columns={'Spending Score (1-100)': 'Spending_Score'}, inplace=True)

    # Encode Gender
    if 'Gender' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Gender_Encoded'] = le.fit_transform(df_clean['Gender'])
    else:
        df_clean['Gender_Encoded'] = 0

    # Features for clustering
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Encoded']

    # Check if all features exist
    missing_features = [f for f in features if f not in df_clean.columns]
    if missing_features:
        st.error(f"Missing columns: {missing_features}")
        st.write("Available columns:", df_clean.columns.tolist())
        st.stop()

    X = df_clean[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    if algorithm == "K-Means":
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df_clean['Cluster'] = labels
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels)
            st.success(f"K-Means Silhouette Score: {sil_score:.4f}")
        else:
            st.warning("Only one cluster found")
    else:
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        labels = dbscan.fit_predict(X_scaled)
        df_clean['Cluster'] = labels.astype(str)
        mask = labels != -1
        if len(np.unique(labels[mask])) > 1:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
            st.success(f"DBSCAN Silhouette Score: {sil_score:.4f}")
        else:
            st.warning("Not enough clusters found")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]

    # Plot
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(data=df_clean, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, ax=ax)
    ax.set_title(f"{algorithm} Clusters")
    st.pyplot(fig)

    # Show cluster statistics
    st.subheader("Cluster Statistics")
    cluster_stats = df_clean.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual_Income': 'mean',
        'Spending_Score': 'mean'
    }).round(1)
    cluster_stats['Size'] = df_clean.groupby('Cluster').size()
    st.dataframe(cluster_stats)

    # Download
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "segmentation_results.csv")

else:
    st.info("Please upload a CSV file to begin")

    # Show expected format
    st.subheader("Expected CSV Format")
    st.write("Your CSV should have columns like:")
    sample = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Gender': ['Male', 'Female', 'Female'],
        'Age': [25, 35, 42],
        'Annual Income (k$)': [50, 75, 100],
        'Spending Score (1-100)': [60, 75, 85]
    })
    st.dataframe(sample)

    st.subheader("Sample Data")
    st.write("Download the Mall Customers dataset or use this format")
