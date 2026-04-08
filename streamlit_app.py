import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Streamlit page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")

# Sidebar for user inputs
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
algorithm = st.sidebar.selectbox("Choose Clustering Algorithm", ["K-Means", "DBSCAN"])
k_value = st.sidebar.slider("Select k for K-Means", min_value=2, max_value=10, value=5)
eps_value = st.sidebar.slider("eps for DBSCAN", min_value=0.1, max_value=5.0, value=0.8)
min_samples_value = st.sidebar.slider("min_samples for DBSCAN", min_value=2, max_value=20, value=5)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
    df_clean = df.copy()
    df_clean.columns = [c.strip().replace(" ", "_") for c in df_clean.columns]

    if 'Gender' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Gender_Encoded'] = le.fit_transform(df_clean['Gender'])
    else:
        df_clean['Gender_Encoded'] = 0

    features = ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Encoded']
    X = df_clean[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Feature Statistics")
    st.dataframe(pd.DataFrame(X_scaled, columns=features).describe())

    # Clustering
    if algorithm == "K-Means":
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df_clean['Cluster'] = labels
        silhouette = silhouette_score(X_scaled, labels)
        st.success(f"K-Means Silhouette Score: {silhouette:.4f}")
    else:
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        labels = dbscan.fit_predict(X_scaled)
        df_clean['Cluster'] = labels
        mask = labels != -1
        if len(set(labels[mask])) >= 2:
            silhouette = silhouette_score(X_scaled[mask], labels[mask])
            st.success(f"DBSCAN Silhouette Score (excluding noise): {silhouette:.4f}")
        else:
            st.warning("DBSCAN produced less than 2 clusters for silhouette score")

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]

    st.subheader("Cluster Visualization (PCA 2D)")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(data=df_clean, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, ax=ax)
    ax.set_title(f"{algorithm} Clusters Visualization", fontsize=14)
    ax.set_xlabel("First Principal Component", fontsize=12)
    ax.set_ylabel("Second Principal Component", fontsize=12)
    st.pyplot(fig)

    # Segment Profiles
    st.subheader("Segment Profiles")
    cluster_profiles = []
    for cluster in sorted(df_clean['Cluster'].unique()):
        cluster_data = df_clean[df_clean['Cluster'] == cluster]
        profile = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Percentage': f"{len(cluster_data)/len(df_clean)*100:.1f}%",
            'Age Mean': round(cluster_data['Age'].mean(), 1),
            'Income Mean': round(cluster_data['Annual_Income'].mean(), 0),
            'Spending Mean': round(cluster_data['Spending_Score'].mean(), 1)
        }
        cluster_profiles.append(profile)

    profile_df = pd.DataFrame(cluster_profiles)
    st.dataframe(profile_df, use_container_width=True)

    # Download results
    st.subheader("Download Segmentation Results")
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV with Clusters",
        data=csv,
        file_name='customer_segmentation_results.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to begin segmentation")

    # Show sample data format
    st.subheader("Expected CSV Format")
    sample_data = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Gender': ['Male', 'Female', 'Female'],
        'Age': [25, 35, 42],
        'Annual_Income': [50000, 75000, 100000],
        'Spending_Score': [60, 75, 85]
    })
    st.dataframe(sample_data)
