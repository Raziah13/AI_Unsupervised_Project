# Update your streamlit_app.py with comparison and profiling
import os
if os.path.exists('streamlit_app.py'):
    os.remove('streamlit_app.py')

with open('streamlit_app.py', 'w') as f:
    f.write('''import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Preprocessing
    df_clean = df.copy()
    
    # Handle column names
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
    
    # Features
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Encoded']
    X = df_clean[features].copy()
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ============================================
    # K-MEANS CLUSTERING
    # ============================================
    st.header("1. K-Means Clustering")
    
    k_value = st.slider("Select k for K-Means", 2, 10, 5, key="kmeans")
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df_clean['KMeans_Cluster'] = kmeans_labels
    
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
    kmeans_davies = davies_bouldin_score(X_scaled, kmeans_labels)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{kmeans_silhouette:.4f}", help="Higher is better (-1 to 1)")
    col2.metric("Calinski-Harabasz", f"{kmeans_calinski:.0f}", help="Higher is better")
    col3.metric("Davies-Bouldin", f"{kmeans_davies:.4f}", help="Lower is better")
    
    # ============================================
    # DBSCAN CLUSTERING
    # ============================================
    st.header("2. DBSCAN Clustering")
    
    col1, col2 = st.columns(2)
    with col1:
        eps_value = st.slider("eps", 0.1, 2.0, 0.8, key="eps")
    with col2:
        min_samples_value = st.slider("min_samples", 2, 20, 5, key="min_samples")
    
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    df_clean['DBSCAN_Cluster'] = dbscan_labels
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters Found", n_clusters)
    col2.metric("Noise Points", n_noise)
    col3.metric("Noise %", f"{(n_noise/len(dbscan_labels))*100:.1f}%")
    
    if n_clusters >= 2:
        mask = dbscan_labels != -1
        dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        st.metric("DBSCAN Silhouette Score", f"{dbscan_silhouette:.4f}")
    
    # ============================================
    # METHOD COMPARISON
    # ============================================
    st.header("3. Method Comparison")
    
    comparison_data = {
        'Metric': ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin', 'Clusters', 'Noise Points'],
        'K-Means': [f"{kmeans_silhouette:.4f}", f"{kmeans_calinski:.0f}", f"{kmeans_davies:.4f}", k_value, 0],
        'DBSCAN': [f"{dbscan_silhouette:.4f}" if n_clusters >= 2 else "N/A", "N/A", "N/A", n_clusters, n_noise]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visual comparison
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # K-Means plot
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    axes[0].set_title(f'K-Means (k={k_value})', fontsize=12)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # DBSCAN plot
    unique_labels = set(dbscan_labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
            label = 'Noise'
        else:
            label = f'Cluster {k}'
        mask = dbscan_labels == k
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], s=50, label=label)
    axes[1].set_title(f'DBSCAN (eps={eps_value})', fontsize=12)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================
    # CUSTOMER SEGMENT PROFILING (using K-Means)
    # ============================================
    st.header("4. Customer Segment Profiles")
    
    st.subheader("K-Means Segment Profiles")
    
    # Create segment profiles
    segment_profiles = []
    for cluster in range(k_value):
        cluster_data = df_clean[df_clean['KMeans_Cluster'] == cluster]
        
        size = len(cluster_data)
        percentage = (size / len(df_clean)) * 100
        
        # Determine segment type based on income and spending
        income_mean = cluster_data['Annual_Income'].mean()
        spending_mean = cluster_data['Spending_Score'].mean()
        age_mean = cluster_data['Age'].mean()
        
        if income_mean > 70 and spending_mean > 60:
            segment = "Premium Spenders"
        elif income_mean > 70 and spending_mean < 40:
            segment = "Affluent Frugal"
        elif income_mean < 40 and spending_mean > 60:
            segment = "Budget Enthusiasts"
        elif income_mean < 40 and spending_mean < 40:
            segment = "Cautious Customers"
        else:
            segment = "Moderate Customers"
        
        segment_profiles.append({
            'Cluster': cluster,
            'Segment': segment,
            'Size': size,
            'Percentage': f"{percentage:.1f}%",
            'Age': round(age_mean, 1),
            'Income': round(income_mean, 0),
            'Spending': round(spending_mean, 1),
            'Gender': f"{cluster_data['Gender'].value_counts().get('Female', 0)/size*100:.0f}% Female"
        })
    
    profile_df = pd.DataFrame(segment_profiles)
    st.dataframe(profile_df, use_container_width=True)
    
    # Visualize segment characteristics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Age by segment
    axes[0].bar(profile_df['Cluster'], profile_df['Age'], color='skyblue', edgecolor='black')
    axes[0].set_title('Age by Segment', fontsize=12)
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Average Age')
    
    # Income by segment
    axes[1].bar(profile_df['Cluster'], profile_df['Income'], color='lightgreen', edgecolor='black')
    axes[1].set_title('Income by Segment', fontsize=12)
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Annual Income (k$)')
    
    # Spending by segment
    axes[2].bar(profile_df['Cluster'], profile_df['Spending'], color='salmon', edgecolor='black')
    axes[2].set_title('Spending by Segment', fontsize=12)
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Spending Score')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================
    # DOWNLOAD RESULTS
    # ============================================
    st.header("5. Download Results")
    
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("Download Complete Results CSV", csv, "segmentation_results.csv")
    
else:
    st.info("Please upload a CSV file to begin")
    
    st.subheader("Expected CSV Format")
    st.write("Your CSV should have columns like:")
    example = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Gender': ['Male', 'Female', 'Female'],
        'Age': [25, 35, 42],
        'Annual Income (k$)': [50, 75, 100],
        'Spending Score (1-100)': [60, 75, 85]
    })
    st.dataframe(example)
''')

print("✅ Updated streamlit_app.py with full comparison and profiling!")
