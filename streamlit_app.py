# FINAL WORKING VERSION - Copy and run this entire cell
import os

# Delete old file
if os.path.exists('streamlit_app.py'):
    os.remove('streamlit_app.py')

# Create clean app
code = '''import streamlit as st
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

st.sidebar.subheader("Navigation")
page = st.sidebar.selectbox("Choose Analysis Type", 
    ["K-Means Clustering", "DBSCAN Clustering", "Method Comparison", "Customer Profiling"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Preprocessing
    df_clean = df.copy()
    if 'Annual Income (k$)' in df_clean.columns:
        df_clean.rename(columns={'Annual Income (k$)': 'Annual_Income'}, inplace=True)
    if 'Spending Score (1-100)' in df_clean.columns:
        df_clean.rename(columns={'Spending Score (1-100)': 'Spending_Score'}, inplace=True)
    
    if 'Gender' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Gender_Encoded'] = le.fit_transform(df_clean['Gender'])
    else:
        df_clean['Gender_Encoded'] = 0
    
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Encoded']
    X = df_clean[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Session state
    if 'kmeans_ran' not in st.session_state:
        st.session_state.kmeans_ran = False
    if 'dbscan_ran' not in st.session_state:
        st.session_state.dbscan_ran = False
    if 'kmeans_labels' not in st.session_state:
        st.session_state.kmeans_labels = None
    if 'kmeans_k' not in st.session_state:
        st.session_state.kmeans_k = None
    if 'kmeans_sil' not in st.session_state:
        st.session_state.kmeans_sil = None
    if 'kmeans_cal' not in st.session_state:
        st.session_state.kmeans_cal = None
    if 'kmeans_dav' not in st.session_state:
        st.session_state.kmeans_dav = None
    if 'dbscan_labels' not in st.session_state:
        st.session_state.dbscan_labels = None
    if 'dbscan_clusters' not in st.session_state:
        st.session_state.dbscan_clusters = None
    if 'dbscan_noise' not in st.session_state:
        st.session_state.dbscan_noise = None
    if 'dbscan_sil' not in st.session_state:
        st.session_state.dbscan_sil = None
    if 'kmeans_df' not in st.session_state:
        st.session_state.kmeans_df = None
    
    # ========== K-MEANS CLUSTERING ==========
    if page == "K-Means Clustering":
        st.header("K-Means Clustering")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            k_value = st.slider("Number of Clusters (k)", 2, 10, 5)
            if st.button("Run K-Means", type="primary"):
                with st.spinner("Running K-Means..."):
                    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)
                    df_clean['Cluster'] = labels
                    sil_score = silhouette_score(X_scaled, labels)
                    cal_score = calinski_harabasz_score(X_scaled, labels)
                    dav_score = davies_bouldin_score(X_scaled, labels)
                    st.session_state.kmeans_ran = True
                    st.session_state.kmeans_labels = labels
                    st.session_state.kmeans_k = k_value
                    st.session_state.kmeans_sil = sil_score
                    st.session_state.kmeans_cal = cal_score
                    st.session_state.kmeans_dav = dav_score
                    st.session_state.kmeans_df = df_clean.copy()
                    st.success(f"Done! Silhouette Score: {sil_score:.4f}")
        
        with col2:
            if st.session_state.kmeans_ran:
                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette Score", f"{st.session_state.kmeans_sil:.4f}")
                m2.metric("Calinski-Harabasz", f"{st.session_state.kmeans_cal:.0f}")
                m3.metric("Davies-Bouldin", f"{st.session_state.kmeans_dav:.4f}")
        
        if st.session_state.kmeans_ran:
            st.subheader("Clustering Results")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=st.session_state.kmeans_labels, cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter)
            ax.set_xlabel("First Principal Component")
            ax.set_ylabel("Second Principal Component")
            ax.set_title("K-Means Clustering Results")
            st.pyplot(fig)
            
            st.subheader("Cluster Sizes")
            cluster_sizes = pd.Series(st.session_state.kmeans_labels).value_counts().sort_index()
            st.bar_chart(cluster_sizes)
            
            st.subheader("What Each Cluster Means - Segment Summary")
            summary_data = []
            for cluster in range(st.session_state.kmeans_k):
                cluster_data = st.session_state.kmeans_df[st.session_state.kmeans_df['Cluster'] == cluster]
                size = len(cluster_data)
                pct = (size / len(st.session_state.kmeans_df)) * 100
                avg_age = cluster_data['Age'].mean()
                avg_income = cluster_data['Annual_Income'].mean()
                avg_spending = cluster_data['Spending_Score'].mean()
                
                if avg_income > 70 and avg_spending > 60:
                    name = "VIP Premium Customers"
                    meaning = "High income, high spending - Your best customers"
                elif avg_income > 70 and avg_spending < 40:
                    name = "Smart Value Shoppers"
                    meaning = "High income but careful spenders - Focus on value deals"
                elif avg_income < 40 and avg_spending > 60:
                    name = "Aspiring Trendsetters"
                    meaning = "Lower income but love spending - Use social media"
                elif avg_income < 40 and avg_spending < 40:
                    name = "Practical Frugal"
                    meaning = "Budget conscious - Focus on essential items"
                elif avg_age < 30 and avg_spending > 60:
                    name = "Young Trend Hunters"
                    meaning = "Young active spenders - Use influencer marketing"
                elif avg_age > 50 and avg_income > 60:
                    name = "Established Affluents"
                    meaning = "Mature stable customers - Focus on quality"
                elif avg_age > 50 and avg_spending < 40:
                    name = "Comfort Keepers"
                    meaning = "Senior cautious spenders - Traditional marketing"
                else:
                    name = "Regular Customers"
                    meaning = "Balanced customers - Mixed marketing approach"
                
                summary_data.append({
                    "Cluster": cluster,
                    "Segment Name": name,
                    "Size": f"{size} ({pct:.1f}%)",
                    "Avg Age": f"{avg_age:.0f}",
                    "Avg Income": f"${avg_income:.0f}K",
                    "Avg Spending": f"{avg_spending:.0f}",
                    "What This Means": meaning
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        else:
            st.info("Click the Run K-Means button above to see results")
    
    # ========== DBSCAN CLUSTERING ==========
    elif page == "DBSCAN Clustering":
        st.header("DBSCAN Clustering")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            eps_value = st.slider("Epsilon (eps)", 0.1, 2.0, 0.8, 0.05)
            min_samples_value = st.slider("Min Samples", 2, 20, 5)
            if st.button("Run DBSCAN", type="primary"):
                with st.spinner("Running DBSCAN..."):
                    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
                    labels = dbscan.fit_predict(X_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    if n_clusters >= 2:
                        mask = labels != -1
                        sil_score = silhouette_score(X_scaled[mask], labels[mask])
                    else:
                        sil_score = -1
                    st.session_state.dbscan_ran = True
                    st.session_state.dbscan_labels = labels
                    st.session_state.dbscan_clusters = n_clusters
                    st.session_state.dbscan_noise = n_noise
                    st.session_state.dbscan_sil = sil_score
                    st.success(f"Done! Found {n_clusters} clusters")
        
        with col2:
            if st.session_state.dbscan_ran:
                m1, m2, m3 = st.columns(3)
                m1.metric("Clusters Found", st.session_state.dbscan_clusters)
                m2.metric("Noise Points", st.session_state.dbscan_noise)
                if st.session_state.dbscan_sil is not None and st.session_state.dbscan_sil > 0:
                    m3.metric("Silhouette Score", f"{st.session_state.dbscan_sil:.4f}")
                else:
                    m3.metric("Silhouette Score", "N/A")
        
        if st.session_state.dbscan_ran:
            st.subheader("Clustering Results")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_labels = set(st.session_state.dbscan_labels)
            for k in unique_labels:
                mask = st.session_state.dbscan_labels == k
                if k == -1:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label='Noise', alpha=0.5)
                else:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label=f'Cluster {k}')
            ax.set_xlabel("First Principal Component")
            ax.set_ylabel("Second Principal Component")
            ax.set_title("DBSCAN Clustering Results")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            st.subheader("Cluster Distribution")
            cluster_counts = pd.Series(st.session_state.dbscan_labels).value_counts()
            st.bar_chart(cluster_counts)
        else:
            st.info("Click the Run DBSCAN button above to see results")
    
    # ========== METHOD COMPARISON ==========
    elif page == "Method Comparison":
        st.header("Algorithm Comparison")
        
        if st.session_state.kmeans_ran and st.session_state.dbscan_ran:
            kmeans_sil = f"{st.session_state.kmeans_sil:.4f}" if st.session_state.kmeans_sil else "N/A"
            dbscan_sil = f"{st.session_state.dbscan_sil:.4f}" if st.session_state.dbscan_sil and st.session_state.dbscan_sil > 0 else "N/A"
            
            comp_data = pd.DataFrame({
                "Algorithm": ["K-Means", "DBSCAN"],
                "Silhouette Score": [kmeans_sil, dbscan_sil],
                "Number of Segments": [st.session_state.kmeans_k, st.session_state.dbscan_clusters],
                "Noise Points": ["None (0)", f"{st.session_state.dbscan_noise} customers"]
            })
            st.dataframe(comp_data, use_container_width=True)
            
            # Visual comparison
            st.subheader("Visual Comparison")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=st.session_state.kmeans_labels, cmap='viridis', s=50, alpha=0.6)
            axes[0].set_title("K-Means Clustering")
            axes[0].set_xlabel("PC1")
            axes[0].set_ylabel("PC2")
            
            unique_labels = set(st.session_state.dbscan_labels)
            for k in unique_labels:
                mask = st.session_state.dbscan_labels == k
                if k == -1:
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label='Noise', alpha=0.5)
                else:
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label=f'Cluster {k}')
            axes[1].set_title("DBSCAN Clustering")
            axes[1].set_xlabel("PC1")
            axes[1].set_ylabel("PC2")
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please run both K-Means and DBSCAN first")
    
    # ========== CUSTOMER PROFILING ==========
    elif page == "Customer Profiling":
        st.header("Customer Segment Profiles")
        
        if st.session_state.kmeans_ran and st.session_state.kmeans_df is not None:
            st.subheader("Segment Overview")
            profiles = []
            for cluster in range(st.session_state.kmeans_k):
                cluster_data = st.session_state.kmeans_df[st.session_state.kmeans_df['Cluster'] == cluster]
                size = len(cluster_data)
                pct = (size / len(st.session_state.kmeans_df)) * 100
                age = cluster_data['Age'].mean()
                income = cluster_data['Annual_Income'].mean()
                spending = cluster_data['Spending_Score'].mean()
                
                if income > 70 and spending > 60:
                    name = "VIP Premium Customers"
                elif income > 70 and spending < 40:
                    name = "Smart Value Shoppers"
                elif income < 40 and spending > 60:
                    name = "Aspiring Trendsetters"
                elif income < 40 and spending < 40:
                    name = "Practical Frugal"
                elif age < 30 and spending > 60:
                    name = "Young Trend Hunters"
                elif age > 50 and income > 60:
                    name = "Established Affluents"
                elif age > 50 and spending < 40:
                    name = "Comfort Keepers"
                else:
                    name = "Regular Customers"
                
                profiles.append({
                    "Segment": name,
                    "Customers": size,
                    "Percentage": f"{pct:.1f}%",
                    "Avg Age": f"{age:.0f}",
                    "Avg Income": f"${income:.0f}K",
                    "Avg Spending": f"{spending:.0f}"
                })
            st.dataframe(pd.DataFrame(profiles), use_container_width=True)
            
            st.subheader("Export Data")
            csv = st.session_state.kmeans_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Segmentation Results", csv, "segmentation_results.csv")
        else:
            st.warning("Please run K-Means clustering first")
            st.info("Go to K-Means Clustering page and click Run K-Means")

else:
    st.info("Please upload a CSV file to begin")
    st.subheader("Expected CSV Format")
    example = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Gender": ["Male", "Female", "Female"],
        "Age": [25, 35, 42],
        "Annual Income (k$)": [50, 75, 100],
        "Spending Score (1-100)": [60, 75, 85]
    })
    st.dataframe(example)
'''

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Streamlit app created successfully!")
