# Create enhanced streamlit_app.py with tabs and dropdowns
import os
if os.path.exists('streamlit_app.py'):
    os.remove('streamlit_app.py')

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write("""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")

# Initialize session state
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    
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
    
    # Select features
    features = ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Encoded']
    available_features = [f for f in features if f in df_clean.columns]
    
    X = df_clean[available_features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store in session state
    st.session_state.df_clean = df_clean
    st.session_state.X_scaled = X_scaled
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "K-Means", "DBSCAN", "Comparison", "Segment Profiling", "Insights"
    ])
    
    # TAB 1: K-MEANS
    with tab1:
        st.header("K-Means Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            k_value = st.slider("Number of Clusters (k)", 2, 10, 5)
            if st.button("Run K-Means", type="primary"):
                with st.spinner("Running K-Means..."):
                    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)
                    df_clean['KMeans_Cluster'] = labels
                    st.session_state.df_clean = df_clean
                    st.session_state.kmeans_labels = labels
                    
                    sil_score = silhouette_score(X_scaled, labels)
                    cal_score = calinski_harabasz_score(X_scaled, labels)
                    dav_score = davies_bouldin_score(X_scaled, labels)
                    
                    st.session_state.kmeans_silhouette = sil_score
                    st.session_state.kmeans_calinski = cal_score
                    st.session_state.kmeans_davies = dav_score
                    
                    st.success(f"K-Means completed! Silhouette Score: {sil_score:.4f}")
        
        with col2:
            if 'kmeans_labels' in st.session_state:
                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette Score", f"{st.session_state.kmeans_silhouette:.4f}")
                m2.metric("Calinski-Harabasz", f"{st.session_state.kmeans_calinski:.0f}")
                m3.metric("Davies-Bouldin", f"{st.session_state.kmeans_davies:.4f}")
        
        if 'kmeans_labels' in st.session_state:
            st.subheader("Cluster Visualization")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=st.session_state.kmeans_labels, 
                                cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            ax.set_title(f"K-Means Clustering (k={k_value})")
            st.pyplot(fig)
            
            # Cluster sizes
            st.subheader("Cluster Sizes")
            cluster_sizes = df_clean['KMeans_Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_sizes)
    
    # TAB 2: DBSCAN
    with tab2:
        st.header("DBSCAN Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            eps_value = st.slider("Epsilon (eps)", 0.1, 2.0, 0.8, 0.05)
            min_samples_value = st.slider("Min Samples", 2, 20, 5)
            
            if st.button("Run DBSCAN", type="primary"):
                with st.spinner("Running DBSCAN..."):
                    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
                    labels = dbscan.fit_predict(X_scaled)
                    df_clean['DBSCAN_Cluster'] = labels
                    st.session_state.df_clean = df_clean
                    st.session_state.dbscan_labels = labels
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    st.session_state.dbscan_n_clusters = n_clusters
                    st.session_state.dbscan_n_noise = n_noise
                    
                    if n_clusters >= 2:
                        mask = labels != -1
                        sil_score = silhouette_score(X_scaled[mask], labels[mask])
                        st.session_state.dbscan_silhouette = sil_score
                    
                    st.success(f"DBSCAN completed! Found {n_clusters} clusters")
        
        with col2:
            if 'dbscan_labels' in st.session_state:
                m1, m2, m3 = st.columns(3)
                m1.metric("Clusters Found", st.session_state.dbscan_n_clusters)
                m2.metric("Noise Points", st.session_state.dbscan_n_noise)
                if hasattr(st.session_state, 'dbscan_silhouette'):
                    m3.metric("Silhouette Score", f"{st.session_state.dbscan_silhouette:.4f}")
        
        if 'dbscan_labels' in st.session_state:
            st.subheader("Cluster Visualization")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_labels = set(st.session_state.dbscan_labels)
            colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]
                    label = f'Noise ({st.session_state.dbscan_n_noise})'
                else:
                    label = f'Cluster {k}'
                mask = st.session_state.dbscan_labels == k
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], s=50, label=label, alpha=0.6)
            
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("DBSCAN Clustering Results")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
    
    # TAB 3: COMPARISON
    with tab3:
        st.header("Algorithm Comparison")
        
        comparison_data = []
        
        if 'kmeans_labels' in st.session_state:
            comparison_data.append({
                'Algorithm': 'K-Means',
                'Clusters': df_clean['KMeans_Cluster'].nunique(),
                'Silhouette': f"{st.session_state.kmeans_silhouette:.4f}",
                'Calinski-Harabasz': f"{st.session_state.kmeans_calinski:.0f}",
                'Davies-Bouldin': f"{st.session_state.kmeans_davies:.4f}",
                'Noise %': '0%'
            })
        
        if 'dbscan_labels' in st.session_state:
            noise_pct = (st.session_state.dbscan_n_noise / len(X_scaled)) * 100
            comparison_data.append({
                'Algorithm': 'DBSCAN',
                'Clusters': st.session_state.dbscan_n_clusters,
                'Silhouette': f"{st.session_state.dbscan_silhouette:.4f}" if hasattr(st.session_state, 'dbscan_silhouette') else 'N/A',
                'Calinski-Harabasz': 'N/A',
                'Davies-Bouldin': 'N/A',
                'Noise %': f"{noise_pct:.1f}%"
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visual comparison
            st.subheader("Visual Comparison")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            if 'kmeans_labels' in st.session_state:
                axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=st.session_state.kmeans_labels, 
                               cmap='viridis', s=50, alpha=0.6)
                axes[0].set_title(f"K-Means\nSilhouette: {st.session_state.kmeans_silhouette:.3f}")
                axes[0].set_xlabel("PC1")
                axes[0].set_ylabel("PC2")
            
            if 'dbscan_labels' in st.session_state:
                unique_labels = set(st.session_state.dbscan_labels)
                for k in unique_labels:
                    if k == -1:
                        color = 'black'
                        label = 'Noise'
                    else:
                        color = None
                        label = None
                    mask = st.session_state.dbscan_labels == k
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6)
                axes[1].set_title(f"DBSCAN\nClusters: {st.session_state.dbscan_n_clusters}")
                axes[1].set_xlabel("PC1")
                axes[1].set_ylabel("PC2")
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Run K-Means or DBSCAN first to see comparison")
    
    # TAB 4: SEGMENT PROFILING
    with tab4:
        st.header("Customer Segment Profiles")
        
        if 'kmeans_labels' in st.session_state:
            segment_profiles = []
            for cluster in sorted(df_clean['KMeans_Cluster'].unique()):
                cluster_data = df_clean[df_clean['KMeans_Cluster'] == cluster]
                
                size = len(cluster_data)
                percentage = (size / len(df_clean)) * 100
                
                # Determine segment type
                income_mean = cluster_data['Annual_Income'].mean()
                spending_mean = cluster_data['Spending_Score'].mean()
                age_mean = cluster_data['Age'].mean()
                
                if income_mean > 70 and spending_mean > 60:
                    segment_type = "Premium Spenders"
                elif income_mean > 70 and spending_mean < 40:
                    segment_type = "Affluent Frugal"
                elif income_mean < 40 and spending_mean > 60:
                    segment_type = "Budget Enthusiasts"
                elif income_mean < 40 and spending_mean < 40:
                    segment_type = "Cautious Customers"
                else:
                    segment_type = "Moderate Customers"
                
                # Gender distribution
                if 'Gender' in cluster_data.columns:
                    female_pct = (cluster_data['Gender'] == 'Female').sum() / size * 100
                    male_pct = 100 - female_pct
                else:
                    female_pct, male_pct = 50, 50
                
                segment_profiles.append({
                    'Cluster': cluster,
                    'Segment Type': segment_type,
                    'Size': size,
                    'Percentage': f"{percentage:.1f}%",
                    'Avg Age': round(age_mean, 1),
                    'Avg Income': f"${income_mean:.0f}K",
                    'Avg Spending': round(spending_mean, 1),
                    'Female %': f"{female_pct:.0f}%",
                    'Male %': f"{male_pct:.0f}%"
                })
            
            profile_df = pd.DataFrame(segment_profiles)
            st.dataframe(profile_df, use_container_width=True)
            
            # Visualize
            st.subheader("Segment Characteristics")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            ages = [p['Avg Age'] for p in segment_profiles]
            clusters = [p['Cluster'] for p in segment_profiles]
            axes[0, 0].bar(clusters, ages, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Average Age by Segment')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Age')
            
            incomes = [float(p['Avg Income'].replace('$', '').replace('K', '')) for p in segment_profiles]
            axes[0, 1].bar(clusters, incomes, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Average Income by Segment')
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Income (K$)')
            
            spendings = [p['Avg Spending'] for p in segment_profiles]
            axes[1, 0].bar(clusters, spendings, color='salmon', edgecolor='black')
            axes[1, 0].set_title('Average Spending by Segment')
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Spending Score')
            
            sizes = [p['Size'] for p in segment_profiles]
            labels = [f"Cluster {p['Cluster']}" for p in segment_profiles]
            axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Segment Size Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Marketing recommendations
            st.subheader("Marketing Recommendations")
            for profile in segment_profiles:
                with st.expander(f"Segment {profile['Cluster']}: {profile['Segment Type']} ({profile['Size']} customers)"):
                    st.write(f"**Characteristics:**")
                    st.write(f"- Age: {profile['Avg Age']} years")
                    st.write(f"- Income: {profile['Avg Income']}")
                    st.write(f"- Spending: {profile['Avg Spending']}/100")
                    
                    st.write(f"**Recommended Strategy:**")
                    if "Premium" in profile['Segment Type']:
                        st.write("- Offer premium products and exclusive deals")
                        st.write("- VIP loyalty program")
                    elif "Frugal" in profile['Segment Type']:
                        st.write("- Focus on discounts and value bundles")
                        st.write("- Cashback offers")
                    elif "Enthusiasts" in profile['Segment Type']:
                        st.write("- Social media campaigns")
                        st.write("- Influencer partnerships")
                    else:
                        st.write("- Balanced marketing approach")
                        st.write("- Loyalty points program")
        else:
            st.info("Run K-Means clustering first to see segment profiles")
    
    # TAB 5: INSIGHTS
    with tab5:
        st.header("Business Insights & Recommendations")
        
        if 'kmeans_labels' in st.session_state:
            st.subheader("Key Findings")
            
            # Find largest segment
            cluster_sizes = df_clean['KMeans_Cluster'].value_counts()
            largest_cluster = cluster_sizes.idxmax()
            largest_size = cluster_sizes.max()
            
            st.write(f"- **Largest Segment:** Cluster {largest_cluster} with {largest_size} customers ({largest_size/len(df_clean)*100:.1f}%)")
            
            # Find highest spending segment
            spending_by_cluster = df_clean.groupby('KMeans_Cluster')['Spending_Score'].mean()
            highest_spending = spending_by_cluster.idxmax()
            highest_value = spending_by_cluster.max()
            
            st.write(f"- **Highest Spending Segment:** Cluster {highest_spending} (Spending Score: {highest_value:.1f}/100)")
            
            # Find highest income segment
            income_by_cluster = df_clean.groupby('KMeans_Cluster')['Annual_Income'].mean()
            highest_income = income_by_cluster.idxmax()
            highest_income_value = income_by_cluster.max()
            
            st.write(f"- **Highest Income Segment:** Cluster {highest_income} (Income: ${highest_income_value:.0f}K)")
            
            st.subheader("Actionable Recommendations")
            st.markdown("""
            ### 1. Marketing Strategy
            - Target high-value segments with personalized offers
            - Retain loyal customers through rewards programs
            
            ### 2. Product Development
            - Premium products for high-income segments
            - Value bundles for price-sensitive customers
            
            ### 3. Customer Retention
            - Monitor churn risk in decreasing segments
            - Collect feedback from each segment
            
            ### 4. Next Steps
            1. Run A/B tests on different segments
            2. Implement segment-based email campaigns
            3. Update clustering monthly with new data
            """)
            
            # Download results
            st.subheader("Export Results")
            csv = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button("Download Complete Results", csv, "segmentation_results.csv")
        else:
            st.info("Run K-Means clustering first to see insights")

else:
    st.info("Please upload a CSV file to begin")
    
    st.subheader("Expected CSV Format")
    sample = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Gender': ['Male', 'Female', 'Female'],
        'Age': [25, 35, 42],
        'Annual Income (k$)': [50, 75, 100],
        'Spending Score (1-100)': [60, 75, 85]
    })
    st.dataframe(sample)
""")

print(" Streamlit app created successfully!")
