# Create enhanced streamlit_app.py with tabs and dropdowns
import os
if os.path.exists('streamlit_app.py'):
    os.remove('streamlit_app.py')

with open('streamlit_app.py', 'w') as f:
    f.write('import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title(" Customer Segmentation Dashboard")

# Initialize session state
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'features' not in st.session_state:
    st.session_state.features = None

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
        gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    else:
        df_clean['Gender_Encoded'] = 0
        gender_mapping = {}
    
    # Select features
    available_features = ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Encoded']
    features = [f for f in available_features if f in df_clean.columns]
    
    X = df_clean[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store in session state
    st.session_state.df_clean = df_clean
    st.session_state.X_scaled = X_scaled
    st.session_state.features = features
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " K-Means", 
        " DBSCAN", 
        " Hierarchical", 
        " Method Comparison", 
        " Segment Profiling",
        " Insights & Reports"
    ])
    
    # ============================================
    # TAB 1: K-MEANS CLUSTERING
    # ============================================
    with tab1:
        st.header("K-Means Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            k_value = st.slider("Number of Clusters (k)", 2, 10, 5, key="kmeans_k")
            init_method = st.selectbox("Initialization", ["k-means++", "random"], key="init")
            max_iter = st.slider("Max Iterations", 100, 500, 300, key="iter")
            
            if st.button("Run K-Means", type="primary", key="run_kmeans"):
                with st.spinner("Running K-Means..."):
                    kmeans = KMeans(n_clusters=k_value, init=init_method, 
                                   max_iter=max_iter, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(st.session_state.X_scaled)
                    st.session_state.df_clean['KMeans_Cluster'] = labels
                    st.session_state.kmeans_labels = labels
                    st.session_state.kmeans_model = kmeans
                    
                    # Calculate metrics
                    sil_score = silhouette_score(st.session_state.X_scaled, labels)
                    cal_score = calinski_harabasz_score(st.session_state.X_scaled, labels)
                    dav_score = davies_bouldin_score(st.session_state.X_scaled, labels)
                    
                    st.session_state.kmeans_silhouette = sil_score
                    st.session_state.kmeans_calinski = cal_score
                    st.session_state.kmeans_davies = dav_score
                    
                    st.success("K-Means completed!")
        
        with col2:
            if 'kmeans_labels' in st.session_state:
                # Metrics display
                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette Score", f"{st.session_state.kmeans_silhouette:.4f}", 
                         help="Higher is better (-1 to 1)")
                m2.metric("Calinski-Harabasz", f"{st.session_state.kmeans_calinski:.0f}", 
                         help="Higher is better")
                m3.metric("Davies-Bouldin", f"{st.session_state.kmeans_davies:.4f}", 
                         help="Lower is better")
                
                # Cluster distribution
                st.write(f"**Cluster Sizes:**")
                cluster_sizes = st.session_state.df_clean['KMeans_Cluster'].value_counts().sort_index()
                st.bar_chart(cluster_sizes)
        
        # Visualization
        if 'kmeans_labels' in st.session_state:
            st.subheader("Cluster Visualization")
            
            viz_type = st.radio("Visualization Type:", ["PCA (2D)", "PCA (3D)", "t-SNE"], 
                               horizontal=True, key="kmeans_viz")
            
            if viz_type == "PCA (2D)":
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(st.session_state.X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                    c=st.session_state.kmeans_labels, 
                                    cmap='viridis', s=100, alpha=0.6)
                plt.colorbar(scatter)
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
                ax.set_title(f"K-Means Clustering (k={k_value})")
                st.pyplot(fig)
                
            elif viz_type == "PCA (3D)":
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(st.session_state.X_scaled)
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=st.session_state.kmeans_labels, 
                               colorscale='Viridis', showscale=True)
                )])
                fig.update_layout(title="3D PCA Visualization", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                tsne = TSNE(n_components=2, random_state=42)
                X_tsne = tsne.fit_transform(st.session_state.X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                    c=st.session_state.kmeans_labels, 
                                    cmap='viridis', s=100, alpha=0.6)
                plt.colorbar(scatter)
                ax.set_title("t-SNE Visualization")
                st.pyplot(fig)
    
    # ============================================
    # TAB 2: DBSCAN CLUSTERING
    # ============================================
    with tab2:
        st.header("DBSCAN Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            eps_value = st.slider("Epsilon (eps)", 0.1, 2.0, 0.8, 0.05, key="eps")
            min_samples_value = st.slider("Min Samples", 2, 20, 5, key="min_samples")
            
            if st.button("Run DBSCAN", type="primary", key="run_dbscan"):
                with st.spinner("Running DBSCAN..."):
                    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
                    labels = dbscan.fit_predict(st.session_state.X_scaled)
                    st.session_state.df_clean['DBSCAN_Cluster'] = labels
                    st.session_state.dbscan_labels = labels
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    st.session_state.dbscan_n_clusters = n_clusters
                    st.session_state.dbscan_n_noise = n_noise
                    
                    if n_clusters >= 2:
                        mask = labels != -1
                        sil_score = silhouette_score(st.session_state.X_scaled[mask], labels[mask])
                        st.session_state.dbscan_silhouette = sil_score
                    
                    st.success(f"DBSCAN completed! Found {n_clusters} clusters")
        
        with col2:
            if 'dbscan_labels' in st.session_state:
                m1, m2, m3 = st.columns(3)
                m1.metric("Clusters Found", st.session_state.dbscan_n_clusters)
                m2.metric("Noise Points", st.session_state.dbscan_n_noise)
                m3.metric("Noise %", f"{(st.session_state.dbscan_n_noise/len(st.session_state.dbscan_labels))*100:.1f}%")
                
                if hasattr(st.session_state, 'dbscan_silhouette'):
                    st.metric("Silhouette Score", f"{st.session_state.dbscan_silhouette:.4f}")
        
        if 'dbscan_labels' in st.session_state:
            st.subheader("Cluster Visualization")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(st.session_state.X_scaled)
            
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
    
    # ============================================
    # TAB 3: HIERARCHICAL CLUSTERING
    # ============================================
    with tab3:
        st.header("Hierarchical (Agglomerative) Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_clusters_hc = st.slider("Number of Clusters", 2, 10, 5, key="hc_k")
            linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], key="linkage")
            
            if st.button("Run Hierarchical Clustering", type="primary", key="run_hc"):
                with st.spinner("Running Hierarchical Clustering..."):
                    hc = AgglomerativeClustering(n_clusters=n_clusters_hc, linkage=linkage)
                    labels = hc.fit_predict(st.session_state.X_scaled)
                    st.session_state.df_clean['Hierarchical_Cluster'] = labels
                    st.session_state.hc_labels = labels
                    
                    sil_score = silhouette_score(st.session_state.X_scaled, labels)
                    st.session_state.hc_silhouette = sil_score
                    
                    st.success(f"Hierarchical clustering completed! Silhouette: {sil_score:.4f}")
        
        with col2:
            if 'hc_labels' in st.session_state:
                st.metric("Silhouette Score", f"{st.session_state.hc_silhouette:.4f}")
                
                cluster_sizes = st.session_state.df_clean['Hierarchical_Cluster'].value_counts().sort_index()
                st.bar_chart(cluster_sizes)
        
        if 'hc_labels' in st.session_state:
            st.subheader("Dendrogram (Simplified)")
            st.info("Full dendrogram requires scipy.cluster.hierarchy")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(st.session_state.X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=st.session_state.hc_labels, 
                                cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter)
            ax.set_title(f"Hierarchical Clustering (k={n_clusters_hc})")
            st.pyplot(fig)
    
    # ============================================
    # TAB 4: METHOD COMPARISON
    # ============================================
    with tab4:
        st.header("Algorithm Comparison")
        
        comparison_data = []
        
        # K-Means metrics
        if 'kmeans_labels' in st.session_state:
            comparison_data.append({
                'Algorithm': 'K-Means',
                'Clusters': st.session_state.df_clean['KMeans_Cluster'].nunique(),
                'Silhouette': f"{st.session_state.kmeans_silhouette:.4f}",
                'Calinski-Harabasz': f"{st.session_state.kmeans_calinski:.0f}",
                'Davies-Bouldin': f"{st.session_state.kmeans_davies:.4f}",
                'Noise %': '0%'
            })
        
        # DBSCAN metrics
        if 'dbscan_labels' in st.session_state:
            comparison_data.append({
                'Algorithm': 'DBSCAN',
                'Clusters': st.session_state.dbscan_n_clusters,
                'Silhouette': f"{st.session_state.dbscan_silhouette:.4f}" if hasattr(st.session_state, 'dbscan_silhouette') else 'N/A',
                'Calinski-Harabasz': 'N/A',
                'Davies-Bouldin': 'N/A',
                'Noise %': f"{(st.session_state.dbscan_n_noise/len(st.session_state.dbscan_labels))*100:.1f}%"
            })
        
        # Hierarchical metrics
        if 'hc_labels' in st.session_state:
            comparison_data.append({
                'Algorithm': 'Hierarchical',
                'Clusters': st.session_state.df_clean['Hierarchical_Cluster'].nunique(),
                'Silhouette': f"{st.session_state.hc_silhouette:.4f}",
                'Calinski-Harabasz': 'N/A',
                'Davies-Bouldin': 'N/A',
                'Noise %': '0%'
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visual comparison
            st.subheader("Visual Comparison")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(st.session_state.X_scaled)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # K-Means
            if 'kmeans_labels' in st.session_state:
                axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=st.session_state.kmeans_labels, 
                               cmap='viridis', s=50, alpha=0.6)
                axes[0].set_title(f"K-Means\nSilhouette: {st.session_state.kmeans_silhouette:.3f}")
            
            # DBSCAN
            if 'dbscan_labels' in st.session_state:
                unique_labels = set(st.session_state.dbscan_labels)
                for k in unique_labels:
                    if k == -1:
                        color = 'black'
                        label = 'Noise'
                    else:
                        color = None
                        label = f'Cluster {k}'
                    mask = st.session_state.dbscan_labels == k
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                   s=50, alpha=0.6, label=label if k == -1 else None)
                axes[1].set_title(f"DBSCAN\nClusters: {st.session_state.dbscan_n_clusters}")
            
            # Hierarchical
            if 'hc_labels' in st.session_state:
                axes[2].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=st.session_state.hc_labels, 
                               cmap='viridis', s=50, alpha=0.6)
                axes[2].set_title(f"Hierarchical\nSilhouette: {st.session_state.hc_silhouette:.3f}")
            
            for ax in axes:
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Run at least one algorithm to see comparison")
    
    # ============================================
    # TAB 5: SEGMENT PROFILING
    # ============================================
    with tab5:
        st.header("Customer Segment Profiles")
        
        # Choose which clustering to profile
        cluster_method = st.selectbox(
            "Select Clustering Method for Profiling",
            ["K-Means", "DBSCAN", "Hierarchical"],
            key="profile_method"
        )
        
        if cluster_method == "K-Means" and 'kmeans_labels' in st.session_state:
            cluster_col = 'KMeans_Cluster'
            n_clusters = st.session_state.df_clean[cluster_col].nunique()
        elif cluster_method == "DBSCAN" and 'dbscan_labels' in st.session_state:
            cluster_col = 'DBSCAN_Cluster'
            n_clusters = st.session_state.dbscan_n_clusters
        elif cluster_method == "Hierarchical" and 'hc_labels' in st.session_state:
            cluster_col = 'Hierarchical_Cluster'
            n_clusters = st.session_state.df_clean[cluster_col].nunique()
        else:
            st.warning(f"Please run {cluster_method} clustering first")
            st.stop()
        
        # Create segment profiles
        segment_profiles = []
        for cluster in sorted(st.session_state.df_clean[cluster_col].unique()):
            cluster_data = st.session_state.df_clean[st.session_state.df_clean[cluster_col] == cluster]
            
            size = len(cluster_data)
            percentage = (size / len(st.session_state.df_clean)) * 100
            
            # Determine segment type
            income_mean = cluster_data['Annual_Income'].mean() if 'Annual_Income' in cluster_data.columns else 0
            spending_mean = cluster_data['Spending_Score'].mean() if 'Spending_Score' in cluster_data.columns else 0
            age_mean = cluster_data['Age'].mean() if 'Age' in cluster_data.columns else 0
            
            if income_mean > 70 and spending_mean > 60:
                segment_type = "Premium Spenders"
                icon = ""
                color = "#FF6B6B"
            elif income_mean > 70 and spending_mean < 40:
                segment_type = "Affluent Frugal"
                icon = ""
                color = "#4ECDC4"
            elif income_mean < 40 and spending_mean > 60:
                segment_type = "Budget Enthusiasts"
                icon = ""
                color = "#45B7D1"
            elif income_mean < 40 and spending_mean < 40:
                segment_type = "Cautious Customers"
                icon = ""
                color = "#96CEB4"
            else:
                segment_type = "Moderate Customers"
                icon = ""
                color = "#FFEAA7"
            
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
        
        # Visualize profiles
        st.subheader("Segment Characteristics")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Age by segment
        ages = [p['Avg Age'] for p in segment_profiles]
        clusters = [p['Cluster'] for p in segment_profiles]
        axes[0, 0].bar(clusters, ages, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Average Age by Segment')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Age')
        
        # Income by segment
        incomes = [float(p['Avg Income'].replace('$', '').replace('K', '')) for p in segment_profiles]
        axes[0, 1].bar(clusters, incomes, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Average Income by Segment')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Income (K$)')
        
        # Spending by segment
        spendings = [p['Avg Spending'] for p in segment_profiles]
        axes[1, 0].bar(clusters, spendings, color='salmon', edgecolor='black')
        axes[1, 0].set_title('Average Spending by Segment')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Spending Score')
        
        # Segment size pie chart
        sizes = [p['Size'] for p in segment_profiles]
        labels = [f"Cluster {p['Cluster']}\n{p['Segment Type']}" for p in segment_profiles]
        axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Segment Size Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Marketing recommendations
        st.subheader("Marketing Recommendations")
        for profile in segment_profiles:
            with st.expander(f"Segment {profile['Cluster']}: {profile['Segment Type']} ({profile['Size']} customers)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Characteristics:**")
                    st.write(f"- Age: {profile['Avg Age']} years")
                    st.write(f"- Income: {profile['Avg Income']}")
                    st.write(f"- Spending: {profile['Avg Spending']}/100")
                    st.write(f"- Gender: {profile['Female %']} Female, {profile['Male %']} Male")
                
                with col2:
                    st.write(f"**Recommended Strategy:**")
                    if "Premium" in profile['Segment Type']:
                        st.write("- Offer premium products and exclusive deals")
                        st.write("- Early access to new collections")
                        st.write("- VIP loyalty program")
                    elif "Frugal" in profile['Segment Type']:
                        st.write("- Focus on discounts and value bundles")
                        st.write("- Price-match guarantees")
                        st.write("- Cashback offers")
                    elif "Enthusiasts" in profile['Segment Type']:
                        st.write("- Social media campaigns")
                        st.write("- Influencer partnerships")
                        st.write("- Flash sales and limited editions")
                    else:
                        st.write("- Balanced marketing approach")
                        st.write("- Loyalty points program")
                        st.write("- Email newsletters")
    
    # ============================================
    # TAB 6: INSIGHTS & REPORTS
    # ============================================
    with tab6:
        st.header("Business Insights & Recommendations")
        
        st.subheader("Key Findings")
        
        # Best performing algorithm
        if 'kmeans_silhouette' in st.session_state and 'dbscan_silhouette' in st.session_state:
            if st.session_state.kmeans_silhouette > st.session_state.dbscan_silhouette:
                best_algo = "K-Means"
                best_score = st.session_state.kmeans_silhouette
            else:
                best_algo = "DBSCAN"
                best_score = st.session_state.dbscan_silhouette
            
            st.info(f"**Best Algorithm:** {best_algo} (Silhouette Score: {best_score:.4f})")
        
        # Customer insights
        if 'kmeans_labels' in st.session_state:
            st.subheader("Customer Segment Summary")
            
            # Find largest segment
            cluster_sizes = st.session_state.df_clean['KMeans_Cluster'].value_counts()
            largest_cluster = cluster_sizes.idxmax()
            largest_size = cluster_sizes.max()
            
            st.write(f"- **Largest Segment:** Cluster {largest_cluster} with {largest_size} customers ({largest_size/len(st.session_state.df_clean)*100:.1f}%)")
            
            # Find highest spending segment
            spending_by_cluster = st.session_state.df_clean.groupby('KMeans_Cluster')['Spending_Score'].mean()
            highest_spending = spending_by_cluster.idxmax()
            highest_value = spending_by_cluster.max()
            
            st.write(f"- **Highest Spending Segment:** Cluster {highest_spending} (Spending Score: {highest_value:.1f}/100)")
            
            # Find highest income segment
            income_by_cluster = st.session_state.df_clean.groupby('KMeans_Cluster')['Annual_Income'].mean()
            highest_income = income_by_cluster.idxmax()
            highest_income_value = income_by_cluster.max()
            
            st.write(f"- **Highest Income Segment:** Cluster {highest_income} (Income: ${highest_income_value:.0f}K)")
        
        st.subheader("Actionable Recommendations")
        
        st.markdown("""
        ### 1. Marketing Strategy
        - **Target high-value segments** with personalized offers
        - **Retain loyal customers** through rewards programs
        - **Re-engage dormant segments** with reactivation campaigns
        
        ### 2. Product Development
        - **Premium products** for high-income segments
        - **Value bundles** for price-sensitive customers
        - **New customer acquisition** for growing segments
        
        ### 3. Customer Retention
        - **Monitor churn risk** in decreasing segments
        - **Feedback collection** from each segment
        - **Service improvement** based on segment needs
        
        ### 4. Next Steps
        1. Run A/B tests on different segments
        2. Implement segment-based email campaigns
        3. Track segment movement over time
        4. Update clustering monthly with new data
        """)
        
        # Download all results
        st.subheader("Export All Results")
        csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("Download Complete Dataset with Clusters", csv, "complete_segmentation_results.csv")
    
else:
    # No file uploaded
    st.info(" Please upload a CSV file to begin")
    
    st.subheader("How to Use This Dashboard")
    st.markdown("""
    ### Steps:
    1. **Upload** your customer dataset (CSV format)
    2. **Choose** a clustering algorithm from the tabs above
    3. **Adjust** parameters using the sidebar controls
    4. **Compare** different algorithms in the Comparison tab
    5. **View** detailed segment profiles and insights
    
    ### Expected CSV Format:
    - CustomerID (optional)
    - Gender (Male/Female)
    - Age (numeric)
    - Annual Income (numeric)
    - Spending Score (numeric 1-100)
    
    ### Sample Data:
    """)
    
    sample = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'Age': [25, 35, 42, 28, 50],
        'Annual Income (k$)': [50, 75, 100, 60, 80],
        'Spending Score (1-100)': [60, 75, 85, 70, 65]
    })
    st.dataframe(sample)
    
    st.markdown("""
    ### Features:
    - **3 Clustering Algorithms** (K-Means, DBSCAN, Hierarchical)
    - **Interactive Visualizations** (2D/3D PCA, t-SNE)
    - **Method Comparison** with multiple metrics
    - **Segment Profiling** with marketing recommendations
    - **Export Results** for further analysis
    """)

print(" Enhanced Streamlit app created with tabs and dropdowns!")
