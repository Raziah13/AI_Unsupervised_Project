# Create fixed streamlit_app.py - NO multi-line f-strings
import os
if os.path.exists('streamlit_app.py'):
    os.remove('streamlit_app.py')

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
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

# Create dropdown menu in sidebar
st.sidebar.subheader("Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["K-Means Clustering", "DBSCAN Clustering", "Method Comparison", "Customer Profiling"]
)

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
    # K-MEANS CLUSTERING PAGE
    # ============================================
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
                    
                    st.session_state.kmeans_labels = labels
                    st.session_state.kmeans_sil = sil_score
                    st.session_state.kmeans_cal = cal_score
                    st.session_state.kmeans_dav = dav_score
                    st.session_state.kmeans_k = k_value
                    st.session_state.df_clean = df_clean
                    
                    st.success("Done! Silhouette Score: " + format(sil_score, '.4f'))
        
        with col2:
            if 'kmeans_labels' in st.session_state:
                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette Score", format(st.session_state.kmeans_sil, '.4f'))
                m2.metric("Calinski-Harabasz", format(st.session_state.kmeans_cal, '.0f'))
                m3.metric("Davies-Bouldin", format(st.session_state.kmeans_dav, '.4f'))
        
        if 'kmeans_labels' in st.session_state:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=st.session_state.kmeans_labels, 
                                cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter)
            ax.set_xlabel("PC1 (" + format(pca.explained_variance_ratio_[0], '.2%') + ")")
            ax.set_ylabel("PC2 (" + format(pca.explained_variance_ratio_[1], '.2%') + ")")
            ax.set_title("K-Means Clustering with " + str(st.session_state.kmeans_k) + " Segments")
            st.pyplot(fig)
            
            st.subheader("Customer Segment Profiles")
            
            segment_profiles = []
            for cluster in range(st.session_state.kmeans_k):
                cluster_data = st.session_state.df_clean[st.session_state.df_clean['Cluster'] == cluster]
                
                size = len(cluster_data)
                age_mean = cluster_data['Age'].mean()
                income_mean = cluster_data['Annual_Income'].mean()
                spending_mean = cluster_data['Spending_Score'].mean()
                
                if income_mean > 70 and spending_mean > 60:
                    segment_name = "VIP Premium Customers"
                elif income_mean > 70 and spending_mean < 40:
                    segment_name = "Smart Value Shoppers"
                elif income_mean < 40 and spending_mean > 60:
                    segment_name = "Aspiring Trendsetters"
                elif income_mean < 40 and spending_mean < 40:
                    segment_name = "Practical Frugal"
                elif age_mean < 30 and spending_mean > 60:
                    segment_name = "Young Trend Hunters"
                elif age_mean > 50 and income_mean > 60:
                    segment_name = "Established Affluents"
                elif age_mean > 50 and spending_mean < 40:
                    segment_name = "Comfort Keepers"
                else:
                    segment_name = "Regular Customers"
                
                segment_profiles.append({
                    'Segment Name': segment_name,
                    'Size': size,
                    'Percentage': format(size/len(st.session_state.df_clean)*100, '.1f') + '%',
                    'Avg Age': format(age_mean, '.0f') + ' years',
                    'Avg Income': '$' + format(income_mean, '.0f') + 'K',
                    'Avg Spending': format(spending_mean, '.0f') + '/100'
                })
            
            profile_df = pd.DataFrame(segment_profiles)
            st.dataframe(profile_df, use_container_width=True)
            
            st.subheader("Segment Size Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sizes = [p['Size'] for p in segment_profiles]
            labels = [p['Segment Name'] for p in segment_profiles]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#D4A5A5', '#9B59B6', '#3498DB']
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(sizes)])
            ax2.set_title('Customer Segment Distribution')
            st.pyplot(fig2)
    
    # ============================================
    # DBSCAN CLUSTERING PAGE
    # ============================================
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
                    df_clean['DBSCAN_Cluster'] = labels
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    st.session_state.dbscan_labels = labels
                    st.session_state.dbscan_clusters = n_clusters
                    st.session_state.dbscan_noise = n_noise
                    st.session_state.df_clean = df_clean
                    
                    if n_clusters >= 2:
                        mask = labels != -1
                        sil_score = silhouette_score(X_scaled[mask], labels[mask])
                        st.session_state.dbscan_sil = sil_score
                        st.success("Done! Found " + str(n_clusters) + " clusters, Silhouette: " + format(sil_score, '.4f'))
                    else:
                        st.warning("Found only " + str(n_clusters) + " clusters. Try adjusting eps or min_samples")
        
        with col2:
            if 'dbscan_labels' in st.session_state:
                m1, m2, m3 = st.columns(3)
                m1.metric("Clusters Found", st.session_state.dbscan_clusters)
                m2.metric("Noise Points", st.session_state.dbscan_noise)
                if 'dbscan_sil' in st.session_state:
                    m3.metric("Silhouette Score", format(st.session_state.dbscan_sil, '.4f'))
        
        if 'dbscan_labels' in st.session_state:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_labels = set(st.session_state.dbscan_labels)
            
            for k in unique_labels:
                mask = st.session_state.dbscan_labels == k
                if k == -1:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label='Noise (' + str(st.session_state.dbscan_noise) + ')', alpha=0.5)
                else:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label='Cluster ' + str(k))
            
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("DBSCAN Clustering")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            st.info("Note: Black dots are 'Noise' - customers that don't fit well into any cluster")
    
    # ============================================
    # METHOD COMPARISON PAGE
    # ============================================
    elif page == "Method Comparison":
        st.header("Algorithm Comparison")
        
        if 'kmeans_labels' in st.session_state and 'dbscan_labels' in st.session_state:
            comparison_data = {
                'Algorithm': ['K-Means', 'DBSCAN'],
                'Silhouette Score': [format(st.session_state.kmeans_sil, '.4f'), format(st.session_state.dbscan_sil, '.4f') if 'dbscan_sil' in st.session_state else 'N/A'],
                'Number of Segments': [st.session_state.kmeans_k, st.session_state.dbscan_clusters],
                'Noise Points': ['None (0)', str(st.session_state.dbscan_noise) + ' customers']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            st.subheader("Recommendation")
            
            if st.session_state.kmeans_sil > 0.5 and st.session_state.dbscan_sil > 0.5:
                st.success("Both algorithms perform well!")
            elif st.session_state.kmeans_sil > st.session_state.dbscan_sil:
                st.success("K-Means performs better for your data")
            else:
                st.success("DBSCAN performs better for your data")
            
            st.subheader("Visual Comparison")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                       c=st.session_state.kmeans_labels, 
                                       cmap='viridis', s=50, alpha=0.6)
            axes[0].set_title('K-Means - ' + str(st.session_state.kmeans_k) + ' Segments')
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            plt.colorbar(scatter1, ax=axes[0])
            
            unique_labels = set(st.session_state.dbscan_labels)
            for k in unique_labels:
                mask = st.session_state.dbscan_labels == k
                if k == -1:
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label='Noise', alpha=0.5)
                else:
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label='Cluster ' + str(k))
            axes[1].set_title('DBSCAN - ' + str(st.session_state.dbscan_clusters) + ' Segments')
            axes[1].set_xlabel('PC1')
            axes[1].set_ylabel('PC2')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning("Please run both K-Means and DBSCAN first")
    
    # ============================================
    # CUSTOMER PROFILING PAGE
    # ============================================
    elif page == "Customer Profiling":
        st.header("Customer Segment Profiles")
        
        if 'kmeans_labels' in st.session_state:
            st.subheader("Understanding Your Customer Segments")
            
            segment_details = []
            
            for cluster in range(st.session_state.kmeans_k):
                cluster_data = st.session_state.df_clean[st.session_state.df_clean['Cluster'] == cluster]
                
                size = len(cluster_data)
                percentage = (size / len(st.session_state.df_clean)) * 100
                age_mean = cluster_data['Age'].mean()
                income_mean = cluster_data['Annual_Income'].mean()
                spending_mean = cluster_data['Spending_Score'].mean()
                
                if 'Gender' in cluster_data.columns:
                    female_pct = (cluster_data['Gender'] == 'Female').sum() / size * 100
                    male_pct = 100 - female_pct
                else:
                    female_pct, male_pct = 50, 50
                
                if income_mean > 70 and spending_mean > 60:
                    segment_name = "VIP Premium Customers"
                    marketing = "Offer exclusive products, early access, VIP events"
                elif income_mean > 70 and spending_mean < 40:
                    segment_name = "Smart Value Shoppers"
                    marketing = "Highlight discounts, cashback, value deals"
                elif income_mean < 40 and spending_mean > 60:
                    segment_name = "Aspiring Trendsetters"
                    marketing = "Social media campaigns, influencer marketing"
                elif income_mean < 40 and spending_mean < 40:
                    segment_name = "Practical Frugal"
                    marketing = "Focus on essential needs, basic value"
                elif age_mean < 30 and spending_mean > 60:
                    segment_name = "Young Trend Hunters"
                    marketing = "Flash sales, limited editions, viral marketing"
                elif age_mean > 50 and income_mean > 60:
                    segment_name = "Established Affluents"
                    marketing = "Quality focus, reliability, service excellence"
                elif age_mean > 50 and spending_mean < 40:
                    segment_name = "Comfort Keepers"
                    marketing = "Trust, familiarity, loyalty rewards"
                else:
                    segment_name = "Regular Customers"
                    marketing = "Mix of offers, loyalty program, referrals"
                
                segment_details.append({
                    'name': segment_name,
                    'size': size,
                    'pct': percentage,
                    'age': age_mean,
                    'income': income_mean,
                    'spending': spending_mean,
                    'female_pct': female_pct,
                    'male_pct': male_pct,
                    'marketing': marketing
                })
            
            for seg in segment_details:
                with st.expander(seg['name'] + " - " + str(seg['size']) + " customers (" + format(seg['pct'], '.1f') + "%)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Customer Characteristics:**")
                        st.write("- Age: " + format(seg['age'], '.0f') + " years")
                        st.write("- Income: $" + format(seg['income'], '.0f') + "K")
                        st.write("- Spending Score: " + format(seg['spending'], '.0f') + "/100")
                        st.write("- Gender: " + format(seg['female_pct'], '.0f') + "% Female / " + format(seg['male_pct'], '.0f') + "% Male")
                    
                    with col2:
                        st.write("**Marketing Strategy:**")
                        st.write("- " + seg['marketing'])
            
            st.subheader("Quick Summary")
            
            largest = max(segment_details, key=lambda x: x['size'])
            highest_spending = max(segment_details, key=lambda x: x['spending'])
            highest_income = max(segment_details, key=lambda x: x['income'])
            
            col1, col2, col3 = st.columns(3)
            col1.info("**Largest Segment**\n\n" + largest['name'] + "\n" + str(largest['size']) + " customers")
            col2.success("**Highest Spenders**\n\n" + highest_spending['name'] + "\nSpending: " + format(highest_spending['spending'], '.0f') + "/100")
            col3.warning("**Highest Income**\n\n" + highest_income['name'] + "\nIncome: $" + format(highest_income['income'], '.0f') + "K")
            
            st.subheader("Export Data")
            csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
            st.download_button("Download Customer Segmentation Results", csv, "segmentation_results.csv")
            
        else:
            st.warning("Please run K-Means clustering first")

else:
    st.info("Please upload a CSV file to begin")
    
    st.subheader("Expected CSV Format")
    example = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'Age': [25, 35, 42, 28, 50],
        'Annual Income (k$)': [50, 75, 100, 60, 80],
        'Spending Score (1-100)': [60, 75, 85, 70, 65]
    })
    st.dataframe(example)
    
    st.subheader("How to Use")
    st.markdown("""
    1. Upload your customer CSV file
    2. Select analysis type from dropdown menu
    3. Run clustering algorithm
    4. View customer segments with meaningful names
    5. Download results for your marketing team
    """)
''')

print("Streamlit app created successfully!")
