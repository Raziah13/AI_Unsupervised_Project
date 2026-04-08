# FINAL WORKING VERSION - Results ONLY show after clicking Run button
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
    
    # Session state - all start as False/None
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
    if 'dbscan_eps' not in st.session_state:
        st.session_state.dbscan_eps = None
    if 'dbscan_min_samples' not in st.session_state:
        st.session_state.dbscan_min_samples = None
    if 'kmeans_df' not in st.session_state:
        st.session_state.kmeans_df = None
    
    # ========== K-MEANS CLUSTERING ==========
    if page == "K-Means Clustering":
        st.header("K-Means Clustering")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            k_value = st.slider("Number of Clusters (k)", 2, 10, 5)
            run_button = st.button("Run K-Means", type="primary")
            
            if run_button:
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
        
        # ONLY show results if run button was clicked
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
            
            # Segment Summary
            st.subheader("Segment Summary - What Each Cluster Means")
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
            st.info("👈 Click the 'Run K-Means' button to see results")
    
    # ========== DBSCAN CLUSTERING ==========
    elif page == "DBSCAN Clustering":
        st.header("DBSCAN Clustering")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            eps_value = st.slider("Epsilon (eps)", 0.1, 2.0, 0.8, 0.05)
            min_samples_value = st.slider("Min Samples", 2, 20, 5)
            run_button = st.button("Run DBSCAN", type="primary")
            
            if run_button:
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
                    st.session_state.dbscan_eps = eps_value
                    st.session_state.dbscan_min_samples = min_samples_value
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
        
        # ONLY show results if run button was clicked
        if st.session_state.dbscan_ran:
            st.subheader("Clustering Results")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_labels = set(st.session_state.dbscan_labels)
            for k in unique_labels:
                mask = st.session_state.dbscan_labels == k
                if k == -1:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label=f'Noise ({st.session_state.dbscan_noise})', alpha=0.5)
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
            
            # DBSCAN Summary
            st.subheader("DBSCAN Summary - What Each Cluster Means")
            if st.session_state.dbscan_clusters >= 2:
                dbscan_summary = []
                unique_clusters = [c for c in set(st.session_state.dbscan_labels) if c != -1]
                for cluster in unique_clusters:
                    mask = st.session_state.dbscan_labels == cluster
                    cluster_data = df_clean[mask]
                    size = len(cluster_data)
                    pct = (size / len(df_clean)) * 100
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
                    
                    dbscan_summary.append({
                        "Cluster": cluster,
                        "Segment Name": name,
                        "Size": f"{size} ({pct:.1f}%)",
                        "Avg Age": f"{avg_age:.0f}",
                        "Avg Income": f"${avg_income:.0f}K",
                        "Avg Spending": f"{avg_spending:.0f}",
                        "What This Means": meaning
                    })
                st.dataframe(pd.DataFrame(dbscan_summary), use_container_width=True)
                
                if st.session_state.dbscan_noise > 0:
                    st.warning(f"Noise Points: {st.session_state.dbscan_noise} customers ({st.session_state.dbscan_noise/len(df_clean)*100:.1f}%) could not be assigned to any cluster. These customers have unusual behavior patterns.")
            else:
                st.warning("Not enough clusters found for summarization. Try adjusting eps and min_samples parameters.")
        else:
            st.info("👈 Click the 'Run DBSCAN' button to see results")
    
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
            if not st.session_state.kmeans_ran:
                st.info("Go to K-Means Clustering page and click Run K-Means")
            if not st.session_state.dbscan_ran:
                st.info("Go to DBSCAN Clustering page and click Run DBSCAN")
    
    # ========== CUSTOMER PROFILING ==========
    elif page == "Customer Profiling":
        st.header("Customer Segment Profiles")
        
        if st.session_state.kmeans_ran and st.session_state.kmeans_df is not None:
            st.subheader("Segment Overview")
            
            segment_details = []
            
            for cluster in range(st.session_state.kmeans_k):
                cluster_data = st.session_state.kmeans_df[st.session_state.kmeans_df['Cluster'] == cluster]
                
                size = len(cluster_data)
                percentage = (size / len(st.session_state.kmeans_df)) * 100
                age_mean = cluster_data['Age'].mean()
                income_mean = cluster_data['Annual_Income'].mean()
                spending_mean = cluster_data['Spending_Score'].mean()
                
                if 'Gender' in cluster_data.columns:
                    female_pct = (cluster_data['Gender'] == 'Female').sum() / size * 100
                    male_pct = 100 - female_pct
                else:
                    female_pct, male_pct = 50, 50
                
                if income_mean > 70 and spending_mean > 60:
                    segment_name = 'VIP Premium Customers'
                elif income_mean > 70 and spending_mean < 40:
                    segment_name = 'Smart Value Shoppers'
                elif income_mean < 40 and spending_mean > 60:
                    segment_name = 'Aspiring Trendsetters'
                elif income_mean < 40 and spending_mean < 40:
                    segment_name = 'Practical Frugal'
                elif age_mean < 30 and spending_mean > 60:
                    segment_name = 'Young Trend Hunters'
                elif age_mean > 50 and income_mean > 60:
                    segment_name = 'Established Affluents'
                elif age_mean > 50 and spending_mean < 40:
                    segment_name = 'Comfort Keepers'
                else:
                    segment_name = 'Regular Customers'
                
                segment_details.append({
                    'name': segment_name,
                    'size': size,
                    'pct': percentage,
                    'age': age_mean,
                    'income': income_mean,
                    'spending': spending_mean,
                    'female_pct': female_pct,
                    'male_pct': male_pct
                })
            
            table_data = []
            for seg in segment_details:
                table_data.append({
                    'Segment': seg['name'],
                    'Customers': seg['size'],
                    'Percentage': str(round(seg['pct'], 1)) + '%',
                    'Avg Age': str(round(seg['age'], 0)) + ' yrs',
                    'Avg Income': '$' + str(round(seg['income'], 0)) + 'K',
                    'Avg Spending': str(round(seg['spending'], 0)) + '/100',
                    'Gender': str(round(seg['female_pct'], 0)) + '% F / ' + str(round(seg['male_pct'], 0)) + '% M'
                })
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                names = [s['name'][:12] for s in segment_details]
                ages = [s['age'] for s in segment_details]
                ax1.bar(names, ages, color='skyblue', edgecolor='black')
                ax1.set_title('Average Age by Segment', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Segment', fontsize=10)
                ax1.set_ylabel('Age (years)', fontsize=10)
                ax1.tick_params(axis='x', rotation=45, labelsize=8)
                plt.tight_layout()
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                incomes = [s['income'] for s in segment_details]
                ax2.bar(names, incomes, color='lightgreen', edgecolor='black')
                ax2.set_title('Average Income by Segment', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Segment', fontsize=10)
                ax2.set_ylabel('Income ($K)', fontsize=10)
                ax2.tick_params(axis='x', rotation=45, labelsize=8)
                plt.tight_layout()
                st.pyplot(fig2)
            
            with col3:
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                spendings = [s['spending'] for s in segment_details]
                ax3.bar(names, spendings, color='salmon', edgecolor='black')
                ax3.set_title('Average Spending by Segment', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Segment', fontsize=10)
                ax3.set_ylabel('Spending Score', fontsize=10)
                ax3.tick_params(axis='x', rotation=45, labelsize=8)
                plt.tight_layout()
                st.pyplot(fig3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig4, ax4 = plt.subplots(figsize=(7, 6))
                sizes = [s['size'] for s in segment_details]
                labels = [s['name'] for s in segment_details]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#D4A5A5', '#9B59B6', '#3498DB']
                ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(sizes)], startangle=90)
                ax4.set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
                st.pyplot(fig4)
            
            with col2:
                fig5, ax5 = plt.subplots(figsize=(7, 6))
                x = np.arange(len(segment_details))
                width = 0.35
                female_data = [s['female_pct'] for s in segment_details]
                male_data = [s['male_pct'] for s in segment_details]
                ax5.bar(x, female_data, width, label='Female', color='pink', edgecolor='black')
                ax5.bar(x, male_data, width, bottom=female_data, label='Male', color='lightblue', edgecolor='black')
                ax5.set_xlabel('Segment', fontsize=10)
                ax5.set_ylabel('Percentage (%)', fontsize=10)
                ax5.set_title('Gender Distribution by Segment', fontsize=12, fontweight='bold')
                ax5.set_xticks(x)
                ax5.set_xticklabels([s['name'][:10] for s in segment_details], rotation=45, ha='right')
                ax5.legend()
                plt.tight_layout()
                st.pyplot(fig5)
            
            st.subheader('Key Insights')
            
            largest = max(segment_details, key=lambda x: x['size'])
            highest_spending = max(segment_details, key=lambda x: x['spending'])
            highest_income = max(segment_details, key=lambda x: x['income'])
            youngest = min(segment_details, key=lambda x: x['age'])
            oldest = max(segment_details, key=lambda x: x['age'])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.info('**Largest**\\n\\n' + largest['name'] + '\\n' + str(largest['size']) + ' customers')
            col2.success('**Biggest Spenders**\\n\\n' + highest_spending['name'] + '\\nScore: ' + str(round(highest_spending['spending'], 0)) + '/100')
            col3.warning('**Highest Income**\\n\\n' + highest_income['name'] + '\\n$' + str(round(highest_income['income'], 0)) + 'K')
            col4.info('**Youngest**\\n\\n' + youngest['name'] + '\\n' + str(round(youngest['age'], 0)) + ' years')
            col5.success('**Oldest**\\n\\n' + oldest['name'] + '\\n' + str(round(oldest['age'], 0)) + ' years')
            
            st.subheader('Export Data')
            csv = st.session_state.kmeans_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download Segmentation Results (CSV)', csv, 'segmentation_results.csv')
        else:
            st.warning('Please run K-Means clustering first')
            st.info('Go to K-Means Clustering page and click Run K-Means')

else:
    st.info("Please upload a CSV file to begin")
    
    st.subheader("Expected CSV Format")
    example = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Gender': ['Male', 'Female', 'Female'],
        'Age': [25, 35, 42],
        'Annual Income (k$)': [50, 75, 100],
        'Spending Score (1-100)': [60, 75, 85]
    })
    st.dataframe(example)
'''

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Streamlit app created successfully!")
