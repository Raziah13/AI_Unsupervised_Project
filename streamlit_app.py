# Create improved streamlit_app.py with dropdown and better cluster names
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
st.title(" Customer Segmentation Dashboard")

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
        st.header(" K-Means Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            k_value = st.slider("Number of Clusters (k)", 2, 10, 5)
            if st.button("Run K-Means", type="primary"):
                with st.spinner("Running K-Means..."):
                    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)
                    df_clean['Cluster'] = labels
                    
                    # Calculate metrics
                    sil_score = silhouette_score(X_scaled, labels)
                    cal_score = calinski_harabasz_score(X_scaled, labels)
                    dav_score = davies_bouldin_score(X_scaled, labels)
                    
                    st.session_state.kmeans_labels = labels
                    st.session_state.kmeans_sil = sil_score
                    st.session_state.kmeans_cal = cal_score
                    st.session_state.kmeans_dav = dav_score
                    st.session_state.kmeans_k = k_value
                    st.session_state.df_clean = df_clean
                    
                    st.success(f" Done! Silhouette Score: {sil_score:.4f}")
        
        with col2:
            if 'kmeans_labels' in st.session_state:
                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette Score", f"{st.session_state.kmeans_sil:.4f}")
                m2.metric("Calinski-Harabasz", f"{st.session_state.kmeans_cal:.0f}")
                m3.metric("Davies-Bouldin", f"{st.session_state.kmeans_dav:.4f}")
        
        if 'kmeans_labels' in st.session_state:
            # PCA Visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=st.session_state.kmeans_labels, 
                                cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            ax.set_title(f"K-Means Clustering with {st.session_state.kmeans_k} Segments")
            st.pyplot(fig)
            
            # Show cluster profiles with meaningful names
            st.subheader(" Customer Segment Profiles")
            
            segment_profiles = []
            for cluster in range(st.session_state.kmeans_k):
                cluster_data = st.session_state.df_clean[st.session_state.df_clean['Cluster'] == cluster]
                
                size = len(cluster_data)
                age_mean = cluster_data['Age'].mean()
                income_mean = cluster_data['Annual_Income'].mean()
                spending_mean = cluster_data['Spending_Score'].mean()
                
                # Give meaningful names based on characteristics
                if income_mean > 70 and spending_mean > 60:
                    segment_name = "VIP Premium Customers"
                    icon = "👑"
                    description = "High income, high spending"
                elif income_mean > 70 and spending_mean < 40:
                    segment_name = "Value Shoppers"
                    icon = "💰"
                    description = "High income, careful spending"
                elif income_mean < 40 and spending_mean > 60:
                    segment_name = "Young Enthusiasts"
                    icon = "🎯"
                    description = "Budget conscious but love spending"
                elif income_mean < 40 and spending_mean < 40:
                    segment_name = "Frugal Customers"
                    icon = "🛡️"
                    description = "Low income, careful spenders"
                elif age_mean < 30 and spending_mean > 60:
                    segment_name = "Trendy Youth"
                    icon = "🔥"
                    description = "Young and active spenders"
                elif age_mean > 50 and income_mean > 60:
                    segment_name = "Established Professionals"
                    icon = "💼"
                    description = "Mature, stable customers"
                elif age_mean > 50 and spending_mean < 40:
                    segment_name = "Retirement Savers"
                    icon = "🏠"
                    description = "Senior, cautious spenders"
                else:
                    segment_name = "Regular Customers"
                    icon = "⭐"
                    description = "Balanced characteristics"
                
                segment_profiles.append({
                    'Icon': icon,
                    'Segment Name': segment_name,
                    'Description': description,
                    'Size': size,
                    'Size %': f"{size/len(st.session_state.df_clean)*100:.1f}%",
                    'Avg Age': f"{age_mean:.0f} years",
                    'Avg Income': f"${income_mean:.0f}K",
                    'Avg Spending': f"{spending_mean:.0f}/100"
                })
            
            profile_df = pd.DataFrame(segment_profiles)
            st.dataframe(profile_df, use_container_width=True)
            
            # Visualize segment sizes
            st.subheader(" Segment Size Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sizes = [p['Size'] for p in segment_profiles]
            labels = [p['Segment Name'] for p in segment_profiles]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#D4A5A5', '#9B59B6', '#3498DB']
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(sizes)])
            ax2.set_title('Customer Segment Distribution')
            st.pyplot(fig2)
    
    # ============================================
    # DBSCAN CLUSTERING PAGE
    # ============================================
    elif page == "DBSCAN Clustering":
        st.header(" DBSCAN Clustering")
        
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
                        st.success(f" Done! Found {n_clusters} clusters, Silhouette: {sil_score:.4f}")
                    else:
                        st.warning(f"Found only {n_clusters} clusters. Try adjusting eps or min_samples")
        
        with col2:
            if 'dbscan_labels' in st.session_state:
                m1, m2, m3 = st.columns(3)
                m1.metric("Clusters Found", st.session_state.dbscan_clusters)
                m2.metric("Noise Points", st.session_state.dbscan_noise)
                if 'dbscan_sil' in st.session_state:
                    m3.metric("Silhouette Score", f"{st.session_state.dbscan_sil:.4f}")
        
        if 'dbscan_labels' in st.session_state:
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
            
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"DBSCAN Clustering (eps={eps_value}, min_samples={min_samples_value})")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            st.info(" Note: Black dots are 'Noise' - customers that don't fit well into any cluster")
    
    # ============================================
    # METHOD COMPARISON PAGE
    # ============================================
    elif page == "Method Comparison":
        st.header(" Algorithm Comparison")
        
        if 'kmeans_labels' in st.session_state and 'dbscan_labels' in st.session_state:
            # Comparison table
            comparison_data = {
                'Algorithm': ['K-Means', 'DBSCAN'],
                'Silhouette Score': [f"{st.session_state.kmeans_sil:.4f}", f"{st.session_state.dbscan_sil:.4f}" if 'dbscan_sil' in st.session_state else 'N/A'],
                'Number of Segments': [st.session_state.kmeans_k, st.session_state.dbscan_clusters],
                'Noise Points': ['None (0)', f"{st.session_state.dbscan_noise} customers"],
                'Best For': ['Spherical clusters, balanced sizes', 'Irregular shapes, outlier detection'],
                'Pros': ['Easy to interpret, stable results', 'No need to specify k, finds noise'],
                'Cons': ['Requires choosing k, sensitive to outliers', 'Parameter sensitive, can find too many noise']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Recommendation
            st.subheader(" Recommendation")
            
            if st.session_state.kmeans_sil > 0.5 and st.session_state.dbscan_sil > 0.5:
                st.success(" Both algorithms perform well! Consider:")
                st.write("- Use **K-Means** if you want equal-sized segments")
                st.write("- Use **DBSCAN** if you want to identify unique/outlier customers")
            elif st.session_state.kmeans_sil > st.session_state.dbscan_sil:
                st.success(" **K-Means** performs better for your data")
                st.write("- Creates clearly separated segments")
                st.write("- Good for standard customer grouping")
            else:
                st.success(" **DBSCAN** performs better for your data")
                st.write("- Better at finding natural groupings")
                st.write("- Can identify unusual customer patterns")
            
            # Visual comparison
            st.subheader(" Visual Comparison")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # K-Means
            scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                       c=st.session_state.kmeans_labels, 
                                       cmap='viridis', s=50, alpha=0.6)
            axes[0].set_title(f'K-Means - {st.session_state.kmeans_k} Segments', fontsize=12)
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            plt.colorbar(scatter1, ax=axes[0])
            
            # DBSCAN
            unique_labels = set(st.session_state.dbscan_labels)
            for k in unique_labels:
                mask = st.session_state.dbscan_labels == k
                if k == -1:
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label='Noise', alpha=0.5)
                else:
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label=f'Cluster {k}')
            axes[1].set_title(f'DBSCAN - {st.session_state.dbscan_clusters} Segments', fontsize=12)
            axes[1].set_xlabel('PC1')
            axes[1].set_ylabel('PC2')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning(" Please run both K-Means and DBSCAN first (go to their tabs and click Run)")
            st.info("1. Go to 'K-Means Clustering' tab and click 'Run K-Means'")
            st.info("2. Go to 'DBSCAN Clustering' tab and click 'Run DBSCAN'")
            st.info("3. Then come back here to see comparison")
    
    # ============================================
    # CUSTOMER PROFILING PAGE
    # ============================================
    elif page == "Customer Profiling":
        st.header(" Customer Segment Profiles")
        
        if 'kmeans_labels' in st.session_state:
            st.subheader(" Understanding Your Customer Segments")
            
            # Create detailed profiles with meaningful names
            segment_details = []
            
            for cluster in range(st.session_state.kmeans_k):
                cluster_data = st.session_state.df_clean[st.session_state.df_clean['Cluster'] == cluster]
                
                size = len(cluster_data)
                percentage = (size / len(st.session_state.df_clean)) * 100
                age_mean = cluster_data['Age'].mean()
                income_mean = cluster_data['Annual_Income'].mean()
                spending_mean = cluster_data['Spending_Score'].mean()
                
                # Determine gender ratio
                if 'Gender' in cluster_data.columns:
                    female_pct = (cluster_data['Gender'] == 'Female').sum() / size * 100
                    male_pct = 100 - female_pct
                else:
                    female_pct, male_pct = 50, 50
                
                # Create meaningful segment name and description
                if income_mean > 70 and spending_mean > 60:
                    segment_name = "👑 VIP Premium Customers"
                    marketing = "Offer exclusive products, early access, VIP events"
                    products = "Luxury items, premium services, bundles"
                    channel = "Email, personal calls, exclusive app access"
                elif income_mean > 70 and spending_mean < 40:
                    segment_name = "💰 Smart Value Shoppers"
                    marketing = "Highlight discounts, cashback, value deals"
                    products = "Mid-range products, bulk discounts"
                    channel = "Email newsletters, coupon apps"
                elif income_mean < 40 and spending_mean > 60:
                    segment_name = "🎯 Aspiring Trendsetters"
                    marketing = "Social media campaigns, influencer marketing"
                    products = "Trendy items, affordable luxuries"
                    channel = "Instagram, TikTok, mobile apps"
                elif income_mean < 40 and spending_mean < 40:
                    segment_name = "🛡️ Practical Frugal"
                    marketing = "Focus on essential needs, basic value"
                    products = "Essential items, budget-friendly options"
                    channel = "SMS, basic email, store flyers"
                elif age_mean < 30 and spending_mean > 60:
                    segment_name = "🔥 Young Trend Hunters"
                    marketing = "Flash sales, limited editions, viral marketing"
                    products = "New arrivals, seasonal items, accessories"
                    channel = "Social media, influencers, apps"
                elif age_mean > 50 and income_mean > 60:
                    segment_name = "💼 Established Affluents"
                    marketing = "Quality focus, reliability, service excellence"
                    products = "Premium brands, long-lasting goods"
                    channel = "Email, phone, physical stores"
                elif age_mean > 50 and spending_mean < 40:
                    segment_name = "🏠 Comfort Keepers"
                    marketing = "Trust, familiarity, loyalty rewards"
                    products = "Household essentials, familiar brands"
                    channel = "Traditional media, direct mail, phone"
                else:
                    segment_name = "⭐ Balanced Regulars"
                    marketing = "Mix of offers, loyalty program, referrals"
                    products = "Variety of products, bundles"
                    channel = "Multi-channel approach"
                
                segment_details.append({
                    'Segment': segment_name,
                    'Size': f"{size} customers ({percentage:.1f}%)",
                    'Age': f"{age_mean:.0f} years",
                    'Income': f"${income_mean:.0f}K",
                    'Spending': f"{spending_mean:.0f}/100",
                    'Gender': f"{female_pct:.0f}% Female / {male_pct:.0f}% Male",
                    'Marketing Focus': marketing,
                    'Product Focus': products,
                    'Channel Focus': channel
                })
            
            # Display profiles
            for seg in segment_details:
                with st.expander(f"{seg['Segment']} - {seg['Size']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Customer Characteristics:**")
                        st.write(f"- Age: {seg['Age']}")
                        st.write(f"- Income: {seg['Income']}")
                        st.write(f"- Spending Score: {seg['Spending']}")
                        st.write(f"- Gender: {seg['Gender']}")
                    
                    with col2:
                        st.write("**Marketing Strategy:**")
                        st.write(f"- Focus: {seg['Marketing Focus']}")
                        st.write(f"- Products: {seg['Product Focus']}")
                        st.write(f"- Channels: {seg['Channel Focus']}")
            
            # Summary statistics
            st.subheader(" Quick Summary")
            
            # Find key insights
            all_segments = segment_details
            largest = max(all_segments, key=lambda x: int(x['Size'].split()[0]))
            highest_spending = max(all_segments, key=lambda x: int(x['Spending'].split('/')[0]))
            highest_income = max(all_segments, key=lambda x: int(x['Income'].replace('$', '').replace('K', '')))
            
            col1, col2, col3 = st.columns(3)
            col1.info(f"**Largest Segment**\n\n{largest['Segment']}\n{largest['Size']}")
            col2.success(f"**Highest Spenders**\n\n{highest_spending['Segment']}\nSpending: {highest_spending['Spending']}")
            col3.warning(f"**Highest Income**\n\n{highest_income['Segment']}\nIncome: {highest_income['Income']}")
            
            # Download
            st.subheader(" Export Data")
            csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
            st.download_button("Download Customer Segmentation Results", csv, "segmentation_results.csv")
            
        else:
            st.warning(" Please run K-Means clustering first")
            st.info("Go to 'K-Means Clustering' page and click 'Run K-Means'")

else:
    st.info(" Please upload a CSV file to begin")
    
    st.subheader(" Expected CSV Format")
    st.write("Your CSV should have columns like:")
    example = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'Age': [25, 35, 42, 28, 50],
        'Annual Income (k$)': [50, 75, 100, 60, 80],
        'Spending Score (1-100)': [60, 75, 85, 70, 65]
    })
    st.dataframe(example)
    
    st.subheader(" How to Use")
    st.markdown("""
    1. **Upload** your customer CSV file
    2. **Select** analysis type from dropdown menu
    3. **Run** clustering algorithm
    4. **View** customer segments with meaningful names
    5. **Download** results for your marketing team
    """)
''')

print(" Streamlit app created with dropdown menu and meaningful segment names!")
