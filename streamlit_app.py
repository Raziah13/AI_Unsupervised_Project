# Create SIMPLE working streamlit_app.py - NO f-string errors
import os

# Delete old file
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# Sidebar navigation
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "Choose Analysis Type",
    ["K-Means Clustering", "DBSCAN Clustering", "Customer Profiles"]
)

if uploaded_file:
    # Load and preprocess data
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
    if page == "K-Means Clustering":
        st.header("K-Means Clustering")
        
        k_value = st.slider("Number of Clusters (k)", 2, 10, 5)
        
        if st.button("Run K-Means", type="primary"):
            with st.spinner("Running K-Means..."):
                kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                df_clean['Cluster'] = labels
                
                sil_score = silhouette_score(X_scaled, labels)
                
                st.success("Done! Silhouette Score: " + str(round(sil_score, 4)))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Number of Clusters", k_value)
                col2.metric("Silhouette Score", str(round(sil_score, 4)))
                col3.metric("Total Customers", len(df_clean))
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6)
                plt.colorbar(scatter)
                ax.set_xlabel("First Principal Component")
                ax.set_ylabel("Second Principal Component")
                ax.set_title("K-Means Clustering Results (" + str(k_value) + " Segments)")
                st.pyplot(fig)
                
                st.subheader("Segment Profiles")
                
                profile_data = []
                for cluster in range(k_value):
                    cluster_data = df_clean[df_clean['Cluster'] == cluster]
                    profile_data.append({
                        'Segment': "Segment " + str(cluster + 1),
                        'Size': len(cluster_data),
                        'Percentage': str(round(len(cluster_data)/len(df_clean)*100, 1)) + "%",
                        'Avg Age': str(round(cluster_data['Age'].mean(), 0)),
                        'Avg Income': "$" + str(round(cluster_data['Annual_Income'].mean(), 0)) + "K",
                        'Avg Spending': str(round(cluster_data['Spending_Score'].mean(), 0))
                    })
                
                st.dataframe(pd.DataFrame(profile_data), use_container_width=True)
                
                st.session_state.kmeans_df = df_clean.copy()
                st.session_state.kmeans_ran = True
                st.session_state.kmeans_k = k_value
        
        else:
            st.info("Click 'Run K-Means' button to start clustering")
    
    # ============================================
    # DBSCAN CLUSTERING
    # ============================================
    elif page == "DBSCAN Clustering":
        st.header("DBSCAN Clustering")
        
        col1, col2 = st.columns(2)
        with col1:
            eps_value = st.slider("Epsilon (eps)", 0.1, 2.0, 0.8, 0.05)
        with col2:
            min_samples_value = st.slider("Min Samples", 2, 20, 5)
        
        if st.button("Run DBSCAN", type="primary"):
            with st.spinner("Running DBSCAN..."):
                dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
                labels = dbscan.fit_predict(X_scaled)
                df_clean['DBSCAN_Cluster'] = labels
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters >= 2:
                    mask = labels != -1
                    sil_score = silhouette_score(X_scaled[mask], labels[mask])
                    st.success("Done! Found " + str(n_clusters) + " clusters")
                    st.info("Silhouette Score (excluding noise): " + str(round(sil_score, 4)))
                else:
                    st.warning("Found only " + str(n_clusters) + " clusters. Try adjusting parameters.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Clusters Found", n_clusters)
                col2.metric("Noise Points", n_noise)
                col3.metric("Noise %", str(round((n_noise/len(labels))*100, 1)) + "%")
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                unique_labels = set(labels)
                
                for k in unique_labels:
                    mask = labels == k
                    if k == -1:
                        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label='Noise (' + str(n_noise) + ')', alpha=0.5)
                    else:
                        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label='Cluster ' + str(k))
                
                ax.set_xlabel("First Principal Component")
                ax.set_ylabel("Second Principal Component")
                ax.set_title("DBSCAN Clustering Results")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
                
                st.session_state.dbscan_ran = True
        
        else:
            st.info("Click 'Run DBSCAN' button to start clustering")
    
    # ============================================
    # CUSTOMER PROFILES
    # ============================================
    elif page == "Customer Profiles":
        st.header("Customer Segment Profiles")
        
        if 'kmeans_df' in st.session_state and st.session_state.kmeans_df is not None:
            df_profiles = st.session_state.kmeans_df
            k = st.session_state.kmeans_k
            
            st.subheader("Segment Characteristics")
            
            for cluster in range(k):
                cluster_data = df_profiles[df_profiles['Cluster'] == cluster]
                size = len(cluster_data)
                percentage = size / len(df_profiles) * 100
                
                income = cluster_data['Annual_Income'].mean()
                spending = cluster_data['Spending_Score'].mean()
                age = cluster_data['Age'].mean()
                
                if income > 70 and spending > 60:
                    name = "VIP Premium Customers"
                    icon = "👑"
                elif income > 70 and spending < 40:
                    name = "Smart Value Shoppers"
                    icon = "💰"
                elif income < 40 and spending > 60:
                    name = "Aspiring Trendsetters"
                    icon = "🎯"
                elif income < 40 and spending < 40:
                    name = "Practical Frugal"
                    icon = "🛡️"
                elif age < 30 and spending > 60:
                    name = "Young Trend Hunters"
                    icon = "🔥"
                elif age > 50 and income > 60:
                    name = "Established Affluents"
                    icon = "💼"
                elif age > 50 and spending < 40:
                    name = "Comfort Keepers"
                    icon = "🏠"
                else:
                    name = "Regular Customers"
                    icon = "⭐"
                
                with st.expander(icon + " " + name + " - " + str(size) + " customers (" + str(round(percentage, 1)) + "%)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Characteristics:**")
                        st.write("- Average Age: " + str(round(cluster_data['Age'].mean(), 0)) + " years")
                        st.write("- Average Income: $" + str(round(cluster_data['Annual_Income'].mean(), 0)) + "K")
                        st.write("- Average Spending: " + str(round(cluster_data['Spending_Score'].mean(), 0)) + "/100")
                        
                        if 'Gender' in cluster_data.columns:
                            female_pct = (cluster_data['Gender'] == 'Female').sum() / size * 100
                            st.write("- Gender: " + str(round(female_pct, 0)) + "% Female, " + str(round(100-female_pct, 0)) + "% Male")
                    
                    with col2:
                        st.write("**Marketing Strategy:**")
                        if "VIP" in name:
                            st.write("- Offer exclusive products and early access")
                            st.write("- VIP loyalty program with premium benefits")
                        elif "Smart Value" in name:
                            st.write("- Focus on discounts, cashback, and bundle deals")
                            st.write("- Price-match guarantees")
                        elif "Aspiring" in name:
                            st.write("- Social media campaigns and influencer marketing")
                            st.write("- Flash sales and limited editions")
                        elif "Practical" in name:
                            st.write("- Essential items at competitive prices")
                            st.write("- Loyalty points for everyday purchases")
                        elif "Young" in name:
                            st.write("- Instagram, TikTok, and mobile app marketing")
                            st.write("- New arrivals and seasonal items")
                        elif "Established" in name:
                            st.write("- Quality focus and service excellence")
                            st.write("- Email newsletters and phone support")
                        elif "Comfort" in name:
                            st.write("- Trust and familiarity messaging")
                            st.write("- Traditional media and direct mail")
                        else:
                            st.write("- Balanced marketing approach")
                            st.write("- Mix of digital and traditional channels")
            
            st.subheader("Key Insights")
            
            largest = max(range(k), key=lambda x: len(df_profiles[df_profiles['Cluster'] == x]))
            largest_data = df_profiles[df_profiles['Cluster'] == largest]
            
            highest_spending = max(range(k), key=lambda x: df_profiles[df_profiles['Cluster'] == x]['Spending_Score'].mean())
            highest_spending_data = df_profiles[df_profiles['Cluster'] == highest_spending]
            
            col1, col2 = st.columns(2)
            col1.info("**Largest Segment**\n\nSegment " + str(largest+1) + "\n" + str(len(largest_data)) + " customers")
            col2.success("**Highest Spending Segment**\n\nSegment " + str(highest_spending+1) + "\nAvg Spending: " + str(round(highest_spending_data['Spending_Score'].mean(), 0)) + "/100")
            
            st.subheader("Export Data")
            csv = df_profiles.to_csv(index=False).encode('utf-8')
            st.download_button("Download Segmentation Results", csv, "segmentation_results.csv")
        
        else:
            st.warning("Please run K-Means clustering first")
            st.info("Go to 'K-Means Clustering' page and click 'Run K-Means'")

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
""")

print("Streamlit app created successfully!")
