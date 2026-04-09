# COMPLETE WORKING VERSION - With Smart Recommendations
import os

if os.path.exists('streamlit_app.py'):
    os.remove('streamlit_app.py')

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

# Initialize session state
if 'kmeans_result' not in st.session_state:
    st.session_state.kmeans_result = None
if 'dbscan_result' not in st.session_state:
    st.session_state.dbscan_result = None
if 'kmeans_k_value' not in st.session_state:
    st.session_state.kmeans_k_value = 5
if 'dbscan_eps_value' not in st.session_state:
    st.session_state.dbscan_eps_value = 0.8
if 'dbscan_min_samples_value' not in st.session_state:
    st.session_state.dbscan_min_samples_value = 5

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
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
    
    # ========== K-MEANS CLUSTERING ==========
    if page == "K-Means Clustering":
        st.header("K-Means Clustering")
        
        # Slider that remembers value
        k_value = st.slider("Number of Clusters (k)", 2, 10, st.session_state.kmeans_k_value, key="kmeans_slider")
        st.session_state.kmeans_k_value = k_value
        
        # Elbow Method Button - uses slider value as max range
        st.subheader("Find Optimal Number of Clusters")
        if st.button(f"Find Optimal k (Elbow Method) - Testing k=2 to {k_value}"):
            with st.spinner(f"Calculating Elbow Method for k=2 to {k_value}..."):
                inertias = []
                K_range = range(2, k_value + 1)
                for k in K_range:
                    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_test.fit(X_scaled)
                    inertias.append(kmeans_test.inertia_)
                
                fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
                ax_elbow.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
                ax_elbow.set_xlabel('Number of Clusters (k)', fontsize=12)
                ax_elbow.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
                ax_elbow.set_title(f'Elbow Method for Optimal k (k=2 to {k_value})', fontsize=14, fontweight='bold')
                
                # Find the elbow point (where decrease slows down)
                if len(inertias) >= 2:
                    # Calculate differences to find elbow
                    diffs = np.diff(inertias)
                    if len(diffs) >= 2:
                        elbow_point = K_range[np.argmin(diffs) + 1]
                        ax_elbow.axvline(x=elbow_point, color='red', linestyle='--', linewidth=2, label=f'Elbow at k={elbow_point}')
                        ax_elbow.legend()
                        st.info(f"📊 The 'elbow' suggests k={elbow_point} is optimal for this dataset.")
                    else:
                        ax_elbow.legend()
                
                plt.tight_layout()
                st.pyplot(fig_elbow)
        
        st.markdown("---")
        
        # Run button - results only show after clicking this
        if st.button("Run K-Means", type="primary"):
            with st.spinner("Running K-Means..."):
                kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                df_clean['Cluster'] = labels
                sil_score = silhouette_score(X_scaled, labels)
                cal_score = calinski_harabasz_score(X_scaled, labels)
                dav_score = davies_bouldin_score(X_scaled, labels)
                
                st.session_state.kmeans_result = {
                    'labels': labels,
                    'k': k_value,
                    'silhouette': sil_score,
                    'calinski': cal_score,
                    'davies': dav_score,
                    'df': df_clean.copy(),
                    'model': kmeans
                }
                st.rerun()
        
        # Results only show if kmeans_result exists
        if st.session_state.kmeans_result is not None:
            res = st.session_state.kmeans_result
            kmeans_model = res.get('model')
            
            st.success(f"K-Means completed successfully with k={res['k']}!")
            
            # ========== SMART RECOMMENDATION ==========
            st.subheader("💡 Recommendation")
            
            # Suggest whether to use Elbow Method or PCA based on silhouette score
            if res['silhouette'] > 0.5:
                st.success(f"✅ Great! Your chosen k={res['k']} gives a good silhouette score of {res['silhouette']:.4f}")
                st.info("📊 **Recommendation:** Use PCA Visualization to understand the cluster separation.")
                st.write("PCA will help you see how well-separated your clusters are in 2D space.")
            elif res['silhouette'] > 0.3:
                st.warning(f"⚠️ Your chosen k={res['k']} gives an average silhouette score of {res['silhouette']:.4f}")
                st.info("🔍 **Recommendation:** Try the Elbow Method to find a better k value, then use PCA to visualize.")
                st.write("A silhouette score above 0.5 indicates well-separated clusters.")
            else:
                st.error(f"❌ Your chosen k={res['k']} gives a poor silhouette score of {res['silhouette']:.4f}")
                st.info("🎯 **Recommendation:** Use the Elbow Method above to find the optimal k value.")
                st.write("Your current k value may not be optimal for this dataset.")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{res['silhouette']:.4f}", help="Higher is better (0.5+ is good)")
            col2.metric("Calinski-Harabasz", f"{res['calinski']:.0f}", help="Higher is better")
            col3.metric("Davies-Bouldin", f"{res['davies']:.4f}", help="Lower is better")
            
            st.subheader("Clustering Results - PCA Visualization")
            
            # PCA transformation
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot clusters
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=res['labels'], cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Plot cluster centers (RED X marks)
            if kmeans_model is not None:
                centers_pca = pca.transform(kmeans_model.cluster_centers_)
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=400, c='red', marker='X', 
                          edgecolors='black', linewidth=3, label='Cluster Centers', zorder=5)
                
                # Add labels for each center
                for i, center in enumerate(centers_pca):
                    ax.annotate(f'Center {i}', (center[0], center[1]), 
                               fontsize=12, fontweight='bold', ha='center', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9, edgecolor='black'))
            
            plt.colorbar(scatter, label='Cluster')
            ax.set_xlabel(f'Wealth & Spending Score (PC1 - {pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'Age & Spending Pattern (PC2 - {pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax.set_title(f'K-Means Clustering Results (k={res["k"]})', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Customer Lifetime Value (CLV) Calculation
            st.subheader("💰 Customer Lifetime Value by Segment")
            df_with_clv = res['df'].copy()
            df_with_clv['Estimated_CLV'] = df_with_clv['Annual_Income'] * (df_with_clv['Spending_Score'] / 100) * 0.1
            clv_by_segment = df_with_clv.groupby('Cluster')['Estimated_CLV'].mean()
            st.bar_chart(clv_by_segment)
            
            # Display CLV table
            clv_data = []
            for cluster in range(res['k']):
                cluster_data = df_with_clv[df_with_clv['Cluster'] == cluster]
                clv_data.append({
                    'Cluster': cluster,
                    'Avg CLV': f"${cluster_data['Estimated_CLV'].mean():,.2f}",
                    'Total CLV': f"${cluster_data['Estimated_CLV'].sum():,.2f}",
                    'Customers': len(cluster_data)
                })
            st.dataframe(pd.DataFrame(clv_data), use_container_width=True)
            
            # Add explanation
            with st.expander("📖 How to interpret these metrics"):
                st.markdown("""
                **Silhouette Score (0.3-0.5 = Good, 0.5+ = Excellent)**
                - Measures how similar a customer is to their own cluster vs other clusters
                - Higher score = better separated clusters
                
                **Customer Lifetime Value (CLV)**
                - Formula: Annual Income × (Spending Score / 100) × 0.1
                - Estimates long-term value of each customer
                - Focus retention efforts on high CLV segments
                
                **PCA Visualization**
                - X-axis (PC1): Wealth & Spending Score
                - Y-axis (PC2): Age & Spending Pattern
                - Red X marks: Cluster centers
                """)
            
            st.subheader("Cluster Sizes")
            cluster_sizes = pd.Series(res['labels']).value_counts().sort_index()
            st.bar_chart(cluster_sizes)
            
            st.subheader("Segment Summary - What Each Cluster Means")
            summary_data = []
            for cluster in range(res['k']):
                cluster_data = res['df'][res['df']['Cluster'] == cluster]
                size = len(cluster_data)
                pct = (size / len(res['df'])) * 100
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
        
        eps_value = st.slider("Epsilon (eps)", 0.1, 2.0, st.session_state.dbscan_eps_value, 0.05, key="dbscan_eps_slider")
        min_samples_value = st.slider("Min Samples", 2, 20, st.session_state.dbscan_min_samples_value, key="dbscan_min_slider")
        st.session_state.dbscan_eps_value = eps_value
        st.session_state.dbscan_min_samples_value = min_samples_value
        
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
                
                st.session_state.dbscan_result = {
                    'labels': labels,
                    'clusters': n_clusters,
                    'noise': n_noise,
                    'silhouette': sil_score,
                    'eps': eps_value,
                    'min_samples': min_samples_value
                }
                st.rerun()
        
        if st.session_state.dbscan_result is not None:
            res = st.session_state.dbscan_result
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters Found", res['clusters'])
            col2.metric("Noise Points", res['noise'])
            if res['silhouette'] > 0:
                col3.metric("Silhouette Score", f"{res['silhouette']:.4f}")
            else:
                col3.metric("Silhouette Score", "N/A")
            
            st.subheader("Clustering Results")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(12, 8))
            unique_labels = set(res['labels'])
            for k in unique_labels:
                mask = res['labels'] == k
                if k == -1:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label=f'Noise ({res["noise"]})', alpha=0.5)
                else:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label=f'Cluster {k}')
            ax.set_xlabel(f'Wealth & Spending Score (PC1)', fontsize=12)
            ax.set_ylabel(f'Age & Spending Pattern (PC2)', fontsize=12)
            ax.set_title("DBSCAN Clustering Results")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Cluster Distribution")
            cluster_counts = pd.Series(res['labels']).value_counts()
            st.bar_chart(cluster_counts)
            
            st.subheader("DBSCAN Summary")
            if res['clusters'] >= 2:
                dbscan_summary = []
                unique_clusters = [c for c in set(res['labels']) if c != -1]
                for cluster in unique_clusters:
                    mask = res['labels'] == cluster
                    cluster_data = df_clean[mask]
                    size = len(cluster_data)
                    pct = (size / len(df_clean)) * 100
                    avg_age = cluster_data['Age'].mean()
                    avg_income = cluster_data['Annual_Income'].mean()
                    avg_spending = cluster_data['Spending_Score'].mean()
                    
                    if avg_income > 70 and avg_spending > 60:
                        name = "VIP Premium Customers"
                    elif avg_income > 70 and avg_spending < 40:
                        name = "Smart Value Shoppers"
                    elif avg_income < 40 and avg_spending > 60:
                        name = "Aspiring Trendsetters"
                    elif avg_income < 40 and avg_spending < 40:
                        name = "Practical Frugal"
                    elif avg_age < 30 and avg_spending > 60:
                        name = "Young Trend Hunters"
                    elif avg_age > 50 and avg_income > 60:
                        name = "Established Affluents"
                    elif avg_age > 50 and avg_spending < 40:
                        name = "Comfort Keepers"
                    else:
                        name = "Regular Customers"
                    
                    dbscan_summary.append({
                        "Cluster": cluster,
                        "Segment Name": name,
                        "Size": f"{size} ({pct:.1f}%)",
                        "Avg Age": f"{avg_age:.0f}",
                        "Avg Income": f"${avg_income:.0f}K",
                        "Avg Spending": f"{avg_spending:.0f}"
                    })
                st.dataframe(pd.DataFrame(dbscan_summary), use_container_width=True)
                
                if res['noise'] > 0:
                    noise_pct = res['noise']/len(df_clean)*100
                    st.warning(f"Noise Points: {res['noise']} customers ({noise_pct:.1f}%) could not be assigned to any cluster.")
            else:
                st.warning("Not enough clusters found. Try adjusting eps and min_samples.")
        else:
            st.info("👈 Click the 'Run DBSCAN' button to see results")
    
    # ========== METHOD COMPARISON ==========
    elif page == "Method Comparison":
        st.header("Algorithm Comparison")
        
        kmeans_done = st.session_state.kmeans_result is not None
        dbscan_done = st.session_state.dbscan_result is not None
        
        if kmeans_done and dbscan_done:
            kmeans_sil = st.session_state.kmeans_result.get('silhouette', 0)
            dbscan_sil = st.session_state.dbscan_result.get('silhouette', -1)
            kmeans_k = st.session_state.kmeans_result.get('k', 0)
            dbscan_clusters = st.session_state.dbscan_result.get('clusters', 0)
            dbscan_noise = st.session_state.dbscan_result.get('noise', 0)
            
            comp_data = pd.DataFrame({
                "Algorithm": ["K-Means", "DBSCAN"],
                "Silhouette Score": [f"{kmeans_sil:.4f}", f"{dbscan_sil:.4f}" if dbscan_sil > 0 else "N/A"],
                "Number of Segments": [kmeans_k, dbscan_clusters],
                "Noise Points": ["None (0)", f"{dbscan_noise} customers"]
            })
            st.dataframe(comp_data, use_container_width=True)
            
            st.subheader("Recommendation")
            if dbscan_sil > 0:
                if kmeans_sil > dbscan_sil:
                    st.success("K-Means performs better with a higher silhouette score.")
                elif dbscan_sil > kmeans_sil:
                    st.success("DBSCAN performs better with a higher silhouette score.")
                else:
                    st.info("Both algorithms perform similarly.")
            else:
                st.info("K-Means is recommended as DBSCAN found limited clusters.")
            
            st.subheader("Visual Comparison")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            kmeans_labels = st.session_state.kmeans_result.get('labels')
            if kmeans_labels is not None:
                axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.6)
                axes[0].set_title(f"K-Means ({kmeans_k} Segments)", fontsize=12)
                axes[0].set_xlabel("PC1")
                axes[0].set_ylabel("PC2")
            
            dbscan_labels = st.session_state.dbscan_result.get('labels')
            if dbscan_labels is not None:
                unique_labels = set(dbscan_labels)
                for k in unique_labels:
                    mask = dbscan_labels == k
                    if k == -1:
                        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', s=50, label='Noise', alpha=0.5)
                    else:
                        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, alpha=0.6, label=f'Cluster {k}')
                axes[1].set_title(f"DBSCAN ({dbscan_clusters} Segments, {dbscan_noise} Noise)", fontsize=12)
                axes[1].set_xlabel("PC1")
                axes[1].set_ylabel("PC2")
                axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please run both K-Means and DBSCAN first")
            if not kmeans_done:
                st.info("Go to K-Means Clustering page and click Run K-Means")
            if not dbscan_done:
                st.info("Go to DBSCAN Clustering page and click Run DBSCAN")
    
    # ========== CUSTOMER PROFILING ==========
    elif page == "Customer Profiling":
        st.header("Customer Segment Profiles")
        
        if st.session_state.kmeans_result is not None:
            df_profiles = st.session_state.kmeans_result['df']
            k = st.session_state.kmeans_result['k']
            
            st.subheader("Segment Overview")
            
            segment_details = []
            for cluster in range(k):
                cluster_data = df_profiles[df_profiles['Cluster'] == cluster]
                size = len(cluster_data)
                percentage = (size / len(df_profiles)) * 100
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
                    'Percentage': f"{seg['pct']:.1f}%",
                    'Avg Age': f"{seg['age']:.0f} yrs",
                    'Avg Income': f"${seg['income']:.0f}K",
                    'Avg Spending': f"{seg['spending']:.0f}/100",
                    'Gender': f"{seg['female_pct']:.0f}% F / {seg['male_pct']:.0f}% M"
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
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                incomes = [s['income'] for s in segment_details]
                ax2.bar(names, incomes, color='lightgreen', edgecolor='black')
                ax2.set_title('Average Income by Segment', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Segment', fontsize=10)
                ax2.set_ylabel('Income ($K)', fontsize=10)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig2)
            
            with col3:
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                spendings = [s['spending'] for s in segment_details]
                ax3.bar(names, spendings, color='salmon', edgecolor='black')
                ax3.set_title('Average Spending by Segment', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Segment', fontsize=10)
                ax3.set_ylabel('Spending Score', fontsize=10)
                plt.xticks(rotation=45, ha='right')
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
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.info("Largest: " + largest['name'] + " (" + str(largest['size']) + " customers)")
            c2.success("Biggest Spenders: " + highest_spending['name'] + " (Score: " + str(round(highest_spending['spending'], 0)) + ")")
            c3.warning("Highest Income: " + highest_income['name'] + " ($" + str(round(highest_income['income'], 0)) + "K)")
            c4.info("Youngest: " + youngest['name'] + " (" + str(round(youngest['age'], 0)) + " years)")
            c5.success("Oldest: " + oldest['name'] + " (" + str(round(oldest['age'], 0)) + " years)")
            
            st.subheader('Export Data')
            csv = df_profiles.to_csv(index=False).encode('utf-8')
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
