# Create working streamlit_app.py
import os

# Delete old file if exists
if os.path.exists('streamlit_app.py'):
    os.remove('streamlit_app.py')

# Create new streamlit_app.py line by line
with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write('import streamlit as st\n')
    f.write('import pandas as pd\n')
    f.write('import numpy as np\n')
    f.write('import matplotlib.pyplot as plt\n')
    f.write('import seaborn as sns\n')
    f.write('from sklearn.preprocessing import StandardScaler, LabelEncoder\n')
    f.write('from sklearn.cluster import KMeans, DBSCAN\n')
    f.write('from sklearn.metrics import silhouette_score\n')
    f.write('from sklearn.decomposition import PCA\n\n')
    
    f.write('st.set_page_config(page_title="Customer Segmentation", layout="wide")\n')
    f.write('st.title("Customer Segmentation Dashboard")\n\n')
    
    f.write('# Sidebar\n')
    f.write('st.sidebar.header("Upload & Settings")\n')
    f.write('uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])\n\n')
    
    f.write('if uploaded_file:\n')
    f.write('    # Load data\n')
    f.write('    df = pd.read_csv(uploaded_file)\n')
    f.write('    st.subheader("Dataset Preview")\n')
    f.write('    st.dataframe(df.head())\n\n')
    
    f.write('    # Preprocessing\n')
    f.write('    df_clean = df.copy()\n')
    f.write('    if "Annual Income (k$)" in df_clean.columns:\n')
    f.write('        df_clean.rename(columns={"Annual Income (k$)": "Annual_Income"}, inplace=True)\n')
    f.write('    if "Spending Score (1-100)" in df_clean.columns:\n')
    f.write('        df_clean.rename(columns={"Spending Score (1-100)": "Spending_Score"}, inplace=True)\n\n')
    
    f.write('    # Encode Gender\n')
    f.write('    if "Gender" in df_clean.columns:\n')
    f.write('        le = LabelEncoder()\n')
    f.write('        df_clean["Gender_Encoded"] = le.fit_transform(df_clean["Gender"])\n')
    f.write('    else:\n')
    f.write('        df_clean["Gender_Encoded"] = 0\n\n')
    
    f.write('    # Features\n')
    f.write('    features = ["Age", "Annual_Income", "Spending_Score", "Gender_Encoded"]\n')
    f.write('    X = df_clean[features].copy()\n\n')
    
    f.write('    # Scale\n')
    f.write('    scaler = StandardScaler()\n')
    f.write('    X_scaled = scaler.fit_transform(X)\n\n')
    
    f.write('    # Algorithm selection\n')
    f.write('    algorithm = st.sidebar.selectbox("Choose Algorithm", ["K-Means", "DBSCAN"])\n\n')
    
    f.write('    if algorithm == "K-Means":\n')
    f.write('        k_value = st.sidebar.slider("Number of Clusters (k)", 2, 10, 5)\n')
    f.write('        if st.sidebar.button("Run K-Means"):\n')
    f.write('            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)\n')
    f.write('            labels = kmeans.fit_predict(X_scaled)\n')
    f.write('            df_clean["Cluster"] = labels\n')
    f.write('            sil_score = silhouette_score(X_scaled, labels)\n')
    f.write('            st.success(f"Silhouette Score: {sil_score:.4f}")\n\n')
    
    f.write('    else:\n')
    f.write('        eps_value = st.sidebar.slider("Eps", 0.1, 2.0, 0.8, 0.05)\n')
    f.write('        min_samples_value = st.sidebar.slider("Min Samples", 2, 20, 5)\n')
    f.write('        if st.sidebar.button("Run DBSCAN"):\n')
    f.write('            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)\n')
    f.write('            labels = dbscan.fit_predict(X_scaled)\n')
    f.write('            df_clean["Cluster"] = labels\n')
    f.write('            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)\n')
    f.write('            st.success(f"Clusters found: {n_clusters}")\n')
    f.write('            if n_clusters >= 2:\n')
    f.write('                mask = labels != -1\n')
    f.write('                sil_score = silhouette_score(X_scaled[mask], labels[mask])\n')
    f.write('                st.info(f"Silhouette Score: {sil_score:.4f}")\n\n')
    
    f.write('    # Visualization\n')
    f.write('    if "Cluster" in df_clean.columns:\n')
    f.write('        pca = PCA(n_components=2)\n')
    f.write('        X_pca = pca.fit_transform(X_scaled)\n')
    f.write('        df_clean["PCA1"] = X_pca[:, 0]\n')
    f.write('        df_clean["PCA2"] = X_pca[:, 1]\n\n')
    
    f.write('        fig, ax = plt.subplots(figsize=(10, 6))\n')
    f.write('        sns.scatterplot(data=df_clean, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", s=100, ax=ax)\n')
    f.write('        ax.set_title(f"{algorithm} Clustering Results")\n')
    f.write('        st.pyplot(fig)\n\n')
    
    f.write('        # Cluster Statistics\n')
    f.write('        st.subheader("Cluster Statistics")\n')
    f.write('        stats = df_clean.groupby("Cluster").agg({\n')
    f.write('            "Age": "mean",\n')
    f.write('            "Annual_Income": "mean",\n')
    f.write('            "Spending_Score": "mean"\n')
    f.write('        }).round(1)\n')
    f.write('        stats["Size"] = df_clean.groupby("Cluster").size()\n')
    f.write('        st.dataframe(stats)\n\n')
    
    f.write('        # Download\n')
    f.write('        csv = df_clean.to_csv(index=False).encode("utf-8")\n')
    f.write('        st.download_button("Download Results", csv, "segmentation_results.csv")\n\n')
    
    f.write('else:\n')
    f.write('    st.info("Please upload a CSV file to begin")\n\n')
    f.write('    st.subheader("Expected CSV Format")\n')
    f.write('    sample = pd.DataFrame({\n')
    f.write('        "CustomerID": [1, 2, 3],\n')
    f.write('        "Gender": ["Male", "Female", "Female"],\n')
    f.write('        "Age": [25, 35, 42],\n')
    f.write('        "Annual Income (k$)": [50, 75, 100],\n')
    f.write('        "Spending Score (1-100)": [60, 75, 85]\n')
    f.write('    })\n')
    f.write('    st.dataframe(sample)\n')

print(" Streamlit app created successfully!")
