import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

st.title("K-Means Clustering Dashboard")

# Upload Dataset
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or ZIP file", type=["csv", "zip"])

# Load CSV or ZIP
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                file_names = z.namelist()
                csv_files = [f for f in file_names if f.endswith('.csv')]
                if len(csv_files) == 0:
                    st.error("❌ ZIP tidak mengandung file CSV.")
                else:
                    with z.open(csv_files[0]) as f:
                        df = pd.read_csv(f)
        else:
            df = pd.read_csv(uploaded_file)

        st.success("✅ Dataset loaded successfully!")
        st.subheader("Preview of Dataset")
        st.dataframe(df.head())

        # Selected columns
        selected_columns = ["Item", "Qty", "Purchase Price", "Sales Price", "Item Amount", "Category"]
        df = df.dropna(subset=selected_columns).reset_index(drop=True)

        # Convert numeric columns
        num_cols = ["Qty", "Purchase Price", "Sales Price", "Item Amount"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=num_cols).reset_index(drop=True)

        # Group by item
        df_grouped = df.groupby("Item").agg({
            "Qty": "sum",
            "Purchase Price": "mean",
            "Sales Price": "mean",
            "Item Amount": "sum"
        }).reset_index()

        # Merge category
        df_grouped = df_grouped.merge(df[["Item", "Category"]].drop_duplicates(), on="Item", how="left")
        df_grouped = df_grouped[df_grouped["Category"].str.lower() != "non medis"].reset_index(drop=True)

        # Encode category
        le = LabelEncoder()
        df_grouped["Category"] = le.fit_transform(df_grouped["Category"].astype(str))

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_grouped[["Qty", "Purchase Price", "Sales Price", "Item Amount", "Category"]])

        # Elbow Method
        st.subheader("Elbow Method to Determine Optimal K")
        sse = []
        k_values = list(range(2, 11))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            sse.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_values, sse, marker='o', color='blue')
        ax.set_title("Elbow Method for Optimal K")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("SSE")
        st.pyplot(fig)

        # Select K
        st.sidebar.subheader("Select Number of Clusters")
        optimal_k = st.sidebar.slider("K", min_value=2, max_value=10, value=5)
        st.write(f"🧠 Using K = {optimal_k} for clustering")

        # KMeans
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_grouped["Cluster"] = kmeans.fit_predict(X_scaled)

        # Results
        st.subheader("Cluster Assignment")
        st.dataframe(df_grouped[["Item", "Cluster"]])

        # Items per Cluster
        st.subheader("Items per Cluster")
        cluster_counts = df_grouped["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots()
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        total = cluster_counts.sum()
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{(height/total)*100:.1f}%", ha='center')
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Items")
        st.pyplot(fig)

        # PCA Visualization
        st.subheader("Cluster Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df_grouped["Cluster"], cmap='viridis')
        plt.colorbar(scatter, ax=ax)
        ax.set_title("Clusters in 2D Space")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        st.pyplot(fig)

        # Boxplot per feature
        st.subheader("Feature Distributions per Cluster")
        features = ['Qty', 'Purchase Price', 'Sales Price', 'Item Amount']
        for feature in features:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=feature, data=df_grouped, palette="husl", ax=ax)
            ax.set_title(f"Boxplot of {feature} per Cluster")
            st.pyplot(fig)

        # Cluster stats
        st.subheader("Cluster Statistics")
        for i in range(optimal_k):
            st.write(f"📊 Cluster {i} Statistics")
            st.dataframe(df_grouped[df_grouped["Cluster"] == i].describe())

        # Evaluation
        st.subheader("Clustering Evaluation Metrics")
        silhouette = silhouette_score(X_scaled, df_grouped["Cluster"])
        db_index = davies_bouldin_score(X_scaled, df_grouped["Cluster"])
        st.write(f"Silhouette Score (higher is better): **{silhouette:.2f}**")
        st.write(f"Davies-Bouldin Index (lower is better): **{db_index:.2f}**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

else:
    st.info("📂 Silakan upload file CSV atau ZIP untuk mulai analisis.")
