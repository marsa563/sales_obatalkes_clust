import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("K-Means Clustering Dashboard")

# Fungsi membaca CSV dari dalam ZIP
@st.cache_data
def read_csv_from_zip(zip_file):
    with zipfile.ZipFile(zip_file) as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV file found inside the ZIP.")
        with z.open(csv_files[0]) as csvfile:
            return pd.read_csv(csvfile)

# Upload file ZIP
st.sidebar.header("Upload your dataset (ZIP with CSV)")
uploaded_file = st.sidebar.file_uploader("Upload ZIP file", type=["zip"])

if uploaded_file is not None:
    try:
        df = read_csv_from_zip(uploaded_file)
        st.success("CSV file loaded successfully from ZIP!")
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Kolom yang diperlukan
        selected_columns = ["Item", "Qty", "Purchase Price", "Sales Price", "Item Amount", "Category"]

        # Drop baris yang memiliki NaN di kolom penting
        df = df.dropna(subset=selected_columns).reset_index(drop=True)

        # Grouping by Item
        df_grouped = df.groupby("Item").agg({
            "Qty": "sum",
            "Purchase Price": "mean",
            "Sales Price": "mean",
            "Item Amount": "sum"
        }).reset_index()
        df_grouped = df_grouped.merge(df[["Item", "Category"]].drop_duplicates(), on="Item", how="left")

        # Filter kategori Non Medis
        df_grouped = df_grouped[df_grouped['Category'] != 'Non Medis'].reset_index(drop=True)

        # Encoding Category
        le = LabelEncoder()
        df_grouped['Category'] = le.fit_transform(df_grouped['Category'])

        # Scaling data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_grouped[['Qty', 'Purchase Price', 'Sales Price', 'Item Amount', 'Category']])

        # Elbow Method
        st.subheader("Elbow Method to Determine Optimal K")
        sse = []
        k_values = list(range(2, 11))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            sse.append(kmeans.inertia_)

        # Plot Elbow
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(k_values, sse, marker='o', color='blue')
        ax.set_title("Elbow Method for Optimal K")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("SSE")
        st.pyplot(fig)

        # Pilih jumlah cluster
        st.sidebar.subheader("Select K for Clustering")
        k = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        st.write(f"Using K = {k} for clustering.")

        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        df_grouped["Cluster"] = kmeans.labels_

        # Tampilkan hasil clustering
        st.subheader("Clustering Result")
        st.dataframe(df_grouped[["Item", "Cluster"]])

        # Distribusi item per cluster
        st.subheader("Distribution of Items per Cluster")
        cluster_counts = df_grouped["Cluster"].value_counts().sort_index()
        total_items = cluster_counts.sum()

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_items) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_title("Number of Items per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Item Count")
        st.pyplot(fig)

        # PCA plot
        st.subheader("Cluster Visualization using PCA")
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=df_grouped["Cluster"], cmap='viridis')
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"PCA Cluster Visualization (K={k})")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)

        # Boxplot tiap fitur
        st.subheader("Feature Distributions per Cluster")
        features = ['Qty', 'Purchase Price', 'Sales Price', 'Item Amount']
        for feature in features:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=feature, data=df_grouped, palette="husl", ax=ax)
            ax.set_title(f"Boxplot of {feature} per Cluster")
            st.pyplot(fig)

        # Statistik tiap cluster
        st.subheader("Cluster Statistics Summary")
        for cluster_num in sorted(df_grouped["Cluster"].unique()):
            st.write(f"Statistics for Cluster {cluster_num}:")
            st.dataframe(df_grouped[df_grouped["Cluster"] == cluster_num].describe())

        # Evaluasi cluster
        st.subheader("Cluster Evaluation Metrics")
        dbi = davies_bouldin_score(X_scaled, kmeans.labels_)
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        st.write(f"Davies-Bouldin Index (lower is better): **{dbi:.2f}**")
        st.write(f"Silhouette Score (higher is better): **{silhouette_avg:.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan upload file ZIP yang berisi file CSV.")
