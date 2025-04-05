import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Title of the dashboard
st.title("K-Means Clustering Dashboard")

# Upload Dataset
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    
    # Show basic information
    st.subheader("Dataset Info")
    st.write(df)  # Show first few rows for quick inspection

    # Seleksi otomatis kolom yang diinginkan (Item, Qty, Purchase Price, Sales Price, Item Amount, Category)
    selected_columns = ["Item", "Qty", "Purchase Price", "Sales Price", "Item Amount", "Category"]
    
    # Data preprocessing steps

    # Drop missing values in selected columns
    df = df.dropna(subset=selected_columns).reset_index(drop=True)
        
    # Grouping Data
    df_grouped = df.groupby("Item").agg({
        "Qty": "sum",
        "Purchase Price": "mean",
        "Sales Price": "mean",
        "Item Amount": "sum"
    }).reset_index()
    df_grouped = df_grouped.merge(df[["Item", "Category"]].drop_duplicates(), on="Item", how="left")
    
    # Filter Non Medis category
    df_grouped = df_grouped[(df_grouped['Category'] != 'Non Medis')].reset_index(drop=True)
    
    # Encoding Category
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_grouped['Category'] = le.fit_transform(df_grouped['Category'])
        
    # Scaling data before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_grouped[['Qty', 'Purchase Price', 'Sales Price', 'Item Amount', 'Category']])
        
    # Perform Elbow Method for optimal K
    st.subheader("Elbow Method to Determine Optimal K")
    sse = []
    k_values = list(range(2, 11))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
        
    # Plot Elbow Method
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_values, sse, marker='o', color='b')
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("SSE (Sum of Squared Errors)")
    st.pyplot(fig)

    # Sidebar for selecting the number of clusters
    st.sidebar.subheader("Select K for K-Means")
    k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)

    # Automatically select optimal K (the "elbow point")
    optimal_k = k
    st.write(f"Optimal number of clusters based on Elbow Method: {optimal_k}")
        
    # Apply KMeans clustering with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df_grouped["Cluster"] = kmeans.labels_

    # Show results
    st.subheader(f"Clustering Results with Optimal K = {optimal_k}")
    st.write(df_grouped[["Item", "Cluster"]])
        
    # Plot distribution of clusters
    st.subheader("Distribution of Items per Cluster")
    cluster_counts = df_grouped["Cluster"].value_counts()
    total_items = cluster_counts.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color='skyblue')

    # Tambahkan label persentase di atas setiap bar
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_items) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_title("Number of Items per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # PCA for visualization of clusters
    st.subheader("Cluster Visualization")
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=df_grouped["Cluster"], cmap='viridis')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Cluster Visualization with K={optimal_k}")
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)

    # Boxplots for each feature
    st.subheader("Feature Distributions per Cluster")
    features = ['Qty', 'Purchase Price', 'Sales Price', 'Item Amount']
    for feature in features:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=df_grouped, palette="husl", ax=ax)
        ax.set_title(f"Boxplot of {feature} per Cluster")
        st.pyplot(fig)
        
    # Show statistics per cluster
    st.subheader("Cluster Statistics")
    for cluster_num in range(optimal_k):
        st.write(f"Statistics for Cluster {cluster_num}:")
        st.write(df_grouped[df_grouped["Cluster"] == cluster_num].describe())
        
    # Evaluation metrics: DBI and Silhouette Score
    st.subheader("Cluster Evaluation")
    dbi = davies_bouldin_score(X_scaled, kmeans.labels_)
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    st.write(f"Davies-Bouldin Index (Lower is Better): {dbi:.2f}")
    st.write(f"Silhouette Score (Higher is Better): {silhouette_avg:.2f}")

else:
    st.write("Please upload a dataset to proceed.")