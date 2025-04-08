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

st.title("Analisis Segmentasi Penjualan Obat dan Alat Kesehatan Untuk Optimalisasi Pengadaan Stok")

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
        st.markdown("""
        Elbow Method digunakan untuk membantu menentukan jumlah cluster optimal. 
        
        Berdasarkan grafik diatas, jumlah cluster optimal adalah 5 cluster.
        """)

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
        st.markdown("""
        Cluster 1 mendominasi dengan persentase 75.63%, menunjukkan bahwa sebagian besar item memiliki karakteristik yang sama pada kluster ini. 
        
        Cluster 0 berada di urutan kedua dengan 23.41%, juga memiliki kontribusi yang signifikan. 
        
        Cluster 2, 3, dan 4 hanya mencakup persentase kecil, masing-masing 0.17%, 0.28%, dan 0.51%, menunjukkan bahwa hanya sedikit item yang termasuk dalam kluster-kluster ini.
        """)

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
        st.markdown("""
        Keterangan:
        
        Silhouette Index (SI) : semakin mendekati 1, maka semakin baik hasil klasterisasi. 
        
        Davies Bouldin Index (DBI) : semakin mendekati 0, maka semakin baik hasil klasterisasi.
        """)

        # Rekomendasi
        st.subheader("Rekomendasi")
        st.markdown("""
        Cluster 0 (Kurang Laris): Cluster ini memiliki rata-rata penjualan berada di tengah, yaitu 367 item dengan harga pembelian dan penjualan yang relatif rendah dibandingkan cluster lainnya. Cluster ini berisi produk dengan permintaan stabil tetapi tidak terlalu besar. Strategi yang dapat diambil yaitu mempertahankan jumlah stok yang cukup untuk menghindari kekurangan stok, tetapi tetap perlu memonitor permintaan secara berkala. Sebaiknya melakukan pengadaan dalam jumlah sedang.
        
        Cluster 1 (Cukup Laris): Cluster ini memiliki rata-rata penjualan yang lebih tinggi (638 item) dengan harga jual dan beli yang lebih rendah, menunjukkan produk ini terjual dengan cepat dan dalam jumlah besar. Produk-produk ini biasanya dibeli dalam jumlah besar tetapi dengan harga yang lebih terjangkau. Rumah Sakit dapat memprioritaskan dalam pengadaan stok agar tidak terjadi kekosongan barang, karena kuantitas penjualan yang tinggi menunjukkan permintaan yang besar. Stok dalam jumlah besar perlu dipersiapkan untuk memenuhi permintaan pasar yang cepat.
        
        Cluster 2 (Laris): Cluster ini memiliki rata-rata penjualan tertinggi kedua setelah cluster 4 (1.548 item) dan harga beli serta jual yang juga tinggi. Produk dalam cluster ini memiliki permintaan tinggi dan margin keuntungan yang besar. Pengadaan produk dalam jumlah penjualan besar harus diprioritaskan dan stok harus selalu tersedia karena produk ini merupakan penyumbang utama penjualan. Rumah Sakit dapat melakukan pengadaan dalam jumlah besar secara berkala, tetapi pastikan untuk memantau tren penjualan guna mencegah overstock.
        
        Cluster 3 (Sangat Kurang Laris): Cluster ini memiliki rata-rata penjualan sangat rendah (5,12 item), namun harga beli dan jual sangat tinggi. Cluster ini berisi produk eksklusif atau dengan harga sangat tinggi, tetapi permintaan sangat terbatas. Produk ini cenderung dijual dalam jumlah kecil, mungkin untuk kategori alkes yang sangat spesifik. Rumah Sakit dapat melakukan pengadaan stok dalam jumlah kecil saja, karena permintaan rendah dan produk ini bisa menyebabkan stok menumpuk jika dibeli dalam jumlah besar. Produk dalam cluster ini hanya dibeli sesuai permintaan, lakukan evaluasi berkala untuk menghindari penyimpanan berlebihan dan tetap sediakan stok minimum untuk kebutuhan khusus.
        
        Cluster 4 (Sangat Laris): Rata-rata penjualan terbesar (27.208 item), namun harga beli dan jual sangat rendah. Produk ini memiliki kuantitas penjualan yang sangat besar, meskipun margin keuntungan per unit kecil. Produk ini mungkin merupakan barang-barang kebutuhan dasar yang dijual dalam volume besar sehingga stok dalam jumlah sangat besar harus dipersiapkan karena perputaran produk sangat cepat. Produk ini harus selalu tersedia untuk memenuhi kebutuhan tinggi. Rumah Sakit perlu melakukan pengadaan dalam volume besar dengan negosiasi harga rendah dari pemasok sangat penting untuk mengoptimalkan keuntungan.
        """)

        # Saran
        st.subheader("Saran")
        st.markdown("""
        Hasil analisis dan visualisasi ini berdasarkan dataset transaksi bulan Januari-Desember 2024.
        Jika terdapat perubahan data, maka analisis dan visualisasi akan berubah juga.
        """)


    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

else:
    st.info("📂 Silakan upload file CSV atau ZIP untuk mulai analisis.")
