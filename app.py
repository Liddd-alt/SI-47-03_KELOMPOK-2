import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Dashboard Analisis Wisatawan Nusantara",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #fffbe6 !important;  
        color: #222 !important;               
        font-size: 1.3rem;
        font-weight: bold;
        border-left: 4px solid #f7b731;
        margin-bottom: 1rem;
    }
    .cluster-info {
           background-color: #e8f4fd;
           color: #111 !important;
           padding: 1rem;
           border-radius: 0.5rem;
           margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('dataset.csv')
        
        if 'Bulan13' not in df.columns:
            df['Bulan13'] = np.nan
            df.to_csv('dataset.csv', index=False)
            st.sidebar.success("✅ Kolom 'Bulan13' berhasil ditambahkan ke dataset!")
        
        return df
    except FileNotFoundError:
        st.error("File dataset.csv tidak ditemukan! Pastikan file berada di direktori yang sama dengan app.py")
        return None

@st.cache_resource
def train_clustering_model(df):
    """Train K-Means clustering model"""
    columns_to_drop = ['Provinsi']
    if 'Bulan13' in df.columns:
        columns_to_drop.append('Bulan13')
    
    features = df.drop(columns_to_drop, axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    return kmeans, scaler, cluster_labels

@st.cache_resource
def train_classification_model(df):
    """Train Logistic Regression model"""
    bins = [0, 10000000, 50000000, df['Tahunan'].max()]
    labels = ['Rendah', 'Sedang', 'Tinggi']
    df['Kategori_Kunjungan'] = pd.cut(df['Tahunan'], bins=bins, labels=labels, include_lowest=True)
    
    columns_to_drop = ['Provinsi', 'Tahunan', 'Kategori_Kunjungan']
    if 'Bulan13' in df.columns:
        columns_to_drop.append('Bulan13')
    
    X = df.drop(columns_to_drop, axis=1)
    y = df['Kategori_Kunjungan']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_train, y_train)
    
    return logreg_model, X_test, y_test

def predict_bulan13_model(df):
    if 'Bulan13' not in df.columns:
        return None, None
    
    if df['Bulan13'].notna().sum() > 0:
        X = df[['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']]
        y = df['Bulan13'].dropna()  
        X = X.loc[y.index]  
        
        if len(y) > 0:
            model = LinearRegression()
            model.fit(X, y)
            return model, X.columns.tolist(), 'actual'
    

    monthly_cols = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                   'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

    monthly_avg = df[monthly_cols].mean()
    np.random.seed(42) 
    synthetic_bulan13 = monthly_avg.mean() * (0.8 + 0.4 * np.random.rand(len(df)))
    
    X = df[monthly_cols]
    y = synthetic_bulan13
    
    model = LinearRegression()
    model.fit(X, y)
    return model, monthly_cols, 'dummy'

def main():

    st.markdown('<h1 class="main-header">🏝️ Dashboard Analisis Wisatawan Nusantara</h1>', unsafe_allow_html=True)
    st.markdown("### Segmentasi Provinsi & Prediksi Kategori Kunjungan")
    

    df = load_data()
    if df is None:
        return

    st.sidebar.title("📊 Menu Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["🏠 Dashboard Utama", "📈 Data Overview", "🎯 Clustering Analysis", "🔮 Prediction Model", "📅 Prediksi Bulan ke-13", "📊 Visualizations"]
    )
    
    if page == "🏠 Dashboard Utama":
        show_dashboard(df)
    elif page == "📈 Data Overview":
        show_data_overview(df)
    elif page == "🎯 Clustering Analysis":
        show_clustering_analysis(df)
    elif page == "🔮 Prediction Model":
        show_prediction_model(df)
    elif page == "📅 Prediksi Bulan ke-13":
        show_predict_bulan13(df)
    elif page == "📊 Visualizations":
        show_visualizations(df)

def show_dashboard(df):
    """Main dashboard page"""
    st.markdown("## 📊 Dashboard Utama")
    

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Provinsi", len(df))
    
    with col2:
        st.metric("Total Kunjungan Tahunan", f"{df['Tahunan'].sum():,}")
    
    with col3:
        st.metric("Rata-rata Kunjungan", f"{df['Tahunan'].mean():,.0f}")
    
    with col4:
        st.metric("Provinsi Tertinggi", df.loc[df['Tahunan'].idxmax(), 'Provinsi'])
    
    st.markdown("### 🏆 Top 10 Provinsi dengan Kunjungan Tertinggi")
    
    top_10 = df.nlargest(10, 'Tahunan')
    fig = px.bar(
        top_10, 
        x='Provinsi', 
        y='Tahunan',
        title="Top 10 Provinsi dengan Kunjungan Tertinggi",
        color='Tahunan',
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📅 Pola Kunjungan Bulanan")
    
    monthly_data = df[['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                      'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']].mean()
    
    fig = px.line(
        x=monthly_data.index,
        y=monthly_data.values,
        title="Rata-rata Kunjungan Bulanan",
        markers=True
    )
    fig.update_layout(xaxis_title="Bulan", yaxis_title="Rata-rata Kunjungan")
    st.plotly_chart(fig, use_container_width=True)

def show_data_overview(df):
    """Data overview page"""
    st.markdown("## 📈 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Informasi Dataset")
        st.write(f"**Jumlah Baris:** {len(df)}")
        st.write(f"**Jumlah Kolom:** {len(df.columns)}")
        st.write(f"**Ukuran Dataset:** {df.shape}")
        
        st.markdown("### 🔧 Tipe Data")
        st.dataframe(df.dtypes.to_frame('Tipe Data'))
    
    with col2:
        st.markdown("### 📊 Statistik Deskriptif")
        st.dataframe(df.describe())
    
    st.markdown("### ❓ Missing Values")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("✅ Tidak ada missing values dalam dataset!")
    else:
        st.warning("⚠️ Terdapat missing values:")
        st.dataframe(missing_data[missing_data > 0])
    
    st.markdown("### 📄 Data Mentah")
    st.dataframe(df)

def show_clustering_analysis(df):
    """Clustering analysis page"""
    st.markdown("## 🎯 Analisis Clustering (K-Means)")
    
    kmeans, scaler, cluster_labels = train_clustering_model(df)
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    st.markdown("### 🎨 Hasil Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Distribusi Cluster")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title="Distribusi Provinsi per Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Statistik per Cluster")
        cluster_stats = df_clustered.groupby('Cluster')['Tahunan'].agg(['mean', 'min', 'max', 'count'])
        cluster_stats.columns = ['Rata-rata', 'Minimum', 'Maksimum', 'Jumlah Provinsi']
        st.dataframe(cluster_stats)
    
    st.markdown("### 🔍 Detail Provinsi per Cluster")
    
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_provinces = df_clustered[df_clustered['Cluster'] == cluster_id]['Provinsi'].tolist()
        st.markdown(f"""
        <div class="cluster-info">
        <h4>Cluster {cluster_id}</h4>
        <p><strong>Jumlah Provinsi:</strong> {len(cluster_provinces)}</p>
        <p><strong>Provinsi:</strong> {', '.join(cluster_provinces)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Visualisasi Cluster")
    
    monthly_cols = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                   'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    
    cluster_monthly = df_clustered.groupby('Cluster')[monthly_cols].mean()
    
    fig = go.Figure()
    for cluster_id in cluster_monthly.index:
        fig.add_trace(go.Scatter(
            x=monthly_cols,
            y=cluster_monthly.loc[cluster_id],
            mode='lines+markers',
            name=f'Cluster {cluster_id}',
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title="Pola Kunjungan Bulanan per Cluster",
        xaxis_title="Bulan",
        yaxis_title="Rata-rata Kunjungan",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_model(df):
    """Prediction model page"""
    st.markdown("## 🔮 Model Prediksi Kategori Kunjungan")
    
    if 'Kategori_Kunjungan' not in df.columns:
        bins = [0, 10000000, 50000000, df['Tahunan'].max()]
        labels = ['Rendah', 'Sedang', 'Tinggi']
        df['Kategori_Kunjungan'] = pd.cut(df['Tahunan'], bins=bins, labels=labels, include_lowest=True)
    
    logreg_model, X_test, y_test = train_classification_model(df)
    
    st.markdown("### 📊 Performa Model")
    
    y_pred = logreg_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Akurasi Model", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Jumlah Data Test", len(X_test))
    
    with col3:
        st.metric("Jumlah Kategori", len(df['Kategori_Kunjungan'].unique()))
    

    st.markdown("### 📋 Laporan Klasifikasi")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(report_df)
    

    st.markdown("### 🎯 Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Rendah', 'Sedang', 'Tinggi'],
        y=['Rendah', 'Sedang', 'Tinggi']
    )
    st.plotly_chart(fig, use_container_width=True)
    

    st.markdown("### 🔮 Prediksi Kategori Kunjungan")
    st.markdown("Masukkan data bulanan untuk memprediksi kategori kunjungan:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        januari = st.number_input("Januari", min_value=0, value=1000000)
        februari = st.number_input("Februari", min_value=0, value=1000000)
        maret = st.number_input("Maret", min_value=0, value=1000000)
        april = st.number_input("April", min_value=0, value=1000000)
    
    with col2:
        mei = st.number_input("Mei", min_value=0, value=1000000)
        juni = st.number_input("Juni", min_value=0, value=1000000)
        juli = st.number_input("Juli", min_value=0, value=1000000)
        agustus = st.number_input("Agustus", min_value=0, value=1000000)
    
    with col3:
        september = st.number_input("September", min_value=0, value=1000000)
        oktober = st.number_input("Oktober", min_value=0, value=1000000)
        november = st.number_input("November", min_value=0, value=1000000)
        desember = st.number_input("Desember", min_value=0, value=1000000)
    
    if st.button("🔮 Prediksi Kategori"):

        input_data = np.array([januari, februari, maret, april, mei, juni, 
                              juli, agustus, september, oktober, november, desember]).reshape(1, -1)
        

        prediction = logreg_model.predict(input_data)[0]
        probability = logreg_model.predict_proba(input_data)[0]
        
        st.markdown("### 📊 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>Kategori Prediksi: {prediction}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Probabilitas per Kategori:**")
            categories = ['Rendah', 'Sedang', 'Tinggi']
            for cat, prob in zip(categories, probability):
                st.write(f"{cat}: {prob:.2%}")

def show_predict_bulan13(df):
    st.markdown("## 📅 Prediksi Bulan ke-13 (Regresi)")
    st.markdown("Masukkan data 12 bulan untuk memprediksi nilai bulan ke-13 menggunakan Linear Regression.")

    model, bulan_cols, data_type = predict_bulan13_model(df)
    if model is None:
        st.warning("Dataset tidak memiliki kolom 'Bulan13'. Contoh dummy akan digunakan.")
        bulan_cols = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                     'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
        model = LinearRegression()
        # Dummy fit
        X_dummy = np.arange(1, 13).reshape(1, -1)
        y_dummy = np.array([100])
        model.fit(X_dummy, y_dummy)
        data_type = 'dummy'


    if data_type == 'actual':
        st.success("✅ Model dilatih dengan data aktual dari kolom Bulan13")
    else:
        st.info("ℹ️ Model menggunakan data sintetis untuk demonstrasi (kolom Bulan13 kosong)")


    st.markdown("### Input Data Bulanan")
    cols = st.columns(4)
    input_vals = []
    for i, bulan in enumerate(bulan_cols):
        val = cols[i % 4].number_input(bulan, min_value=0, value=1000000)
        input_vals.append(val)

    if st.button("🔮 Prediksi Bulan ke-13"):
        input_array = np.array(input_vals).reshape(1, -1)
        pred = model.predict(input_array)[0]
        
        st.markdown(f"""
        <div class="metric-card">
        <h3>Prediksi Bulan ke-13: <span style='color:#d35400'>{pred:,.0f}</span></h3>
        </div>
        """, unsafe_allow_html=True)
        

        if data_type == 'actual':
            st.markdown("""
            <div style='margin-top:10px; font-size:1.1rem; color:#333;'>
            <b>✅ Model Aktual:</b> <br>
            Prediksi di atas adalah hasil regresi yang dilatih dengan data aktual dari kolom Bulan13.<br>
            Model mempelajari pola dari data historis untuk memprediksi bulan ke-13.<br>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='margin-top:10px; font-size:1.1rem; color:#333;'>
            <b>ℹ️ Model Demonstrasi:</b> <br>
            Prediksi di atas adalah hasil regresi menggunakan data sintetis.<br>
            Kolom Bulan13 saat ini kosong, sehingga model menggunakan pola data bulanan yang ada.<br>
            <b>Untuk hasil yang akurat:</b> Isi kolom Bulan13 dengan data aktual (misal: data Januari tahun berikutnya).<br>
            </div>
            """, unsafe_allow_html=True)
        

        st.markdown("### 📊 Statistik Input")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Input", f"{sum(input_vals):,}")
        with col2:
            st.metric("Rata-rata Bulanan", f"{np.mean(input_vals):,.0f}")
        with col3:
            st.metric("Bulan Tertinggi", f"{max(input_vals):,}")
        

        st.markdown("### 📈 Grafik Input Bulanan")
        input_df = pd.DataFrame({
            'Bulan': bulan_cols,
            'Jumlah': input_vals
        })
        
        fig = px.bar(
            input_df,
            x='Bulan',
            y='Jumlah',
            title="Data Input Bulanan",
            color='Jumlah',
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df):
    """Visualizations page"""
    st.markdown("## 📊 Visualisasi Data")
    
    viz_option = st.selectbox(
        "Pilih Visualisasi:",
        ["📈 Line Chart Tahunan", "🔥 Heatmap Bulanan", "📊 Box Plot per Bulan", "🎯 Scatter Plot"]
    )
    
    if viz_option == "📈 Line Chart Tahunan":
        st.markdown("### 📈 Jumlah Kunjungan Tahunan per Provinsi")
        
        fig = px.line(
            df, 
            x='Provinsi', 
            y='Tahunan',
            title="Jumlah Kunjungan Tahunan per Provinsi",
            markers=True
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "🔥 Heatmap Bulanan":
        st.markdown("### 🔥 Heatmap Pola Kunjungan Bulanan")
        
        monthly_cols = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                       'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
        
        heatmap_data = df.set_index('Provinsi')[monthly_cols]
        
        fig = px.imshow(
            heatmap_data,
            title="Heatmap Pola Kunjungan Bulanan per Provinsi",
            aspect="auto",
            color_continuous_scale="YlOrRd"
        )
        fig.update_layout(
            xaxis_title="Bulan",
            yaxis_title="Provinsi"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "📊 Box Plot per Bulan":
        st.markdown("### 📊 Distribusi Kunjungan per Bulan")
        
        monthly_cols = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                       'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
        
        fig = px.box(
            df[monthly_cols],
            title="Distribusi Kunjungan per Bulan"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "🎯 Scatter Plot":
        st.markdown("### 🎯 Scatter Plot: Bulan vs Bulan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_month = st.selectbox("Pilih Bulan X:", ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni'])
        
        with col2:
            y_month = st.selectbox("Pilih Bulan Y:", ['Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'])
        
        fig = px.scatter(
            df,
            x=x_month,
            y=y_month,
            hover_data=['Provinsi'],
            title=f"Scatter Plot: {x_month} vs {y_month}",
            size='Tahunan',
            color='Tahunan',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 