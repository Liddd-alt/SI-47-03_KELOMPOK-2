# Dashboard Analisis Wisatawan Nusantara Indonesia 🏝️

Aplikasi dashboard interaktif untuk menganalisis dan memvisualisasikan tren perjalanan wisatawan nusantara di Indonesia berdasarkan provinsi asal selama periode 2022-2024.

## Fitur

- 📊 Visualisasi interaktif tren wisatawan
- 🔍 Filter berdasarkan provinsi dan tahun
- 📈 Analisis statistik dasar (total, rata-rata, pertumbuhan)
- 🗓️ Analisis tren bulanan dan tahunan
- 🔮 Prediksi jumlah wisatawan untuk periode mendatang

## Instalasi

1. Clone repository ini
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Penggunaan

1. Train model prediktif (hanya perlu dilakukan sekali):
```bash
python train_model.py
```

2. Jalankan dashboard:
```bash
streamlit run app.py
```

## Struktur Dashboard

### 1. Metrics Utama
- Total Wisatawan
- Rata-rata Bulanan
- Pertumbuhan Year-over-Year

### 2. Visualisasi
- Tren Perjalanan Wisatawan Over Time
- Rata-rata Wisatawan per Bulan
- Total Wisatawan per Tahun
- Prediksi Jumlah Wisatawan

### 3. Fitur Interaktif
- Filter Tahun
- Filter Provinsi
- Pemilihan Provinsi untuk Prediksi

## Data

Data bersumber dari statistik perjalanan wisatawan nusantara berdasarkan provinsi asal periode 2022-2024.

## Model Prediktif

Model menggunakan Facebook Prophet untuk time series forecasting dengan mempertimbangkan:
- Seasonality tahunan
- Tren jangka panjang
- Pola musiman

## Requirements

- Python 3.8+
- pandas
- numpy
- streamlit
- plotly
- scikit-learn
- prophet
- joblib 