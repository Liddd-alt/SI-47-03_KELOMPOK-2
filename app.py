import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(
    page_title="Dashboard Wisatawan Nusantara Indonesia",
    page_icon="🏝️",
    layout="wide"
)

# Function to load and preprocess data
@st.cache_data
def load_data():
    # Read the CSV file
    df = pd.read_csv('DataWisatawanNusantara.csv', sep=';')
    
    # Melt the dataframe to convert months to rows
    df_melted = df.melt(
        id_vars=[df.columns[0]], 
        var_name='Bulan', 
        value_name='Jumlah_Wisatawan'
    )
    
    # Rename the province column
    df_melted = df_melted.rename(columns={df_melted.columns[0]: 'Provinsi'})
    
    # Convert '-' to 0 and clean the Jumlah_Wisatawan column
    df_melted['Jumlah_Wisatawan'] = df_melted['Jumlah_Wisatawan'].replace('-', '0')
    df_melted['Jumlah_Wisatawan'] = pd.to_numeric(df_melted['Jumlah_Wisatawan'], errors='coerce')
    
    # Convert month-year to datetime
    df_melted['Bulan'] = pd.to_datetime(df_melted['Bulan'], format='%b-%y')
    
    return df_melted

# Load the data
df = load_data()

# Title and description
st.title("📊 Dashboard Analisis Wisatawan Nusantara Indonesia")
st.markdown("""
Dashboard ini menampilkan analisis perjalanan wisatawan nusantara di Indonesia berdasarkan data provinsi.
Gunakan filter di sidebar untuk menyesuaikan visualisasi sesuai kebutuhan Anda.
""")

# Sidebar filters
st.sidebar.header("Filter Data")

# Year filter
years = df['Bulan'].dt.year.unique()
selected_years = st.sidebar.multiselect(
    "Pilih Tahun",
    options=years,
    default=years
)

# Province filter
provinces = sorted(df['Provinsi'].unique())
selected_provinces = st.sidebar.multiselect(
    "Pilih Provinsi",
    options=provinces,
    default=provinces[:5]  # Default to first 5 provinces
)

# Filter data based on selection
filtered_df = df[
    (df['Bulan'].dt.year.isin(selected_years)) &
    (df['Provinsi'].isin(selected_provinces))
]

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    total_wisatawan = filtered_df['Jumlah_Wisatawan'].sum()
    st.metric("Total Wisatawan", f"{total_wisatawan:,.0f}")

with col2:
    avg_wisatawan = filtered_df['Jumlah_Wisatawan'].mean()
    st.metric("Rata-rata Wisatawan per Bulan", f"{avg_wisatawan:,.0f}")

with col3:
    # Calculate year over year growth
    if len(selected_years) > 1:
        current_year = filtered_df[filtered_df['Bulan'].dt.year == max(selected_years)]['Jumlah_Wisatawan'].sum()
        prev_year = filtered_df[filtered_df['Bulan'].dt.year == max(selected_years)-1]['Jumlah_Wisatawan'].sum()
        yoy_growth = ((current_year - prev_year) / prev_year) * 100
        st.metric("Pertumbuhan YoY", f"{yoy_growth:,.1f}%")
    else:
        st.metric("Pertumbuhan YoY", "N/A")

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tren Wisatawan Over Time")
    # Line chart for tourist trends
    fig_trend = px.line(
        filtered_df,
        x='Bulan',
        y='Jumlah_Wisatawan',
        color='Provinsi',
        title='Tren Jumlah Wisatawan per Provinsi'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("Perbandingan Antar Provinsi")
    # Bar chart for province comparison
    province_totals = filtered_df.groupby('Provinsi')['Jumlah_Wisatawan'].sum().sort_values(ascending=True)
    fig_provinces = px.bar(
        province_totals,
        orientation='h',
        title='Total Wisatawan per Provinsi'
    )
    st.plotly_chart(fig_provinces, use_container_width=True)

# Monthly patterns
st.subheader("Pola Bulanan")
monthly_avg = filtered_df.copy()
monthly_avg['Bulan'] = monthly_avg['Bulan'].dt.month
monthly_avg = monthly_avg.groupby(['Provinsi', 'Bulan'])['Jumlah_Wisatawan'].mean().reset_index()
fig_monthly = px.line(
    monthly_avg,
    x='Bulan',
    y='Jumlah_Wisatawan',
    color='Provinsi',
    title='Rata-rata Wisatawan per Bulan'
)
fig_monthly.update_xaxes(ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        tickvals=list(range(1, 13)))
st.plotly_chart(fig_monthly, use_container_width=True)

# Heatmap visualization
st.subheader("Heatmap Wisatawan")
pivot_data = filtered_df.pivot_table(
    index='Provinsi',
    columns=pd.Grouper(key='Bulan', freq='M'),
    values='Jumlah_Wisatawan',
    aggfunc='sum'
)

fig_heatmap = px.imshow(
    pivot_data,
    title='Heatmap Jumlah Wisatawan per Provinsi dan Bulan',
    labels=dict(x='Bulan', y='Provinsi', color='Jumlah Wisatawan'),
    aspect='auto'
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Dashboard Wisatawan Nusantara Indonesia | Data source: DataWisatawanNusantara.csv</p>
</div>
""", unsafe_allow_html=True) 