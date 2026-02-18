import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prediksi Biaya Listrik",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    /* Background utama - kuning sangat muda, hampir krem */
    .stApp {
        background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    /* Sidebar - kuning muda lembut */
    [data-testid="stSidebar"] {
        background: #fef9e7;
    }
    
    /* Card style - putih dengan border kuning */
    .info-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(251, 191, 36, 0.15);
        border: 1px solid #fcd34d;
        margin-bottom: 20px;
        color: #78350f;
    }
    
    .info-card h3 {
        color: #92400e;
    }
    
    /* Metric card - gradient amber/kuning gelap */
    .metric-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    
    /* Primary button - amber */
    .stButton > button[kind="primary"] {
        background-color: #f59e0b;
        border-color: #f59e0b;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #d97706;
        border-color: #d97706;
    }
    
    /* Title dan text - coklat tua */
    h1, h2, h3 {
        color: #78350f !important;
    }
    
    /* Footer */
    .footer-bar {
        padding: 15px;
        text-align: center;
        margin-top: 20px;
        color: #78350f;
        border-top: 1px solid #fcd34d;
    }
    
    .footer-bar a {
        color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA DAN TRAIN MODEL
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("energy_consumption_modified.csv")
    df.drop(columns='id_Customer', inplace=True)
    return df

@st.cache_resource
def train_model(_df):
    X = _df.drop("Biaya_Listrik", axis=1)
    y = _df["Biaya_Listrik"]
    
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_encoded, y)
    return model, encoder

df = load_data()
model, encoder = train_model(df)

# =====================================================
# SIDEBAR NAVIGASI
# =====================================================
st.sidebar.title("Menu Navigasi")
st.sidebar.markdown("---")

halaman = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Informasi", "📊 Prediksi"]
)

st.sidebar.markdown("""
<div class="footer-bar" style="font-size: 11px;">
    © 2026 Praktikum DGX<br>
    Universitas Gunadarma
</div>
""", unsafe_allow_html=True)

# =====================================================
# HALAMAN INFORMASI
# =====================================================
if halaman == "🏠 Informasi":
    st.title("⚡ Aplikasi Prediksi Biaya Listrik")
    st.markdown("Menggunakan Algoritma **Regresi Linier**")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📖 Tentang Aplikasi</h3>
            <p>Aplikasi ini memprediksi biaya listrik bulanan berdasarkan:</p>
            <ul>
                <li><b>Tipe Customer</b> - Jenis pelanggan (Residential/Commercial)</li>
                <li><b>Region</b> - Wilayah tempat tinggal</li>
                <li><b>Luas Bangunan (m²)</b> - Semakin luas, semakin tinggi konsumsi</li>
                <li><b>Jumlah Penghuni</b> - Lebih banyak orang = lebih banyak pemakaian</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🔧 Cara Menggunakan</h3>
            <ol>
                <li>Pilih menu <b>Prediksi</b> di sidebar</li>
                <li>Pilih tipe customer dan region</li>
                <li>Masukkan luas bangunan dan jumlah penghuni</li>
                <li>Klik tombol <b>Prediksi</b></li>
                <li>Lihat hasil estimasi biaya listrik</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Info Model</h3>
            <p><b>Algoritma:</b> Linear Regression</p>
            <p><b>Fitur:</b> Tipe Customer, Region, Luas Bangunan, Jumlah Penghuni</p>
            <p><b>Encoding:</b> One-Hot Encoding</p>
            <p><b>Target:</b> Biaya Listrik ($)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Info Dataset - centered below
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>📊 Info Dataset</h3>", unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Data", f"{len(df):,}")
        with col_m2:
            st.metric("Rata-rata", f"${df['Biaya_Listrik'].mean():,.2f}")
        with col_m3:
            st.metric("Min", f"${df['Biaya_Listrik'].min():,.2f}")
        with col_m4:
            st.metric("Max", f"${df['Biaya_Listrik'].max():,.2f}")
        
        # Visualisasi distribusi biaya listrik
        import plotly.express as px
        fig_hist = px.histogram(
            df, 
            x='Biaya_Listrik', 
            nbins=30,
            title='Distribusi Biaya Listrik',
            labels={'Biaya_Listrik': 'Biaya Listrik ($)', 'count': 'Jumlah'}
        )
        fig_hist.update_layout(height=350, showlegend=False)
        fig_hist.update_traces(marker_color='#3b82f6')
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Bar chart rata-rata biaya per Region dan Tipe Customer
        col_bar1, col_bar2 = st.columns(2)
        
        with col_bar1:
            avg_by_region = df.groupby('Region')['Biaya_Listrik'].mean().reset_index()
            fig_region = px.bar(
                avg_by_region,
                x='Region',
                y='Biaya_Listrik',
                title='Rata-rata Biaya per Region',
                labels={'Biaya_Listrik': 'Biaya ($)', 'Region': 'Region'}
            )
            fig_region.update_layout(height=350, showlegend=False)
            fig_region.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col_bar2:
            avg_by_type = df.groupby('Tipe_Customer')['Biaya_Listrik'].mean().reset_index()
            fig_type = px.bar(
                avg_by_type,
                x='Tipe_Customer',
                y='Biaya_Listrik',
                title='Rata-rata Biaya per Tipe Customer',
                labels={'Biaya_Listrik': 'Biaya ($)', 'Tipe_Customer': 'Tipe Customer'}
            )
            fig_type.update_layout(height=350, showlegend=False)
            fig_type.update_traces(marker_color='#f59e0b')
            st.plotly_chart(fig_type, use_container_width=True)
        
        # Pie chart proporsi Tipe Customer
        customer_counts = df['Tipe_Customer'].value_counts().reset_index()
        customer_counts.columns = ['Tipe_Customer', 'Jumlah']
        fig_pie = px.pie(
            customer_counts,
            values='Jumlah',
            names='Tipe_Customer',
            title='Proporsi Tipe Customer',
            color_discrete_sequence=['#3b82f6', '#f59e0b', '#10b981']
        )
        fig_pie.update_layout(height=450)
        st.plotly_chart(fig_pie, use_container_width=True)

# =====================================================
# HALAMAN PREDIKSI
# =====================================================
elif halaman == "📊 Prediksi":
    st.title("📊 Prediksi Biaya Listrik")
    st.markdown("Masukkan data untuk mendapatkan estimasi biaya listrik bulanan.")
    
    st.markdown("---")
    
    # Info Rentang Data
    st.markdown("""
    <div class="info-card">
        <h3>📋 Informasi Rentang Data</h3>
        <p>Berikut adalah rentang nilai dari dataset yang digunakan untuk melatih model:</p>
        <ul>
            <li><b>Luas Bangunan:</b> 17 - 77 m² (nilai yang tersedia: 17, 24, 45, 52, 77)</li>
            <li><b>Jumlah Penghuni:</b> 1 - 4 orang</li>
            <li><b>Biaya Listrik:</b> $52.56 - $158.14 per bulan</li>
        </ul>
        <p><i>Gunakan rentang ini sebagai acuan untuk input yang optimal.</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input Data (centered)
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        st.subheader("📝 Input Data")
        
        tipe_customer = st.selectbox(
            "Tipe Customer",
            options=df['Tipe_Customer'].unique(),
            help="Pilih jenis pelanggan listrik"
        )
        
        region = st.selectbox(
            "Region",
            options=df['Region'].unique(),
            help="Pilih wilayah tempat tinggal"
        )
        
        luas_bangunan = st.number_input(
            "Luas Bangunan (m²)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=1.0,
            help="Masukkan luas bangunan dalam meter persegi"
        )
        
        jumlah_penghuni = st.number_input(
            "Jumlah Penghuni",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Masukkan jumlah orang yang tinggal"
        )
        
        prediksi_button = st.button("Prediksi Biaya Listrik", type="primary", use_container_width=True)
        
        if prediksi_button:
            # Validasi input
            if luas_bangunan < 1 or jumlah_penghuni < 1:
                st.error("⚠️ Value must be greater than or equal to 1")
            else:
                # Lakukan prediksi
                input_data = pd.DataFrame({
                    'Tipe_Customer': [tipe_customer],
                    'Region': [region],
                    'Luas_Bangunan_m2': [luas_bangunan],
                    'Jumlah_Penghuni': [jumlah_penghuni]
                })
                
                input_encoded = encoder.transform(input_data)
                prediksi_biaya = model.predict(input_encoded)[0]
                
                st.markdown("---")
                
                # Tampilkan hasil
                st.subheader("✅ Hasil Prediksi")
                
                st.markdown(f"""
                <div class="metric-card">
                    <h2>Estimasi Biaya Listrik</h2>
                    <h1 style="font-size: 2.5em;">${prediksi_biaya:,.2f}</h1>
                    <p>per bulan</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualisasi dengan number line
                st.subheader("📈 Visualisasi Biaya Prediksi")
                
                min_val = df['Biaya_Listrik'].min()
                max_val = df['Biaya_Listrik'].max()
                
                fig = go.Figure()
                
                # Garis horizontal (background bar)
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='#3b82f6', width=8),
                    hoverinfo='skip'
                ))
                
                # Marker untuk posisi prediksi
                fig.add_trace(go.Scatter(
                    x=[prediksi_biaya],
                    y=[0],
                    mode='markers',
                    marker=dict(
                        size=16,
                        color='#f59e0b',
                        symbol='circle',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate=f'Prediksi: ${prediksi_biaya:,.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    height=120,
                    showlegend=False,
                    xaxis=dict(
                        title='Biaya Listrik ($)',
                        tickprefix='$',
                        tickformat=',.0f',
                        range=[min_val - 10, max_val + 10]
                    ),
                    yaxis=dict(
                        visible=False,
                        range=[-0.5, 0.5]
                    ),
                    margin=dict(l=20, r=20, t=10, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=False)
                
                # Detail input
                with st.expander("📋 Detail Input"):
                    st.write(f"- **Tipe Customer:** {tipe_customer}")
                    st.write(f"- **Region:** {region}")
                    st.write(f"- **Luas Bangunan:** {luas_bangunan} m²")
                    st.write(f"- **Jumlah Penghuni:** {jumlah_penghuni} orang")

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <div class="footer-bar">
        Copyright © 2026 by Pengelola MK Praktikum Unggulan (Praktikum DGX), Universitas Gunadarma
        <br>
        <a href="https://www.praktikum-hpc.gunadarma.ac.id/" target="_blank">
            https://www.praktikum-hpc.gunadarma.ac.id/
        </a>
        <br>
        <a href="https://www.hpc-hub.gunadarma.ac.id/" target="_blank">
            https://www.hpc-hub.gunadarma.ac.id/
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
