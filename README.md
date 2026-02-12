# ⚡ Prediksi Biaya Listrik

Aplikasi prediksi biaya listrik bulanan menggunakan Linear Regression untuk menganalisis pengaruh tipe pelanggan, wilayah, luas bangunan, dan jumlah penghuni terhadap konsumsi energi.

## 🎯 Tujuan Proyek

Proyek ini dibuat sebagai bagian dari Forum Group Discussion (FGD) Mata Kuliah Praktikum Unggulan untuk memahami dan menerapkan machine learning dalam konteks prediksi konsumsi energi listrik. Proyek ini dirancang mengikuti metodologi CRISP-DM agar proses pengembangan model dilakukan secara sistematis, terstruktur, dan mudah dievaluasi.

**Link Demonstrasi:** [https://prediksi-biaya-listrik.streamlit.app/](https://prediksi-biaya-listrik.streamlit.app/)

## 📈 Dataset

Dataset yang digunakan berasal dari Kaggle ([Residential and Commercial Energy Cost Dataset](https://www.kaggle.com/datasets/andreylss/residential-and-commercial-energy-cost-dataset)), yang berisi:

| Fitur | Deskripsi |
|-------|-----------|
| Tipe_Customer | Jenis pelanggan (Residential/Commercial) |
| Region | Wilayah tempat tinggal |
| Luas_Bangunan_m2 | Luas bangunan dalam meter persegi |
| Jumlah_Penghuni | Jumlah orang yang tinggal |
| Biaya_Listrik | Target prediksi - biaya listrik ($) |

## ✨ Fitur Aplikasi

- 🏠 **Halaman Informasi** - Penjelasan aplikasi dan visualisasi data
- 📊 **Halaman Prediksi** - Input data dan hasil prediksi biaya listrik
- 📈 **Visualisasi** - Distribusi data, bar chart, dan pie chart
- 🎯 **Number Line** - Posisi prediksi relatif terhadap range data

## 🛠 Teknologi

| Package | Keterangan |
|---------|------------|
| Python 3.x | Bahasa pemrograman |
| Streamlit | Web Framework |
| Scikit-learn | Machine Learning (Linear Regression) |
| Pandas | Data Processing |
| Plotly | Visualisasi Interaktif |

## 📦 Instalasi

```bash
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📚 Struktur File

```
.
├── app.py                              # Aplikasi Streamlit
├── energy_consumption_modified.csv     # Dataset
├── requirements.txt                    # Dependencies
├── README.md                           # File ini
├── Prediksi Biaya Listrik.ipynb        # Notebook analisis & training
└── .streamlit/
    └── config.toml                     # Konfigurasi tema Streamlit
```

## 🖥️ Kebutuhan Sistem

- **Sistem Operasi (OS):** Windows 10, Mac, Linux
- **Web Browser:** Google Chrome, Microsoft Edge, Firefox, Safari

## 🥼 Dibuat oleh

- Zahwa Annisa Hendajani
- Wicaksono Hanif Supriyanto
- Muhammad Riza Jamalul Akbar

## 📚 Referensi

- Andreyls. Residential and Commercial Energy Cost Dataset. Kaggle. [https://www.kaggle.com/datasets/andreylss/residential-and-commercial-energy-cost-dataset](https://www.kaggle.com/datasets/andreylss/residential-and-commercial-energy-cost-dataset)
- Data Science PM. What is CRISP DM. [https://www.datascience-pm.com/crisp-dm-2/](https://www.datascience-pm.com/crisp-dm-2/)

## 📝 Lisensi

Copyright © 2026 Pengelola MK Praktikum Unggulan (Praktikum DGX), Universitas Gunadarma
