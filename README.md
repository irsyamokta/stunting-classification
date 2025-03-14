# Klasifikasi Status Gizi Balita

Proyek ini bertujuan untuk mengklasifikasikan status gizi balita berdasarkan berbagai faktor kesehatan dan pertumbuhan. Model yang digunakan dalam proyek ini adalah **Random Forest Classifier** dan **Support Vector Machine (SVM)**. Setelah dilakukan evaluasi, model **KNN** dipilih sebagai model terbaik karena memberikan akurasi tertinggi.

## Dataset
Dataset yang digunakan dalam proyek ini berasal dari **[Kaggle](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows)**. Dataset ini berisi berbagai fitur kesehatan dan pertumbuhan balita, seperti:
- **Age (Month)**: Umur balita dalam bulan (0 hingga 60 bulan). Rentang usia ini penting untuk menentukan fase pertumbuhan anak dan membandingkannya dengan standar pertumbuhan sehat.
- **Gender**: Kategori terdiri dari 'male' dan 'female'. Gender merupakan faktor penting dalam menganalisis pola pertumbuhan dan risiko stunting.
- **Height (cm)**: Tinggi badan balita dalam sentimeter. Indikator kunci untuk menilai pertumbuhan fisik anak di bawah lima tahun.
- **Nutrition Status**: Status gizi dikategorikan menjadi:
  - **Severely Stunted**: Kondisi sangat serius (<-3 SD).
  - **Stunted**: Kondisi terhambat (-3 SD hingga <-2 SD).
  - **Normal**: Status gizi sehat (-2 SD hingga +3 SD).
  - **Tall**: Pertumbuhan di atas rata-rata (>+3 SD).

## Instalasi dan Persyaratan
Sebelum menjalankan proyek ini, pastikan Anda memiliki **Python 3.x** dan install library yang terdapat di requirements.txt:

```bash
pip install -r requirements.txt
```

## Langkah-langkah Implementasi
1. **Import Library**: Memuat pustaka yang diperlukan.
2. **Load Dataset**: Membaca data dan melakukan eksplorasi awal.
3. **Preprocessing Data**: Menangani duplikasi data, menghapus outlier dan pembagian dataset menjadi training dan testing.
4. **Training Model**:
   - Membangun model **KNN** dan **SVM**
5. **Evaluasi Model**:
   - Menghitung metrik evaluasi seperti **akurasi, precision, recall, dan F1-score**.
   - Membandingkan performa KNN dan SVM.
6. **Kesimpulan**: Menentukan algoritma terbaik berdasarkan hasil evaluasi.

## Hasil Perbandingan Model
| Model | Akurasi |
|--------|----------|
| KNN | **93%** |
| SVM | 87% |

Berdasarkan hasil evaluasi, **KNN memberikan akurasi lebih tinggi dibandingkan SVM**, sehingga dipilih sebagai model akhir untuk klasifikasi diabetes.

## Cara Menjalankan Proyek
1. Clone repositori ini:
   ```bash
   git clone https://github.com/irsyamokta/stunting-classification.git
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd stunting-classification
   ```
3. Jalankan notebook Jupyter:
   ```bash
   jupyter notebook
   ```
4. Buka file `stunting_classification.ipynb` dan jalankan setiap sel untuk melihat hasilnya.

## Deployment
Proyek ini sudah publik dan bisa diakses melalui tautan berikut https://stunting-classification.streamlit.app/

## Kontribusi
Kontribusi sangat diterima! Jika ingin menambahkan fitur atau meningkatkan model, silakan buat **pull request** atau buka **issue**.

## Author
[@irsyamokta](https://github.com/irsyamokta)
