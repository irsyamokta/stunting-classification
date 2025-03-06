import streamlit as st
import pandas as pd
import joblib

model_path = "./model/gizi_model.pkl"
model = joblib.load(model_path)
label_encoders = joblib.load("./model/label_encoders.pkl")

gender_mapping = {
    "Laki-laki": "laki-laki",
    "Perempuan": "perempuan"
}

st.title("Prediksi Status Gizi Balita")
st.write("Masukkan data balita untuk mendapatkan prediksi status gizinya.")

umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, step=1)
jenis_kelamin_display = st.radio("Jenis Kelamin", list(gender_mapping.keys()))
jenis_kelamin = gender_mapping[jenis_kelamin_display]
tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=150.0, step=0.1)

jenis_kelamin_encoded = label_encoders["Jenis Kelamin"].transform([jenis_kelamin])[0]
input_data = pd.DataFrame([[umur, jenis_kelamin_encoded, tinggi_badan]], columns=["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"])

if st.button("Prediksi"):
    prediksi = model.predict(input_data)[0]
    status_gizi = label_encoders["Status Gizi"].inverse_transform([prediksi])[0].title()
    
    info = {
        "Severely Stunted": "Kondisi sangat serius (<-3 SD), anak mengalami kekurangan gizi yang parah dan memerlukan intervensi segera.",
        "Stunted": "Kondisi anak mengalami stunting (-3 SD hingga <-2 SD), yang berarti pertumbuhan terhambat dan memerlukan perhatian lebih.",
        "Normal": "Status gizi sehat (-2 SD hingga +3 SD), pertumbuhan anak sesuai dengan standar.",
        "Tinggi": "Pertumbuhan di atas rata-rata (>+3 SD), anak memiliki tinggi badan lebih dari standar untuk usianya."
    }
    
    st.success(f"Prediksi Status Gizi: {status_gizi}")
    st.info(info.get(status_gizi, "Informasi tidak tersedia"))
