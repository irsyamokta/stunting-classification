import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

model_path = "./model/gizi_model.pkl"
model = joblib.load(model_path)
label_encoders = joblib.load("./model/label_encoders.pkl")

gender_mapping = {
    "Laki-laki": "laki-laki",
    "Perempuan": "perempuan"
}

@st.cache_data
def load_data():
    return pd.read_csv("./dataset/data_balita.csv")

df = load_data()

st.title("Prediksi Status Gizi Balita Menggunakan Algoritma Random Forest Classifier")
st.write("Model ini dibuat untuk memprediksi Status Gizi Balita berdasarkan umur (bulan), jenis kelamin, dan tinggi badan (cm). Model ini menggunakan algoritma Random Forest Classifier untuk mengklasifikasikan status gizi balita menjadi beberapa kategori, yaitu: Severely Stunted (Gizi Buruk), Stunted (Gizi Kurang), Normal (Gizi Baik), Tinggi (Gizi Lebih)")

st.subheader("Data Frame")
st.dataframe(df.head(10))
st.write(f"Total data dalam dataset: {df.shape[0]}")

st.subheader("Total Missing Value")
st.write(df.isnull().sum())

st.subheader("Distribusi Jenis Kelamin")
gender_counts = df["Jenis Kelamin"].value_counts()
total_data = len(df)
gender_percentages = (gender_counts / total_data) * 100

plt.figure(figsize=(6, 4))
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="colorblind")

for i, v in enumerate(gender_counts.values):
    percentage = gender_percentages[i]
    offset = max(0.02 * v, 2)
    ax.text(i, v + offset, f"{v} ({percentage:.1f}%)", ha='center', fontsize=8)

plt.xlabel("Jenis Kelamin")
plt.ylabel("Jumlah")
st.pyplot(plt)

st.subheader("Distribusi Status Gizi")
status_counts = df["Status Gizi"].value_counts()
total_data = len(df)
status_percentages = (status_counts / total_data) * 100

plt.figure(figsize=(6, 4))
ax = sns.barplot(x=status_counts.index, y=status_counts.values, palette="viridis")

for i, v in enumerate(status_counts.values):
    percentage = status_percentages[i]
    offset = max(0.02 * v, 2)
    ax.text(i, v + offset, f"{v} ({percentage:.1f}%)", ha='center', fontsize=8)

plt.xlabel("Status Gizi")
plt.ylabel("Jumlah")
st.pyplot(plt)

st.subheader("Disribusi Status Gizi Berdasarkan Jenis Kelamin")
gender_status_counts = df.groupby("Jenis Kelamin")["Status Gizi"].value_counts().unstack()
gender_totals = df["Jenis Kelamin"].value_counts()

plt.figure(figsize=(8, 6))
ax = sns.countplot(x="Jenis Kelamin", hue="Status Gizi", data=df, palette="colorblind")

for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            x_pos = bar.get_x() + bar.get_width() / 2 
            # offset = max(0.02 * height, 2)
            ax.text(x_pos, height + 0.01 * max(gender_totals), f"{int(height)}",
                    ha='center', fontsize=8, color='black')

plt.xlabel("Jenis Kelamin")
plt.ylabel("Jumlah")
plt.legend(title="Status Gizi", loc='upper left', fontsize=8)
st.pyplot(plt)

st.subheader("Evaluasi Model")
df["Jenis Kelamin"] = label_encoders["Jenis Kelamin"].transform(
    df["Jenis Kelamin"])
X_test = df.drop(columns=["Status Gizi"])
y_test = label_encoders["Status Gizi"].transform(df["Status Gizi"])
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"🎯 **Akurasi Model:** {accuracy:.2%}")

report = classification_report(y_test, y_pred, target_names=label_encoders["Status Gizi"].classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

st.subheader("ROC Curve")
y_score = model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
st.pyplot(plt)

st.subheader("Pengujian Model")
st.write("Masukkan data balita untuk mendapatkan prediksi status gizinya.")

umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, step=1)
jenis_kelamin_display = st.radio("Jenis Kelamin", list(gender_mapping.keys()))
jenis_kelamin = gender_mapping[jenis_kelamin_display]
tinggi_badan = st.number_input(
    "Tinggi Badan (cm)", min_value=30.0, max_value=150.0, step=0.1)

jenis_kelamin_encoded = label_encoders["Jenis Kelamin"].transform([jenis_kelamin])[
    0]
input_data = pd.DataFrame([[umur, jenis_kelamin_encoded, tinggi_badan]], columns=["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"])

if st.button("Prediksi"):
    prediksi = model.predict(input_data)[0]
    status_gizi = label_encoders["Status Gizi"].inverse_transform([prediksi])[
        0].title()

    info = {
        "Severely Stunted": "Kondisi sangat serius (<-3 SD), anak mengalami kekurangan gizi yang parah dan memerlukan intervensi segera.",
        "Stunted": "Kondisi anak mengalami stunting (-3 SD hingga <-2 SD), yang berarti pertumbuhan terhambat dan memerlukan perhatian lebih.",
        "Normal": "Status gizi sehat (-2 SD hingga +3 SD), pertumbuhan anak sesuai dengan standar.",
        "Tinggi": "Pertumbuhan di atas rata-rata (>+3 SD), anak memiliki tinggi badan lebih dari standar untuk usianya."
    }

    st.success(f"Prediksi Status Gizi: {status_gizi}")
    st.info(info.get(status_gizi, "Informasi tidak tersedia"))
    
st.subheader("Kesimpulan")
st.write("Model ini dapat digunakan sebagai alat bantu dalam monitoring status gizi balita, khususnya bagi tenaga kesehatan atau orang tua yang ingin mengetahui apakah anak mereka mengalami gizi buruk, kurang gizi, atau memiliki status gizi normal. Meskipun model ini memiliki performa yang baik, hasil prediksi tetap harus dikombinasikan dengan pemeriksaan langsung oleh ahli gizi atau dokter untuk tindakan lebih lanjut.")