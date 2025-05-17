import streamlit as st
import sys
import joblib
import text_processor 

st.set_page_config(page_title="Mental Health Detector", layout="centered")

sys.modules['main'] = text_processor

# header 
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4A90E2;'>🧠 Mental Health Detector</h1>
        <p style='font-size: 18px; color: #555;'>Deteksi awal perasaanmu bisa jadi langkah pertama untuk pulih</p>
        <hr style='border: 1px solid #ccc;'/>
    </div>
""", unsafe_allow_html=True)

# Load models
try:
    model_relevansi = joblib.load("model_relevansi.pkl")
except Exception as e:
    st.error(f"❌ Gagal memuat model relevansi: {e}")
    st.stop()

try:
    model_kategori = joblib.load("model_kategori.pkl")
except Exception as e:
    st.error(f"❌ Gagal memuat model kategori: {e}")
    st.stop()

# Classification logic
def classify_tweet(text):
    relevansi = model_relevansi.predict([text])[0]
    if relevansi == 'Tidak':
        return "Tidak Relevan", "😕 Sepertinya teks ini tidak berhubungan dengan topik kesehatan mental. Coba lagi, ya. Kalau butuh bantuan, kami siap mendengarkan!", None
        #return "Tidak Relevan", "udah yappingnya?☝️🤓", None    
    else:
        kategori = model_kategori.predict([text])[0]

        if kategori == 'Terindikasi':
            return "Berisiko", "🔍 Sepertinya kamu mengalami beberapa gejala gangguan mental. Mungkin ini saatnya mulai mengenali lebih dalam perasaanmu. Konsultasi dengan ahli bisa sangat membantu.", kategori
        elif kategori == 'Penderita':
            return "Berisiko", "💔 Kelihatannya kamu sedang berjuang dengan gangguan mental. Tapi kamu tidak sendiri. Ada banyak cara untuk mendapatkan dukungan. Coba hubungi profesional, ya.", kategori
        elif kategori == 'Penyintas':
            return "Berisiko", "🌱 Kamu telah melalui masa sulit dan tetap bertahan. Itu luar biasa! Jika sesekali masih terasa berat, itu sangat manusiawi kok.", kategori
        elif kategori == 'Selfdiagnosed':
            return "Berisiko", "🧩 Kamu tampaknya sudah menyadari perasaanmu. Tapi tetap penting untuk verifikasi melalui bantuan profesional agar lebih jelas dan aman.", kategori

        return "Berisiko", f"Ada potensi kamu termasuk dalam kategori: **{kategori}**.", kategori

# UI Input
st.markdown("## ✍️ Ceritakan Perasaanmu")
user_input = st.text_area(
    "Tulis apa yang sedang kamu rasakan atau pikirkan...",
    placeholder="Contoh: Aku merasa sangat lelah dan tidak semangat akhir-akhir ini...",
    height=150
)

# Process Button
if st.button("🚀 Proses"):
    if user_input.strip():
        status, message, kategori = classify_tweet(user_input)

        if status == "Berisiko":
            st.markdown("## 📋 Hasil Deteksi")

            # Visualization
            progress_color = {
                "Terindikasi": "#9D7BFB",
                "Penderita": "#F44336",
                "Penyintas": "#4CAF50",
                "Selfdiagnosed": "#FF9800"
            }

            description_text = {
                "Terindikasi": "Ada indikasi awal dari gangguan mental.",
                "Penderita": "Sedang mengalami gangguan mental.",
                "Penyintas": "Pernah mengalami namun saat ini sudah lebih baik.",
                "Selfdiagnosed": "Sudah menyadari kondisi sendiri, perlu verifikasi."
            }

            st.markdown("### 🎯 Kategori Deteksi")
            st.markdown(f"""
            <div style='
                padding: 1em;
                background-color: {progress_color[kategori]}20;
                border-left: 6px solid {progress_color[kategori]};
                border-radius: 5px;
                margin-bottom: 1em;'>
                <strong style='color: {progress_color[kategori]}; font-size: 18px;'>{kategori}</strong><br>
                <span style='color: #333;'>{description_text[kategori]}</span>
            </div>
            """, unsafe_allow_html=True)

        # Always show the feedback message
        st.markdown("### 💌 Pesan untuk Kamu")
        st.success(message)
    else:
        st.warning("⚠️ Silakan isi dulu teksnya sebelum diproses.")

# Footer
st.markdown("""
    <hr style='border: 0.5px solid #ccc;'/>
    <div style='text-align: center; font-size: 13px; color: gray;'>
        Dibuat untuk membantu mengenali kesehatan mental — bukan sebagai diagnosis akhir. Konsultasikan dengan profesional untuk bantuan lebih lanjut.
    </div>
""", unsafe_allow_html=True)
