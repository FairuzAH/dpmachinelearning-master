import streamlit as st
import sys
import joblib
import text_processor  # Your real module with TextPreprocessor

# ✅ Must come first
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# ✅ Patch the pickle loader so it finds TextPreprocessor
sys.modules['main'] = text_processor

# --- Decorative header (optional image banner)
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4A90E2;'>🧠 Mental Health Detector</h1>
        <p style='font-size: 18px; color: #555;'>Deteksi awal perasaanmu bisa jadi langkah pertama untuk pulih</p>
        <hr style='border: 1px solid #ccc;'/>
    </div>
""", unsafe_allow_html=True)

# --- Load models
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

# --- Classification logic
def classify_tweet(text):
    relevansi = model_relevansi.predict([text])[0]
    if relevansi == 'Tidak':
        return "Tidak Relevan", "😕 Sepertinya teks ini tidak berhubungan dengan topik kesehatan mental. Coba lagi, ya. Kalau butuh bantuan, kami siap mendengarkan!", None
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

# --- UI Input
st.markdown("## ✍️ Ceritakan Perasaanmu")
user_input = st.text_area(
    "Tulis apa yang sedang kamu rasakan atau pikirkan...",
    placeholder="Contoh: Aku merasa sangat lelah dan tidak semangat akhir-akhir ini...",
    height=150
)

# --- Process Button
if st.button("🚀 Proses"):
    if user_input.strip():
        status, message, kategori = classify_tweet(user_input)

        if status == "Berisiko":
            st.markdown("## 📋 Hasil Deteksi")

            # --- Visualization
            if kategori == 'Terindikasi':
                color, label = "#9D7BFB", "Terindikasi"
            elif kategori == 'Penderita':
                color, label = "#F44336", "Penderita"
            elif kategori == 'Penyintas':
                color, label = "#4CAF50", "Penyintas"
            elif kategori == 'Selfdiagnosed':
                color, label = "#FF9800", "Selfdiagnosed"

            st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='width: 100px; height: 100px; border-radius: 50%;
                                background: conic-gradient({color} 0% 65%, #EEE 65% 100%);
                                margin: auto;'></div>
                    <p style='font-weight: bold; color: {color}; font-size: 20px;'>{label}</p>
                </div>
            """, unsafe_allow_html=True)

        # --- Always show the feedback message
        st.markdown("### 💌 Pesan untuk Kamu")
        st.success(message)
    else:
        st.warning("⚠️ Silakan isi dulu teksnya sebelum diproses.")

# --- Footer
st.markdown("""
    <hr style='border: 0.5px solid #ccc;'/>
    <div style='text-align: center; font-size: 13px; color: gray;'>
        Dibuat untuk membantu mengenali kesehatan mental — bukan sebagai diagnosis akhir. Konsultasikan dengan profesional untuk bantuan lebih lanjut.
    </div>
""", unsafe_allow_html=True)
