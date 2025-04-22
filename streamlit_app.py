import streamlit as st
import sys
import joblib
import text_processor  # Your real module with TextPreprocessor

# ✅ Must come first
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# ✅ Patch the pickle loader so it finds TextPreprocessor
sys.modules['main'] = text_processor

# --- UI: Loading message
st.markdown("## ⏳ Memuat model dan preprocessing...")

# --- Load models
try:
    model_relevansi = joblib.load("model_relevansi_SVM_full_pipeline.pkl")
    st.success("✅ Model relevansi berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model relevansi: {e}")
    st.stop()

try:
    model_kategori = joblib.load("model_kategori_RF_full_pipeline.pkl")
    st.success("✅ Model kategori berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model kategori: {e}")
    st.stop()

# --- Classification logic
def classify_tweet(text):
    relevansi = model_relevansi.predict([text])[0]
    if relevansi == 'Tidak':
        return "Tidak Relevan", "Tulisan tidak relevan dengan gangguan kesehatan mental"
    else:
        kategori = model_kategori.predict([text])[0]
        return "Berisiko", f"Ada potensi kamu termasuk dalam kategori: **{kategori}**."

# --- UI input
st.markdown("## Apa yang ada di pikiranmu?")
user_input = st.text_area(
    "Tulis tentang bagaimana perasaanmu, apa yang sedang kamu pikirkan, atau hal lain yang ingin kamu ungkapkan",
    placeholder="Contoh: Aku merasa sangat lelah dan tidak semangat akhir-akhir ini...",
    height=120
)

# --- Process button
if st.button("Proses") and user_input.strip():
    status, message = classify_tweet(user_input)

    # --- Display status
    st.markdown("### Status Deteksi")
    if status == "Berisiko":
        st.markdown("""
        <div style='text-align: center;'>
            <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#9D7BFB 0% 65%, #E3DAFB 65% 100%); margin: auto;'></div>
            <p style='font-weight: bold; color: #6C3FC5;'>Terindikasi</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#B0BEC5 0% 30%, #ECEFF1 30% 100%); margin: auto;'></div>
            <p style='font-weight: bold; color: #546E7A;'>Tidak Relevan</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Message
    st.markdown("### Pesan untuk Kamu")
    st.write(message)
