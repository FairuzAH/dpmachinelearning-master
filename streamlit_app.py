import streamlit as st
import joblib

# Load models
model_relevansi = joblib.load("model_relevansi_SVM_full_pipeline.pkl")
model_kategori = joblib.load("model_Kategori_RF_full_pipeline.pkl")

# Prediction logic
def classify_tweet(text):
    relevansi = model_relevansi.predict([text])[0]
    if relevansi == 'Tidak':
        return "Tidak Relevan", "Tulisan tidak relevan dengan gangguan kesehatan mental"
    else:
        kategori = model_kategori.predict([text])[0]
        return "Berisiko", f"Ada potensi kamu termasuk dalam kategori: **{kategori}**."

# Streamlit Page Config
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# --- Title & Input ---
st.markdown("## Apa yang ada di pikiranmu?")
user_input = st.text_area(
    "Tulis tentang bagaimana perasaanmu, apa yang sedang kamu pikirkan, atau hal lain yang ingin kamu ungkapkan",
    placeholder="Contoh: Aku merasa sangat lelah dan tidak semangat akhir-akhir ini...",
    height=120
)

# --- Button & Prediction ---
if st.button("Proses"):
    status, message = classify_tweet(user_input)

    # --- Display Status ---
    st.markdown("### Status Deteksi")
    if status == "Terindikasi":
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

    # --- Message for User ---
    st.markdown("### Pesan untuk Kamu")
    st.write(message)
