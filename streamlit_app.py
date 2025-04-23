import streamlit as st
import sys
import joblib
import text_processor  # Your real module with TextPreprocessor

# ✅ Must come first
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# ✅ Patch the pickle loader so it finds TextPreprocessor
sys.modules['main'] = text_processor

# --- Load models
try:
    model_relevansi = joblib.load("model_relevansi_SVM_full_pipeline.pkl")
except Exception as e:
    st.error(f"❌ Gagal memuat model relevansi: {e}")
    st.stop()

try:
    model_kategori = joblib.load("model_kategori_RF_full_pipeline.pkl")
except Exception as e:
    st.error(f"❌ Gagal memuat model kategori: {e}")
    st.stop()

# --- Classification logic
def classify_tweet(text):
    relevansi = model_relevansi.predict([text])[0]
    if relevansi == 'Tidak':
        return "Tidak Relevan", "Kayaknya teks yang kamu masukin nggak nyambung sama topik kesehatan mental. Coba lagi deh kalau ada yang mau kamu ceritain tentang itu. Kalau butuh bantuan, kami siap denger kok!"
    else:
        kategori = model_kategori.predict([text])[0]
        
        if kategori == 'Terindikasi':
            return "Berisiko", "Kayaknya kamu lagi ngalamin beberapa gejala yang bisa jadi gangguan mental. Ini mungkin jadi langkah pertama buat lebih ngerti perasaanmu. Kalau kamu ngerasa nggak nyaman, coba deh ngobrol sama orang yang ahli, kayak psikolog atau konselor. Ingat, nggak ada salahnya minta bantuan, itu malah langkah berani dan bisa ngebantu banget!"
        
        elif kategori == 'Penderita':
            return "Berisiko", "Kelihatannya kamu lagi berjuang dengan gangguan mental. Ini nggak mudah, tapi percayalah, kamu nggak sendirian. Jangan ragu untuk cari bantuan dari seorang profesional. Kadang ngobrol sama orang yang ngerti bisa bantu banget untuk merasa lebih baik, dan kamu punya hak untuk itu."
        
        elif kategori == 'Penyintas':
            return "Berisiko", "Kamu udah melalui banyak hal dan tetap bertahan, itu luar biasa! Kadang-kadang, meskipun kita udah merasa lebih baik, ada kalanya perasaan berat datang lagi, itu wajar kok. Kalau kamu butuh dukungan lagi, jangan ragu buat cari bantuan. Kami bangga sama ketangguhanmu!"
        
        elif kategori == 'Selfdiagnosed':
            return "Berisiko", "Kayaknya kamu udah cukup ngerti soal perasaanmu, tapi tetap aja, nggak ada salahnya kalau coba cek sama ahli untuk bener-bener memastikan. Bisa jadi ada cara yang lebih efektif untuk ngatasin apa yang kamu rasain, dan profesional bisa bantu kamu lebih jelas dalam prosesnya."

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
