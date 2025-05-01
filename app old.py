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
        return "Tidak Relevan", "Kayaknya teks yang kamu masukin nggak nyambung sama topik kesehatan mental. Coba lagi deh kalau ada yang mau kamu ceritain tentang itu. Kalau butuh bantuan, kami siap denger kok!", None  # Return None for kategori when not relevant
    else:
        kategori = model_kategori.predict([text])[0]
        
        print(f"Predicted kategori: {kategori}")  # Debug print to verify prediction

        # Return the status, message, and kategori
        if kategori == 'Terindikasi':
            return "Berisiko", "Kayaknya kamu lagi ngalamin beberapa gejala yang bisa jadi gangguan mental. Ini mungkin jadi langkah pertama buat lebih ngerti perasaanmu. Kalau kamu ngerasa nggak nyaman, coba deh ngobrol sama orang yang ahli, kayak psikolog atau konselor.", kategori
        
        elif kategori == 'Penderita':
            return "Berisiko", "Kelihatannya kamu lagi berjuang dengan gangguan mental. Ini nggak mudah, tapi percayalah, kamu nggak sendirian. Jangan ragu untuk cari bantuan dari seorang profesional.", kategori
        
        elif kategori == 'Penyintas':
            return "Berisiko", "Kamu udah melalui banyak hal dan tetap bertahan, itu luar biasa! Kadang-kadang, meskipun kita udah merasa lebih baik, ada kalanya perasaan berat datang lagi, itu wajar kok.", kategori
        
        elif kategori == 'Selfdiagnosed':
            return "Berisiko", "Kayaknya kamu udah cukup ngerti soal perasaanmu, tapi tetap aja, nggak ada salahnya kalau coba cek sama ahli untuk bener-bener memastikan.", kategori

        return "Berisiko", f"Ada potensi kamu termasuk dalam kategori: **{kategori}**.", kategori

# --- UI input
st.markdown("## Apa yang ada di pikiranmu?")
user_input = st.text_area(
    "Tulis tentang bagaimana perasaanmu, apa yang sedang kamu pikirkan, atau hal lain yang ingin kamu ungkapkan",
    placeholder="Contoh: Aku merasa sangat lelah dan tidak semangat akhir-akhir ini...",
    height=120
)

# --- Process button
if st.button("Proses") and user_input.strip():
    status, message, kategori = classify_tweet(user_input)  # Receive kategori here

    # --- Display status based on the predicted category (kategori)
    if status == "Berisiko":
        st.markdown("### Status Deteksi")
        
        # Check the specific kategori value to render a corresponding color/status
        if kategori == 'Terindikasi':
            st.markdown("""
                <div style='text-align: center;'>
                    <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#9D7BFB 0% 65%, #E3DAFB 65% 100%); margin: auto;'></div>
                    <p style='font-weight: bold; color: #6C3FC5;'>Terindikasi</p>
                </div>
            """, unsafe_allow_html=True)
        elif kategori == 'Penderita':
            st.markdown("""
                <div style='text-align: center;'>
                    <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#F44336 0% 65%, #FFCDD2 65% 100%); margin: auto;'></div>
                    <p style='font-weight: bold; color: #D32F2F;'>Penderita</p>
                </div>
            """, unsafe_allow_html=True)
        elif kategori == 'Penyintas':
            st.markdown("""
                <div style='text-align: center;'>
                    <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#4CAF50 0% 65%, #C8E6C9 65% 100%); margin: auto;'></div>
                    <p style='font-weight: bold; color: #388E3C;'>Penyintas</p>
                </div>
            """, unsafe_allow_html=True)
        elif kategori == 'Selfdiagnosed':
            st.markdown("""
                <div style='text-align: center;'>
                    <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#FF9800 0% 65%, #FFECB3 65% 100%); margin: auto;'></div>
                    <p style='font-weight: bold; color: #F57C00;'>Selfdiagnosed</p>
                </div>
            """, unsafe_allow_html=True)

    # --- Message
    st.markdown("### Pesan untuk Kamu")
    st.write(message)
