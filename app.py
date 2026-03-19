import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="MARA Program Recommendation",
    page_icon="🎓",
    layout="wide"
)

# Tajuk
st.title("🎓 MARA Program Recommendation System")
st.markdown("Sistem ini membantu mencadangkan program yang sesuai berdasarkan profil akademik dan demografi pelajar.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('mara_model.pkl')

model = load_model()

# Sidebar untuk navigasi
st.sidebar.image("https://www.mara.gov.my/wp-content/uploads/2021/12/logo-mara.png", width=200)
st.sidebar.header("📋 Maklumat Pelajar")

# Input pelajar
with st.sidebar.form("pelajar_form"):
    st.subheader("Demografi")
    jantina = st.selectbox("Jantina", ["L", "P"])
    lokasi = st.selectbox("Lokasi", ["BANDAR", "LUAR BANDAR"])
    aliran = st.selectbox("Aliran", ["STEM", "KEMANUSIAN & SASTERA IKHTISAS"])
    pendapatan = st.number_input("Pendapatan Ibu Bapa (RM)", min_value=0, value=5000, step=100)

    st.subheader("SPM (Gred)")
    
    # Subjek utama
    bm = st.selectbox("Bahasa Melayu", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    bi = st.selectbox("Bahasa Inggeris", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    math = st.selectbox("Matematik", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    sejarah = st.selectbox("Sejarah", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    
    st.subheader("Subjek Elektif")
    add_math = st.selectbox("Matematik Tambahan", ["A", "A-", "B+", "B", "C+", "C", "D", "E", "NA"])
    physics = st.selectbox("Fizik", ["A", "A-", "B+", "B", "C+", "C", "D", "E", "NA"])
    chemistry = st.selectbox("Kimia", ["A", "A-", "B+", "B", "C+", "C", "D", "E", "NA"])
    biology = st.selectbox("Biologi", ["A", "A-", "B+", "B", "C+", "C", "D", "E", "NA"])
    accounting = st.selectbox("Prinsip Perakaunan", ["A", "A-", "B+", "B", "C+", "C", "D", "E", "NA"])
    
    submitted = st.form_submit_button("🎯 Dapatkan Cadangan")

# Fungsi tukar gred ke numerik
def grade_to_numeric(grade):
    mapping = {
        'A': 90, 'A-': 90,
        'B+': 75, 'B': 75, 'B-': 75,
        'C+': 60, 'C': 60, 'C-': 60,
        'D': 50,
        'E': 40,
        'NA': 0
    }
    return mapping.get(grade, 0)

# Kalau submit form
if submitted:
    with st.spinner("Memproses cadangan..."):
        # Convert input ke format numerik
        input_data = {
            'JANTINA': 1 if jantina == 'P' else 0,
            'LOKASI': 1 if lokasi == 'BANDAR' else 0,
            'ALIRAN': 1 if aliran == 'STEM' else 0,
            'PENDAPATAN': pendapatan,
            'BM': grade_to_numeric(bm),
            'BI': grade_to_numeric(bi),
            'MAT': grade_to_numeric(math),
            'SEJ': grade_to_numeric(sejarah),
            'MT': grade_to_numeric(add_math),
            'FIZ': grade_to_numeric(physics),
            'KIM': grade_to_numeric(chemistry),
            'BIO': grade_to_numeric(biology),
            'ACC': grade_to_numeric(accounting)
        }
        
        # Buat dataframe
        input_df = pd.DataFrame([input_data])
        
        # Ramal
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # TUNJUK HASIL
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Keputusan")
            if prediction == 1:
                st.success(f"✅ **DITAWARKAN**")
                st.metric("Kebarangkalian", f"{probability:.1%}")
            else:
                st.error(f"❌ **TIDAK DITAWARKAN**")
                st.metric("Kebarangkalian", f"{probability:.1%}")
        
        with col2:
            st.subheader("🎯 Program Dicadangkan")
            
            # Cadangan program berdasarkan cluster
            if prediction == 1:
                if input_data['MT'] >= 75 and (input_data['FIZ'] >= 75 or input_data['KIM'] >= 75):
                    st.info("🏗️ **Engineering & Technology**")
                    st.write("- Asasi Kejuruteraan & Teknologi (UTM)")
                    st.write("- Asasi Kejuruteraan & Teknologi (UMP)")
                
                if input_data['ACC'] >= 60:
                    st.info("💰 **Accounting & Finance**")
                    st.write("- Diploma in Accounting")
                    st.write("- Diploma in Accounting + SAP")
                
                if input_data['BI'] >= 75 and input_data['BM'] >= 75:
                    st.info("🗣️ **Language & Communication**")
                    st.write("- Diploma in English Communication")
            else:
                st.warning("Tiada cadangan program. Sila tingkatkan pencapaian akademik.")

# Footer
st.markdown("---")
st.markdown("💡 *Sistem ini adalah prototype untuk membantu pegawai MARA membuat keputusan.*")
