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

# Dapatkan feature names dari model
feature_names = model.feature_names_in_
expected_features = list(feature_names)

st.sidebar.write(f"✅ Model sedia (menjangka {len(expected_features)} features)")

# Sidebar untuk input
st.sidebar.header("📋 Maklumat Pelajar")

with st.sidebar.form("pelajar_form"):
    st.subheader("Demografi")
    jantina = st.selectbox("Jantina", ["L", "P"])
    lokasi = st.selectbox("Lokasi", ["BANDAR", "LUAR BANDAR"])
    aliran = st.selectbox("Aliran", ["STEM", "KEMANUSIAN & SASTERA IKHTISAS"])
    pendapatan = st.number_input("Pendapatan Ibu Bapa (RM)", min_value=0, value=5000, step=100)

    st.subheader("SPM (Gred)")
    
    bm = st.selectbox("Bahasa Melayu", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    bi = st.selectbox("Bahasa Inggeris", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    math = st.selectbox("Matematik", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    sejarah = st.selectbox("Sejarah", ["A", "A-", "B+", "B", "C+", "C", "D", "E"])
    
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

# Fungsi untuk buat semua features
def create_full_features(jantina, lokasi, aliran, pendapatan, bm, bi, math, sejarah, add_math, physics, chemistry, biology, accounting):
    
    # Base features (yang kita ada)
    base_data = {
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
    
    # Buat dataframe dengan semua features (set 0 untuk yang lain)
    full_data = {}
    for feature in expected_features:
        if feature in base_data:
            full_data[feature] = base_data[feature]
        else:
            full_data[feature] = 0  # isi dengan 0 untuk feature yang tak ada input
    
    return pd.DataFrame([full_data])

if submitted:
    with st.spinner("Memproses cadangan..."):
        # Buat dataframe dengan semua features
        input_df = create_full_features(
            jantina, lokasi, aliran, pendapatan,
            bm, bi, math, sejarah,
            add_math, physics, chemistry, biology, accounting
        )
        
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
            
            if prediction == 1:
                if grade_to_numeric(add_math) >= 75 and (grade_to_numeric(physics) >= 75 or grade_to_numeric(chemistry) >= 75):
                    st.info("🏗️ **Engineering & Technology**")
                    st.write("- Asasi Kejuruteraan & Teknologi (UTM)")
                    st.write("- Asasi Kejuruteraan & Teknologi (UMP)")
                
                if grade_to_numeric(accounting) >= 60:
                    st.info("💰 **Accounting & Finance**")
                    st.write("- Diploma in Accounting")
                    st.write("- Diploma in Accounting + SAP")
                
                if grade_to_numeric(bi) >= 75 and grade_to_numeric(bm) >= 75:
                    st.info("🗣️ **Language & Communication**")
                    st.write("- Diploma in English Communication")
            else:
                st.warning("Tiada cadangan program. Sila tingkatkan pencapaian akademik.")
        
        # Debug info (optional)
        with st.expander("🔍 Technical Details"):
            st.write("Input features:", input_df.to_dict())
            st.write(f"Expected features count: {len(expected_features)}")

# Footer
st.markdown("---")
st.markdown("💡 *Sistem ini adalah prototype untuk membantu pegawai MARA membuat keputusan.*")
