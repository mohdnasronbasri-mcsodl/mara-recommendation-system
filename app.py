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
feature_names = model.feature_names_in_
expected_features = list(feature_names)

# Sidebar
st.sidebar.header("📋 Maklumat Pelajar")

with st.sidebar.form("pelajar_form"):
    # TAMBAH ID PELAJAR
    st.subheader("🆔 Identiti")
    nama_pelajar = st.text_input("Nama Pelajar", value="", placeholder="Contoh: Ahmad Bin Ali")
    id_pelajar = st.text_input("No. ID / MyKad", value="", placeholder="Contoh: 050119050140")
    
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

# Fungsi untuk buat semua 299 features
def create_all_features(jantina, lokasi, aliran, pendapatan, bm, bi, math, sejarah, add_math, physics, chemistry, biology, accounting):
    
    # 1. Encode features asas
    base_features = {
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
    
    # 2. Generate all 299 features dengan nilai 0 untuk yang tak diisi
    full_data = {}
    for feature in expected_features:
        if feature in base_features:
            full_data[feature] = base_features[feature]
        else:
            full_data[feature] = 0  # default value
    
    return pd.DataFrame([full_data])

if submitted:
    with st.spinner("Memproses cadangan..."):
        # Validasi input
        if not nama_pelajar or not id_pelajar:
            st.error("❌ Sila isi Nama dan ID pelajar")
        else:
            # Buat features
            input_df = create_all_features(
                jantina, lokasi, aliran, pendapatan,
                bm, bi, math, sejarah,
                add_math, physics, chemistry, biology, accounting
            )
            
            # Ramal
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # TUNJUK IDENTITI PELAJAR
            st.subheader(f"👤 {nama_pelajar} ({id_pelajar})")
            
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
                    recommendations = []
                    
                    # Engineering
                    if grade_to_numeric(add_math) >= 75 and (grade_to_numeric(physics) >= 75 or grade_to_numeric(chemistry) >= 75):
                        recommendations.append("🏗️ **Engineering & Technology**\n- Asasi Kejuruteraan & Teknologi (UTM)\n- Asasi Kejuruteraan & Teknologi (UMP)")
                    
                    # Accounting
                    if grade_to_numeric(accounting) >= 60:
                        recommendations.append("💰 **Accounting & Finance**\n- Diploma in Accounting\n- Diploma in Accounting + SAP")
                    
                    # Language
                    if grade_to_numeric(bi) >= 75 and grade_to_numeric(bm) >= 75:
                        recommendations.append("🗣️ **Language & Communication**\n- Diploma in English Communication")
                    
                    # Business
                    if grade_to_numeric(math) >= 60 and grade_to_numeric(bi) >= 60:
                        recommendations.append("📊 **Business & Management**\n- Diploma in Business Studies\n- Diploma in International Business")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.info(rec)
                    else:
                        st.info("📚 Program umum - sila rujuk pegawai MARA")
                else:
                    st.warning("Tiada cadangan program khusus. Sila tingkatkan pencapaian akademik.")
            
            # Tunjukkan ringkasan input
            with st.expander("🔍 Ringkasan Input"):
                st.json({
                    "Nama": nama_pelajar,
                    "ID": id_pelajar,
                    "Jantina": jantina,
                    "Lokasi": lokasi,
                    "Aliran": aliran,
                    "Pendapatan": f"RM {pendapatan}",
                    "BM": bm,
                    "BI": bi,
                    "Math": math,
                    "Sejarah": sejarah,
                    "Add Math": add_math,
                    "Fizik": physics,
                    "Kimia": chemistry,
                    "Biologi": biology,
                    "ACC": accounting
                })

# Footer
st.markdown("---")
st.markdown("💡 *Sistem ini adalah prototype untuk membantu pegawai MARA membuat keputusan.*")
