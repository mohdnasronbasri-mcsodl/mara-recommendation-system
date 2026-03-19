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
st.markdown("Sistem ini mencadangkan program berdasarkan data pelajar sebenar.")

# Load model dan data
@st.cache_resource
def load_model_and_data():
    model = joblib.load('mara_model.pkl')
    df = pd.read_csv('data_lengkap.csv')
    return model, df

model, df = load_model_and_data()
feature_names = model.feature_names_in_

# Sidebar
st.sidebar.header("🔍 Cari Pelajar")

# 1. Pilih cara cari
cari_melalui = st.sidebar.radio(
    "Cari melalui:",
    ["NOKP", "Nama"]
)

# 2. Input pencarian
if cari_melalui == "NOKP":
    nokp_input = st.sidebar.text_input("Masukkan 12 digit NOKP", placeholder="Contoh: 030807060678")
else:
    nama_input = st.sidebar.text_input("Masukkan nama penuh", placeholder="Contoh: NUR AELYA BINTI MOHAMAD REZALI")

cari_button = st.sidebar.button("🔍 Cari Pelajar")

# Main area
if cari_button:
    with st.spinner("Mencari pelajar..."):
        # Cari dalam dataframe
        if cari_melalui == "NOKP" and nokp_input:
            pelajar = df[df['NOKP'].astype(str).str.contains(nokp_input, na=False)]
        elif cari_melalui == "Nama" and nama_input:
            pelajar = df[df['NAMA'].str.contains(nama_input, case=False, na=False)]
        else:
            pelajar = pd.DataFrame()
        
        # Kalau jumpa
        if len(pelajar) > 0:
            st.success(f"✅ Dijumpai {len(pelajar)} rekod")
            
            # Pilih pelajar (kalau lebih dari 1)
            if len(pelajar) > 1:
                pilihan = st.selectbox(
                    "Pilih pelajar:",
                    pelajar.apply(lambda row: f"{row['NAMA']} ({row['NOKP']})", axis=1).tolist()
                )
                pelajar_terpilih = pelajar[pelajar.apply(lambda row: f"{row['NAMA']} ({row['NOKP']})", axis=1) == pilihan].iloc[0]
            else:
                pelajar_terpilih = pelajar.iloc[0]
            
            # Tunjukkan maklumat pelajar
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NOKP", pelajar_terpilih['NOKP'])
                st.metric("Jantina", "Perempuan" if pelajar_terpilih.get('JANTINA') == 'P' else "Lelaki")
            with col2:
                st.metric("Nama", pelajar_terpilih['NAMA'])
                st.metric("Lokasi", pelajar_terpilih.get('LOKASI', 'N/A'))
            with col3:
                st.metric("Aliran", pelajar_terpilih.get('ALIRAN', 'N/A'))
                st.metric("Pendapatan", f"RM {pelajar_terpilih.get('PENDAPATAN', 0):,.2f}")
            
            # Sediakan features untuk prediction
            # (Ambil semua kolum kecuali NAMA, NOKP, KURSUSJAYA)
            exclude_cols = ['NAMA', 'NOKP', 'KURSUSJAYA']
            feature_data = pelajar_terpilih.drop(labels=[col for col in exclude_cols if col in pelajar_terpilih.index])
            
            # Pastikan feature order sama dengan model
            feature_df = pd.DataFrame([feature_data])[list(feature_names)]
            
            # Predict
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]
            
            # TUNJUK HASIL
            st.markdown("---")
            st.subheader("📊 KEPUTUSAN CADANGAN")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if prediction == 1:
                    st.success(f"✅ **DITAWARKAN**")
                else:
                    st.error(f"❌ **TIDAK DITAWARKAN**")
                st.metric("Kebarangkalian", f"{probability:.1%}")
            
            with col_b:
                st.subheader("🎯 Program Dicadangkan")
                
                # Fungsi grade to numeric (guna semula)
                def grade_to_numeric(g):
                    if pd.isna(g) or g == 'NA':
                        return 0
                    mapping = {
                        'A': 90, 'A-': 90,
                        'B+': 75, 'B': 75, 'B-': 75,
                        'C+': 60, 'C': 60, 'C-': 60,
                        'D': 50,
                        'E': 40
                    }
                    return mapping.get(str(g).strip(), 0)
                
                # Dapatkan gred
                add_math = grade_to_numeric(pelajar_terpilih.get('M-T', 0))
                physics = grade_to_numeric(pelajar_terpilih.get('FIZ', 0))
                chemistry = grade_to_numeric(pelajar_terpilih.get('KIM', 0))
                accounting = grade_to_numeric(pelajar_terpilih.get('ACC', 0))
                bi = grade_to_numeric(pelajar_terpilih.get('BI', 0))
                bm = grade_to_numeric(pelajar_terpilih.get('BM', 0))
                math = grade_to_numeric(pelajar_terpilih.get('MAT', 0))
                
                recommendations = []
                
                # Engineering
                if add_math >= 75 and (physics >= 75 or chemistry >= 75):
                    recommendations.append("🏗️ **Engineering & Technology**\n- Asasi Kejuruteraan & Teknologi (UTM)\n- Asasi Kejuruteraan & Teknologi (UMP)")
                
                # Accounting
                if accounting >= 60:
                    recommendations.append("💰 **Accounting & Finance**\n- Diploma in Accounting\n- Diploma in Accounting + SAP")
                
                # Language
                if bi >= 75 and bm >= 75:
                    recommendations.append("🗣️ **Language & Communication**\n- Diploma in English Communication")
                
                # Business
                if math >= 60 and bi >= 60:
                    recommendations.append("📊 **Business & Management**\n- Diploma in Business Studies\n- Diploma in International Business")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.info("📚 Program umum - sila rujuk pegawai MARA")
            
            # Tunjukkan pilihan program asal
            with st.expander("📋 Pilihan Program Asal Pelajar"):
                st.write(f"Pilihan 1: {pelajar_terpilih.get('PIL1', 'N/A')}")
                st.write(f"Pilihan 2: {pelajar_terpilih.get('PIL2', 'N/A')}")
                st.write(f"Pilihan 3: {pelajar_terpilih.get('PIL3', 'N/A')}")
                if 'KURSUSJAYA' in pelajar_terpilih.index:
                    st.write(f"Status sebenar: {pelajar_terpilih['KURSUSJAYA']}")
        
        else:
            st.error("❌ Pelajar tidak dijumpai. Sila semak semula NOKP atau nama.")

# Footer
st.markdown("---")
st.markdown("💡 *Sistem ini adalah prototype untuk membantu pegawai MARA membuat keputusan.*")
