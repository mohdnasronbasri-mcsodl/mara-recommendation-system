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
st.markdown("Sistem ini mencadangkan program berdasarkan data pelajar sebenar MARA.")

# Load model dan data
@st.cache_resource
def load_model_and_data():
    model = joblib.load('mara_model.pkl')
    df = pd.read_csv('data_lengkap.csv')
    return model, df

model, df = load_model_and_data()
feature_names = list(model.feature_names_in_)

st.sidebar.success(f"✅ Model sedia: {len(feature_names)} features")

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

# ============================================
# FUNGSI GRADE KE NUMERIK
# ============================================
def grade_to_numeric(grade):
    if pd.isna(grade) or grade == 'NA' or grade == '':
        return 0
    mapping = {
        'A': 90, 'A-': 90, 'A+': 90,
        'B': 75, 'B+': 75, 'B-': 75,
        'C': 60, 'C+': 60, 'C-': 60,
        'D': 50,
        'E': 40
    }
    val = str(grade).strip().upper()
    return mapping.get(val, 0)

# ============================================
# FUNGSI CALCULATE CLUSTER SCORES (VERSI TEPAT)
# ============================================
def calculate_cluster_scores(row):
    scores = {}
    
    # ---------- ENGINEERING SCORE ----------
    add_math = grade_to_numeric(row.get('M-T', 0))
    fizik = grade_to_numeric(row.get('FIZ', 0))
    kim = grade_to_numeric(row.get('KIM', 0))
    math = grade_to_numeric(row.get('MAT', 0))
    bm = grade_to_numeric(row.get('BM', 0))
    
    # Kriteria: Add Math ≥ 75 DAN (Fizik ≥ 75 ATAU Kimia ≥ 75)
    if add_math >= 75 and (fizik >= 75 or kim >= 75):
        scores['Engineering_Score'] = 0.9
        scores['Engineering_Tier'] = 'High'
    elif add_math >= 60 and (fizik >= 60 or kim >= 60):
        scores['Engineering_Score'] = 0.6
        scores['Engineering_Tier'] = 'Medium'
    else:
        scores['Engineering_Score'] = 0.3
        scores['Engineering_Tier'] = 'Low'
    
    # ---------- ACCOUNTING SCORE ----------
    acc = grade_to_numeric(row.get('ACC', 0))
    math = grade_to_numeric(row.get('MAT', 0))
    
    if acc >= 75 and math >= 75:
        scores['Accounting_Score'] = 0.9
        scores['Accounting_Tier'] = 'High'
    elif acc >= 60 or math >= 60:
        scores['Accounting_Score'] = 0.6
        scores['Accounting_Tier'] = 'Medium'
    else:
        scores['Accounting_Score'] = 0.3
        scores['Accounting_Tier'] = 'Low'
    
    # ---------- LANGUAGE SCORE ----------
    bi = grade_to_numeric(row.get('BI', 0))
    bm = grade_to_numeric(row.get('BM', 0))
    arab = grade_to_numeric(row.get('BAT', 0))
    
    if bi >= 75 and bm >= 75:
        scores['Language_Score'] = 0.9
        scores['Language_Tier'] = 'High'
    elif bi >= 60 or bm >= 60:
        scores['Language_Score'] = 0.6
        scores['Language_Tier'] = 'Medium'
    else:
        scores['Language_Score'] = 0.3
        scores['Language_Tier'] = 'Low'
    
    # ---------- BUSINESS SCORE ----------
    commerce = grade_to_numeric(row.get('PT', 0))
    eko = grade_to_numeric(row.get('EKO', 0))
    math = grade_to_numeric(row.get('MAT', 0))
    
    business_subjects = [commerce, eko]
    best_business = max(business_subjects) if business_subjects else 0
    
    if best_business >= 75 and math >= 60:
        scores['Business_Score'] = 0.9
        scores['Business_Tier'] = 'High'
    elif best_business >= 60:
        scores['Business_Score'] = 0.6
        scores['Business_Tier'] = 'Medium'
    else:
        scores['Business_Score'] = 0.3
        scores['Business_Tier'] = 'Low'
    
    # ---------- COMPUTER SCORE ----------
    sk = grade_to_numeric(row.get('SK', 0))
    math = grade_to_numeric(row.get('MAT', 0))
    bi = grade_to_numeric(row.get('BI', 0))
    
    if sk >= 75 and math >= 60 and bi >= 60:
        scores['Computer_Score'] = 0.9
        scores['Computer_Tier'] = 'High'
    elif math >= 75 and bi >= 60:
        scores['Computer_Score'] = 0.6
        scores['Computer_Tier'] = 'Medium'
    else:
        scores['Computer_Score'] = 0.3
        scores['Computer_Tier'] = 'Low'
    
    # ---------- SCIENCE SCORE ----------
    bio = grade_to_numeric(row.get('BIO', 0))
    fizik = grade_to_numeric(row.get('FIZ', 0))
    kim = grade_to_numeric(row.get('KIM', 0))
    
    science_subjects = [b for b in [bio, fizik, kim] if b > 0]
    best_science = max(science_subjects) if science_subjects else 0
    
    if best_science >= 75:
        scores['Science_Score'] = 0.9
        scores['Science_Tier'] = 'High'
    elif best_science >= 60:
        scores['Science_Score'] = 0.6
        scores['Science_Tier'] = 'Medium'
    else:
        scores['Science_Score'] = 0.3
        scores['Science_Tier'] = 'Low'
    
    # ---------- HISTORY PASS ----------
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    scores['History_Pass'] = 1 if sejarah >= 40 else 0
    
    return scores

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
            
            # ============================================
            # FEATURE ENGINEERING
            # ============================================
            
            # 1. Dapatkan cluster scores
            cluster_scores = calculate_cluster_scores(pelajar_terpilih)
            
            # 2. Encode demographic
            demo_features = {
                'JANTINA': 1 if pelajar_terpilih.get('JANTINA') == 'P' else 0,
                'LOKASI': 1 if pelajar_terpilih.get('LOKASI') == 'BANDAR' else 0,
                'ALIRAN': 1 if pelajar_terpilih.get('ALIRAN') == 'STEM' else 0,
                'PENDAPATAN': pelajar_terpilih.get('PENDAPATAN', 0)
            }
            
            # 3. SPM grades (pilih subjek utama)
            spm_features = {}
            main_subjects = ['BM', 'BI', 'MAT', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK']
            for subj in main_subjects:
                spm_features[subj] = grade_to_numeric(pelajar_terpilih.get(subj, 0))
            
            # 4. Gabungkan semua features
            all_features = {}
            all_features.update(demo_features)
            all_features.update(spm_features)
            all_features.update(cluster_scores)
            
            # 5. Isi feature yang missing dengan 0
            for f in feature_names:
                if f not in all_features:
                    all_features[f] = 0
            
            # Buat dataframe
            feature_df = pd.DataFrame([all_features])
            feature_df = feature_df[feature_names]
            
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
                
                recommendations = []
                
                # ---------- CADANGAN PROGRAM ----------
                # Engineering
                if cluster_scores.get('Engineering_Score', 0) >= 0.7:
                    recommendations.append("🏗️ **Engineering & Technology**\n- Asasi Kejuruteraan & Teknologi (UTM)\n- Asasi Kejuruteraan & Teknologi (UMP)")
                
                # Accounting
                if cluster_scores.get('Accounting_Score', 0) >= 0.7:
                    recommendations.append("💰 **Accounting & Finance**\n- Diploma in Accounting\n- Diploma in Accounting + SAP")
                
                # Language
                if cluster_scores.get('Language_Score', 0) >= 0.7:
                    recommendations.append("🗣️ **Language & Communication**\n- Diploma in English Communication")
                
                # Business
                if cluster_scores.get('Business_Score', 0) >= 0.6:
                    recommendations.append("📊 **Business & Management**\n- Diploma in Business Studies\n- Diploma in International Business")
                
                # Computer
                if cluster_scores.get('Computer_Score', 0) >= 0.7:
                    recommendations.append("💻 **Computer Science & IT**\n- Diploma in Computer Science")
                
                # Science
                if cluster_scores.get('Science_Score', 0) >= 0.7:
                    recommendations.append("🔬 **Science**\n- Asasi Sains")
                
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
