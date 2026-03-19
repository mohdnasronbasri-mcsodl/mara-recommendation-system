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
# KAMUS SUBJEK SPM
# ============================================
SUBJECT_NAMES = {
    'BM': 'Bahasa Melayu',
    'BI': 'Bahasa Inggeris',
    'SEJ': 'Sejarah',
    'PI': 'Pendidikan Islam',
    'PM': 'Pendidikan Moral',
    'MAT': 'Matematik',
    'SNS': 'Sains',
    'KI': 'Kesusasteraan Inggeris',
    'LUK': 'Pendidikan Seni Visual',
    'KMK': 'Kesusasteraan Melayu',
    'GEO': 'Geografi',
    'BAT': 'Bahasa Arab',
    'M-T': 'Matematik Tambahan',
    'PT': 'Perdagangan',
    'PGW': 'Pengajian Keusahawanan',
    'ACC': 'Prinsip Perakaunan',
    'LKJ': 'LKJ',
    'PJM': 'PJM',
    'PJA': 'PJA',
    'PJE': 'PJE',
    'RKC': 'RKC',
    'PNG': 'PNG',
    'EKO': 'Ekonomi',
    'AKS': 'AKS',
    'RT': 'Sains Rumah Tangga',
    'SK': 'Sains Komputer',
    'GKT': 'Grafik Komunikasi Teknikal',
    'FIZ': 'Fizik',
    'KIM': 'Kimia',
    'BIO': 'Biologi',
    'SNT': 'Sains Tambahan',
    'SS': 'Sains Sukan',
    'TSI': 'Tasawwur Islam',
    'PQS': 'Pendidikan Al-Quran & Al-Sunnah',
    'PSI': 'Pendidikan Syariah Islamiah',
    'HQ': 'Hifz Al-Quran',
    'MQ': 'Maharat Al-Quran',
    'UAD': 'UAD',
    'AS': 'Asas Sastera',
    'LAM': 'Bahasa Arab Lanjutan',
    'MUI': 'MUI',
    'AWB': 'AWB',
    'BC': 'Bahasa Cina',
    'PGN': 'PGN',
    'LDN': 'LDN',
    'MUL': 'MUL',
    'PRT': 'PRT',
    'HD': 'HD',
    'RGD': 'RGD',
    'BK': 'BK'
}

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
# FUNGSI CADANGAN PROGRAM (BERDASARKAN SUBJEK)
# ============================================
def recommend_programs(row):
    recommendations = []
    
    # Dapatkan gred dalam numerik
    add_math = grade_to_numeric(row.get('M-T', 0))
    fizik = grade_to_numeric(row.get('FIZ', 0))
    kim = grade_to_numeric(row.get('KIM', 0))
    bio = grade_to_numeric(row.get('BIO', 0))
    math = grade_to_numeric(row.get('MAT', 0))
    bm = grade_to_numeric(row.get('BM', 0))
    bi = grade_to_numeric(row.get('BI', 0))
    acc = grade_to_numeric(row.get('ACC', 0))
    sk = grade_to_numeric(row.get('SK', 0))
    commerce = grade_to_numeric(row.get('PT', 0))
    eko = grade_to_numeric(row.get('EKO', 0))
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    
    # Syarat wajib: Sejarah lulus
    if sejarah < 40:
        return ["❌ **Tidak layak** - Sejarah gagal"]
    
    # ---------- Engineering ----------
    if add_math >= 75 and (fizik >= 75 or kim >= 75):
        recommendations.append({
            'cluster': 'Engineering & Technology',
            'programs': [
                'Asasi Kejuruteraan & Teknologi (UTM)',
                'Asasi Kejuruteraan & Teknologi (UMP)'
            ],
            'score': 'Tinggi'
        })
    elif add_math >= 60 and (fizik >= 60 or kim >= 60):
        recommendations.append({
            'cluster': 'Engineering & Technology',
            'programs': [
                'Asasi Kejuruteraan & Teknologi (UTM)',
                'Asasi Kejuruteraan & Teknologi (UMP)'
            ],
            'score': 'Sederhana'
        })
    
    # ---------- Accounting ----------
    if acc >= 75 and math >= 75:
        recommendations.append({
            'cluster': 'Accounting & Finance',
            'programs': [
                'Diploma in Accounting',
                'Diploma in Accounting + SAP'
            ],
            'score': 'Tinggi'
        })
    elif acc >= 60 or math >= 60:
        recommendations.append({
            'cluster': 'Accounting & Finance',
            'programs': [
                'Diploma in Accounting',
                'Diploma in Islamic Finance'
            ],
            'score': 'Sederhana'
        })
    
    # ---------- Language ----------
    if bi >= 75 and bm >= 75:
        recommendations.append({
            'cluster': 'Language & Communication',
            'programs': [
                'Diploma in English Communication'
            ],
            'score': 'Tinggi'
        })
    elif bi >= 60 or bm >= 60:
        recommendations.append({
            'cluster': 'Language & Communication',
            'programs': [
                'Diploma in English Communication'
            ],
            'score': 'Sederhana'
        })
    
    # ---------- Business ----------
    business_score = max([commerce, eko]) if max([commerce, eko]) > 0 else 0
    if business_score >= 75 and math >= 60:
        recommendations.append({
            'cluster': 'Business & Management',
            'programs': [
                'Diploma in Business Studies',
                'Diploma in International Business'
            ],
            'score': 'Tinggi'
        })
    elif business_score >= 60:
        recommendations.append({
            'cluster': 'Business & Management',
            'programs': [
                'Diploma in Business Studies',
                'Diploma in Marketing'
            ],
            'score': 'Sederhana'
        })
    
    # ---------- Computer Science ----------
    if sk >= 75 and math >= 60 and bi >= 60:
        recommendations.append({
            'cluster': 'Computer Science & IT',
            'programs': [
                'Diploma in Computer Science'
            ],
            'score': 'Tinggi'
        })
    elif math >= 75 and bi >= 60:
        recommendations.append({
            'cluster': 'Computer Science & IT',
            'programs': [
                'Diploma in Computer Science'
            ],
            'score': 'Sederhana'
        })
    
    # ---------- Science ----------
    science_score = max([bio, fizik, kim]) if max([bio, fizik, kim]) > 0 else 0
    if science_score >= 75:
        recommendations.append({
            'cluster': 'Science',
            'programs': [
                'Asasi Sains'
            ],
            'score': 'Tinggi'
        })
    elif science_score >= 60:
        recommendations.append({
            'cluster': 'Science',
            'programs': [
                'Asasi Sains'
            ],
            'score': 'Sederhana'
        })
    
    return recommendations if recommendations else [{'cluster': 'Tiada', 'programs': ['Tiada cadangan khusus'], 'score': 'Rendah'}]

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
            
            # ============================================
            # PAPAR PROFIL PELAJAR
            # ============================================
            st.markdown("## 👤 **PROFIL PELAJAR**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**NOKP**")
                st.info(pelajar_terpilih['NOKP'])
                st.markdown(f"**Jantina**")
                st.info("Perempuan" if pelajar_terpilih.get('JANTINA') == 'P' else "Lelaki")
            with col2:
                st.markdown(f"**Nama**")
                st.info(pelajar_terpilih['NAMA'])
                st.markdown(f"**Lokasi**")
                st.info(pelajar_terpilih.get('LOKASI', 'N/A'))
            with col3:
                st.markdown(f"**Aliran**")
                st.info(pelajar_terpilih.get('ALIRAN', 'N/A'))
                st.markdown(f"**Pendapatan**")
                st.info(f"RM {pelajar_terpilih.get('PENDAPATAN', 0):,.2f}")
            
            # ============================================
            # PAPAR SUBJEK SPM
            # ============================================
            st.markdown("## 📚 **SUBJEK SPM**")
            
            # Cari semua kolum subjek (yang ada dalam SUBJECT_NAMES)
            subject_cols = [col for col in pelajar_terpilih.index if col in SUBJECT_NAMES]
            
            # Bahagikan kepada 3 column
            cols = st.columns(3)
            for i, subj_code in enumerate(subject_cols):
                with cols[i % 3]:
                    grade = pelajar_terpilih.get(subj_code, '')
                    if pd.notna(grade) and grade != 'NA' and grade != '':
                        st.markdown(f"**{SUBJECT_NAMES[subj_code]}**")
                        st.info(grade)
            
            # ============================================
            # FEATURE ENGINEERING UNTUK MODEL
            # ============================================
            
            # 1. Dapatkan cluster scores (guna fungsi sama)
            cluster_scores = {}  # Kita tak guna untuk paperan, tapi untuk model
            for rec in recommend_programs(pelajar_terpilih):
                if rec['cluster'] != 'Tiada':
                    cluster_scores[f"{rec['cluster']}_Score"] = 0.9 if rec['score'] == 'Tinggi' else 0.6
            
            # 2. Encode demographic
            demo_features = {
                'JANTINA': 1 if pelajar_terpilih.get('JANTINA') == 'P' else 0,
                'LOKASI': 1 if pelajar_terpilih.get('LOKASI') == 'BANDAR' else 0,
                'ALIRAN': 1 if pelajar_terpilih.get('ALIRAN') == 'STEM' else 0,
                'PENDAPATAN': pelajar_terpilih.get('PENDAPATAN', 0)
            }
            
            # 3. SPM grades (semua subjek)
            spm_features = {}
            for subj in subject_cols:
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
            
            # ============================================
            # PAPAR KEPUTUSAN
            # ============================================
            st.markdown("---")
            st.markdown("## 📊 **KEPUTUSAN CADANGAN**")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### 📈 Status Tawaran")
                if prediction == 1:
                    st.success(f"✅ **DITAWARKAN**")
                else:
                    st.error(f"❌ **TIDAK DITAWARKAN**")
                st.metric("Kebarangkalian", f"{probability:.1%}")
            
            with col_b:
                st.markdown("### 🎯 Program Dicadangkan")
                
                # Dapatkan cadangan berdasarkan subjek
                recommendations = recommend_programs(pelajar_terpilih)
                
                if recommendations and recommendations[0]['cluster'] != 'Tiada':
                    for rec in recommendations:
                        with st.expander(f"**{rec['cluster']}** (Kesesuaian: {rec['score']})"):
                            for prog in rec['programs']:
                                st.write(f"- {prog}")
                else:
                    st.warning("Tiada cadangan program khusus berdasarkan subjek yang diambil.")
            
            # ============================================
            # PAPAR PILIHAN PROGRAM ASAL
            # ============================================
            with st.expander("📋 Pilihan Program Asal Pelajar"):
                st.write(f"**Pilihan 1:** {pelajar_terpilih.get('PIL1', 'N/A')}")
                st.write(f"**Pilihan 2:** {pelajar_terpilih.get('PIL2', 'N/A')}")
                st.write(f"**Pilihan 3:** {pelajar_terpilih.get('PIL3', 'N/A')}")
                if 'KURSUSJAYA' in pelajar_terpilih.index:
                    st.write(f"**Status sebenar:** {pelajar_terpilih['KURSUSJAYA']}")
        
        else:
            st.error("❌ Pelajar tidak dijumpai. Sila semak semula NOKP atau nama.")

# Footer
st.markdown("---")
st.markdown("💡 *Sistem ini adalah prototype untuk membantu pegawai MARA membuat keputusan.*")
