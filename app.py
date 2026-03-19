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
# KAMUS SUBJEK SPM (DARI FILE ANDA)
# ============================================
SUBJECT_NAMES = {
    'BM': 'BAHASA MELAYU',
    'BI': 'BAHASA INGGERIS',
    'PI': 'PENDIDIKAN ISLAM',
    'PM': 'PENDIDIKAN MORAL',
    'SEJ': 'SEJARAH',
    'MAT': 'MATEMATIK',
    'SN': 'SAINS',
    'KSI': 'LITERATURE IN ENGLISH',
    'KI': 'KESUSASTERAAN INGGERIS',
    'KSM': 'KESUSASTERAAN MELAYU',
    'KMK': 'KESUSASTERAAN MELAYU KOMUNIKATIF',
    'GEO': 'GEOGRAFI',
    'BAT': 'BAHASA ARAB / BAHASA ARAB TINGGI',
    'LUK': 'PENDIDIKAN SENI VISUAL',
    'MUZ': 'PENDIDIKAN MUZIK',
    'MZK': 'MUZIK',
    'TRI': 'TARI',
    'TAT': 'TEATER',
    'LS': 'LUKISAN',
    'SPS': 'SEJARAH DAN PENGURUSAN SENI',
    'SH2': 'SENI HALUS 2D',
    'SH3': 'SENI HALUS 3D',
    'RBG': 'REKA BENTUK GRAFIK',
    'MK': 'MULTIMEDIA KREATIF',
    'RBK': 'REKA BENTUK KRAF',
    'RBI': 'REKA BENTUK INDUSTRI',
    'PSP': 'PRODUKSI SENI PERSEMBAHAN',
    'AMU': 'ALAT MUZIK UTAMA',
    'MKO': 'MUZIK KOMPUTER',
    'ADT': 'AURAL DAN TEORI',
    'TR': 'TARIAN',
    'KT': 'KAREOGRAFI TARI',
    'AT': 'APRESIASI TARI',
    'LN': 'LAKONAN',
    'PS': 'PENULISAN SKRIP',
    'SG': 'SINOGRAFI',
    'M-T': 'MATEMATIK TAMBAHAN',
    'SPT': 'SAINS PERTANIAN',
    'PGO': 'PENGAJIAN AGROTEKNOLOGI',
    'PT': 'PERTANIAN',
    'PGW': 'PENGAJIAN KEUSAHAWANAN',
    'PDG': 'PERDAGANGAN',
    'PAK': 'PRINSIP PERAKAUNAN',
    'EKA': 'EKONOMI ASAS',
    'ERT': 'EKONOMI RUMAH TANGGA',
    'LKJ': 'LUKISAN KEJURUTERAAN',
    'PJM': 'PENGAJIAN KEJURUTERAAN MEKANIKAL',
    'PJA': 'PENGAJIAN KEJURUTERAAN AWAM',
    'PJE': 'PENGAJIAN KEJURUTERAAN ELEKTRIK DAN ELEKTRONIK',
    'RKC': 'REKA CIPTA',
    'TKJ': 'TEKNOLOGI KEJURUTERAAN',
    'TEK': 'INFORMATION AND COMMUNICATION TECHNOLOGY',
    'PNG': 'PERNIAGAAN',
    'EKO': 'EKONOMI',
    'AKS': 'ASAS KELESTARIAN',
    'RT': 'SAINS RUMAH TANGGA',
    'SK': 'SAINS KOMPUTER',
    'GKT': 'GRAFIK KOMUNIKASI TEKNIKAL',
    'FIZ': 'FIZIK',
    'KIM': 'KIMIA',
    'BIO': 'BIOLOGI',
    'SNT': 'ADDITIONAL SCIENCE',
    'PSS': 'PENGETAHUAN SAINS SUKAN',
    'SS': 'SAINS SUKAN',
    'ASN': 'APPLIED SCIENCE',
    'TSI': 'TASAWWUR ISLAM',
    'PQS': 'PENDIDIKAN AL-QURAN DAN AL-SUNNAH',
    'PSI': "PENDIDIKAN SYARI'AH ISLAMIAH",
    'HQ': 'HIFZ AL QURAN',
    'MQ': 'MAHARAT AL QURAN',
    'TQS': 'TURATH AL-QURAN DAN AL-SUNNAH',
    'TDI': 'TURATH DIRASAT ISLAMIAH',
    'TBA': 'TURATH BAHASA ARAB',
    'UAD': 'USUL AL-DIN',
    'AS': 'AL-SYARIAH',
    'LAM': 'AL-LUGHAH AL-ARABIAH AL-MU\'ASIRAH',
    'MUI': 'MANAHIJ AL-\'ULUM AL-ISLAMIAH',
    'AWB': 'AL-ADAB WA AL-BALAGHAH',
    'BC': 'BAHASA CINA',
    'BT': 'BAHASA TAMIL',
    'EST': 'ENGLISH FOR SCIENCE AND TECHNOLOGY',
    'BIB': 'BAHASA IBAN',
    'BS': 'BAHASA SEMAI',
    'BAK': 'BAHASA ARAB KOMUNIKASI',
    'PDM': 'PEMBINAAN DOMESTIK',
    'MPR': 'MEMBUAT PERABOT',
    'KPD': 'KERJA PAIP DOMESTIK',
    'PND': 'PENDAWAIAN DOMESTIK',
    'KAG': 'KIMPALAN ARKA DAN GAS',
    'MAM': 'MENSERVIS AUTOMOBIL',
    'MMS': 'MENSERVIS MOTOSIKAL',
    'PPU': 'MENSERVIS PERALATAN PENYEJUKAN DAN PENYAMANAN UDARA',
    'MPE': 'MENSERVIS PERALATAN ELEKTRIK DOMESTIK',
    'PR': 'PEMBUATAN PERABOT',
    'RDJ': 'REKAAN DAN JAHITAN PAKAIAN',
    'TKP': 'KATERING DAN PENYAJIAN',
    'PGN': 'PEMPROSESAN MAKANAN',
    'MDR': 'PENJAGAAN MUKA DAN DANDANAN RAMBUT',
    'PKK': 'ASUHAN DAN PENDIDIKAN AWAL KANAK-KANAK',
    'GAG': 'GERONTOLOGI ASAS DAN PERKHIDMATAN GERIATRIK',
    'PRM': 'PENJAGAAN MUKA DAN PENGGAYAAN RAMBUT',
    'GDG': 'GERONTOLOGI ASAS DAN GERIATRIK',
    'LDN': 'LANDSKAP DAN NURSERI',
    'AHR': 'AKUAKULTUR DAN HAIWAN REKREASI',
    'TNM': 'TANAMAN MAKANAN',
    'SRT': 'SENI REKA TANDA',
    'HAD': 'HIASAN DALAMAN ASAS',
    'MUL': 'PRODUKSI MULTIMEDIA',
    'GRA': 'GRAFIK BERKOMPUTER',
    'PRT': 'PRODUKSI REKA TANDA',
    'HD': 'HIASAN DALAMAN',
    'RGD': 'REKA BENTUK GRAFIK DIGITAL',
    'BHB': 'BAHAN BINAAN',
    'TKB': 'TEKNOLOGI BINAAN',
    'PEE': 'PRINSIP ELEKTRIK DAN ELEKTRONIK',
    'AEE': 'APLIKASI ELEKTRIK DAN ELEKTRONIK',
    'PBK': 'PEMESINAN BERKOMPUTER',
    'ABM': 'AMALAN BENGKEL MEKANIKAL',
    'PYJ': 'PENYEJUKAN',
    'PYU': 'PENYAMANAN UDARA',
    'ATK': 'AUTOMOTIF KENDERAAN',
    'AED': 'AUTOMOTIF ELEKTRIK DAN DISEL',
    'KRK': 'KIMPALAN ARKA',
    'KGS': 'KIMPALAN GAS',
    'FOP': 'FUNDAMENTALS OF PROGRAMMING',
    'PDT': 'PROGRAMMING AND DEVELOPMENT TOOLS',
    'MPK': 'MEMBUAT PAKAIAN',
    'POP': 'POLA PAKAIAN',
    'MRP': 'ROTI DAN MAKANAN YIS',
    'PTS': 'PATISSERIE',
    'SOL': 'PERSOLEKAN',
    'DDR': 'DANDANAN RAMBUT',
    'PPK': 'PENGAJIAN PERKEMBANGAN KANAK-KANAK',
    'PKA': 'PERKHIDMATAN AWAL KANAK-KANAK',
    'PBT': 'PENYEDIAAN MASAKAN BARAT DAN TIMUR',
    'PMM': 'PENYAJIAN MAKANAN DAN MINUMAN',
    'AKP': 'APLIKASI KOMPUTER DALAM PERNIAGAAN',
    'PAP': 'PERAKAUNAN PERNIAGAAN',
    'TPP': 'TEKNOLOGI PEJABAT PERNIAGAAN',
    'TBM': 'TEKNOLOGI BENGKEL MESIN',
    'KMM': 'KERJA MENGGEGAS & MELARIK',
    'LGM': 'LUKISAN GEOMETRI & MESIN',
    'TBB': 'TEKNOLOGI BINAAN BANGUNAN',
    'KKB': 'KERJA KAYU & BATA',
    'LGB': 'LUKISAN GEOMETRI & BINAAN BANGUNAN',
    'TGR': 'TEKNOLOGI ELEKTRIK',
    'PKE': 'PEMASANGAN & KAWALAN ELEKTRIK',
    'LGR': 'LUKISAN GEOMETRI & ELEKTRIK',
    'TGN': 'TEKNOLOGI ELEKTRONIK',
    'RTV': 'MENSERVIS RADIO & TV',
    'LGN': 'LUKISAN GEOMETRI & ELEKTRONIK',
    'TKL': 'TEKNOLOGI KIMPALAN & FABRIKASI LOGAM',
    'KK': 'KERJA KIMPALAN',
    'LGF': 'LUKISAN GEOMETRI & FABRIKASI LOGAM',
    'TGA': 'TEKNOLOGI AUTOMOTIF',
    'SMK': 'MENSERVIS & MEMBAIKI KENDERAAN',
    'LGA': 'LUKISAN GEOMETRI & AUTOMOTIF',
    'TPU': 'TEKNOLOGI PENYEJUKAN & PENYAMANAN UDARA',
    'KPU': 'KERJA PENYEJUKAN & PENYAMANAN UDARA',
    'LGP': 'LUKISAN GEOMETRI & PENYAMANAN UDARA',
    'AMM': 'ASAS PEMEROSESAN MAKLUMAT',
    'TKR': 'TEKNOLOGI KATERING',
    'PPM': 'PENYEDIAAN & PERKHIDMATAN MAKANAN',
    'TRF': 'TEKNOLOGI REKAAN FESYEN & MEMBUAT PAKAIAN',
    'RFP': 'REKAAN FESYEN & MEMBUAT PAKAIAN',
    'TSK': 'TEKNOLOGI SENI KECANTIKAN',
    'PMR': 'PERSOLEKAN & MENDANDAN RAMBUT',
    'TBK': 'TEKNOLOGI BAKERI & KONFEKSIONERI',
    'AKK': 'ASUHAN KANAK-KANAK',
    'MMK': 'MENGASUH DAN MEMBIMBING KANAK-KANAK',
    'PTM': 'PENGELUARAN TANAMAN',
    'PTN': 'PENGELUARAN TERNAKAN',
    'HHL': 'HORTIKULTUR HIASAN DAN LANDSKAP',
    'JLG': 'KEJENTERAAN LADANG',
    'LDG': 'PENGURUSAN LADANG',
    'KSE': 'KESUSASTERAAN CINA',
    'KST': 'KESUSASTERAAN TAMIL',
    'BK': 'BIBLE KNOWLEDGE',
    'BF': 'BAHASA PERANCIS',
    'BPJ': 'BAHASA PUNJABI',
    'KVL': 'KOMUNIKASI VISUAL',
    'SNH': 'SENI HALUS',
    'RKB': 'REKABENTUK',
    'BKD': 'BAHASA KADAZAN/DUSUN',
    'PDR': 'PELANCONGAN DAN REKREASI',
    'AMP': 'ASAS KEMAHIRAN PELANCONGAN',
    'JPT': 'KEJENTERAAN PERTANIAN',
    'PHP': 'PEMPROSESAN HASIL PERTANIAN',
    'BG': 'BAHASA JERMAN',
    'BJ': 'BAHASA JEPUN',
    'ZZZ': 'MATA PELAJARAN SELAIN DI ATAS'
}

# ============================================
# FUNGSI GRADE KE NUMERIK
# ============================================
def grade_to_numeric(grade):
    if pd.isna(grade) or grade == 'NA' or grade == '':
        return 0
    mapping = {
        'A+': 95, 'A': 90, 'A-': 85,
        'B+': 80, 'B': 75, 'B-': 70,
        'C+': 65, 'C': 60, 'C-': 55,
        'D': 50,
        'E': 45,
        'F': 40,
        'G': 30
    }
    val = str(grade).strip().upper()
    return mapping.get(val, 0)

# ============================================
# FUNGSI CADANGAN PROGRAM (BERDASARKAN SYARAT SEBENAR)
# ============================================
def recommend_programs(row):
    recommendations = []
    reasons = []
    
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
    
    # SUBJEK ISLAM
    pi = grade_to_numeric(row.get('PI', 0))  # Pendidikan Islam
    pqs = grade_to_numeric(row.get('PQS', 0))  # Pendidikan Al-Quran & Al-Sunnah
    psi = grade_to_numeric(row.get('PSI', 0))  # Pendidikan Syariah Islamiah
    tsi = grade_to_numeric(row.get('TSI', 0))  # Tasawwur Islam
    hq = grade_to_numeric(row.get('HQ', 0))  # Hifz Al-Quran
    mq = grade_to_numeric(row.get('MQ', 0))  # Maharat Al-Quran
    arab = grade_to_numeric(row.get('BAT', 0))  # Bahasa Arab
    
    # Syarat wajib: Sejarah lulus
    if sejarah < 40:
        return [{"cluster": "Tidak Layak", "programs": [], "reasons": ["Sejarah gagal - tidak layak ke mana-mana program"]}]
    
    # ========== PROGRAM ISLAM ==========
    
    # ---------- DIPLOMA IN HALAL INDUSTRY + HALAL EXECUTIVE CERTIFICATION ----------
    # Syarat: Pendidikan Islam ≥ B (75) ATAU PQS/PSI ≥ B, BM ≥ C (60), BI ≥ C (60)
    halal_programs = []
    halal_reasons = []
    islam_score = max([pi, pqs, psi, tsi]) if max([pi, pqs, psi, tsi]) > 0 else 0
    
    if islam_score >= 75 and bm >= 60 and bi >= 60:
        halal_programs.append("Diploma in Halal Industry + Halal Executive Certification")
        halal_reasons.append(f"Pendidikan Islam ({row.get('PI', 'N/A')}) / PQS/PSI ≥ B, BM & BI ≥ C")
    elif islam_score >= 60 and bm >= 60:
        halal_programs.append("Diploma in Halal Industry (Pertimbangan Khas)")
        halal_reasons.append(f"Pendidikan Islam/PQS/PSI sederhana, perlu semakan tambahan")
    
    if halal_programs:
        recommendations.append({
            'cluster': 'Halal Management',
            'programs': halal_programs,
            'reasons': halal_reasons,
            'score': 'Layak' if islam_score >= 75 else 'Pertimbangan'
        })
    
    # ---------- DIPLOMA IN ISLAMIC FINANCE + ASSOCIATE QUALIFICATION ----------
    # Syarat: Math ≥ C (60), Pendidikan Islam ≥ C (60), BM ≥ C (60)
    islamic_finance_programs = []
    islamic_finance_reasons = []
    
    if math >= 60 and islam_score >= 60 and bm >= 60:
        islamic_finance_programs.append("Diploma in Islamic Finance + Associate Qualification")
        islamic_finance_reasons.append(f"Math ≥ C, Pendidikan Islam ≥ C, BM ≥ C")
    
    if islamic_finance_programs:
        recommendations.append({
            'cluster': 'Islamic Finance',
            'programs': islamic_finance_programs,
            'reasons': islamic_finance_reasons,
            'score': 'Layak'
        })
    
    # ========== PROGRAM KONVENSIONAL ==========
    
    # ---------- DIPLOMA IN ENGLISH COMMUNICATION ----------
    # Syarat: BI ≥ B (75), BM ≥ C (60), Sejarah lulus
    english_programs = []
    english_reasons = []
    if bi >= 75 and bm >= 60 and sejarah >= 40:
        english_programs.append("Diploma in English Communication")
        english_reasons.append(f"BI ({row.get('BI', 'N/A')}) ≥ B, BM ({row.get('BM', 'N/A')}) ≥ C")
    
    if english_programs:
        recommendations.append({
            'cluster': 'Language & Communication',
            'programs': english_programs,
            'reasons': english_reasons,
            'score': 'Layak'
        })
    
    # ---------- ASASI KEJURUTERAAN ----------
    # Syarat: Add Math ≥ B (75), (Fizik ≥ B ATAU Kimia ≥ B), BM & Math ≥ A- (85)
    eng_programs = []
    eng_reasons = []
    if add_math >= 75 and (fizik >= 75 or kim >= 75) and bm >= 85 and math >= 85:
        eng_programs.append("Asasi Kejuruteraan & Teknologi (UTM)")
        eng_programs.append("Asasi Kejuruteraan & Teknologi (UMP)")
        eng_reasons.append(f"Add Math ({row.get('M-T', 'N/A')}) ≥ B, Fizik/Kimia ≥ B, BM & Math ≥ A-")
    
    if eng_programs:
        recommendations.append({
            'cluster': 'Engineering & Technology',
            'programs': eng_programs,
            'reasons': eng_reasons,
            'score': 'Layak'
        })
    
    # ---------- DIPLOMA IN ACCOUNTING ----------
    # Syarat: ACC ≥ B (75), Math ≥ B (75)
    acc_programs = []
    acc_reasons = []
    if acc >= 75 and math >= 75:
        acc_programs.append("Diploma in Accounting")
        acc_programs.append("Diploma in Accounting + SAP")
        acc_reasons.append(f"ACC ({row.get('ACC', 'N/A')}) ≥ B, Math ({row.get('MAT', 'N/A')}) ≥ B")
    
    if acc_programs:
        recommendations.append({
            'cluster': 'Accounting & Finance',
            'programs': acc_programs,
            'reasons': acc_reasons,
            'score': 'Layak'
        })
    
    # ---------- DIPLOMA IN COMPUTER SCIENCE ----------
    # Syarat: Math ≥ B (75), BI ≥ B (75), BM ≥ C (60)
    cs_programs = []
    cs_reasons = []
    if math >= 75 and bi >= 75 and bm >= 60:
        cs_programs.append("Diploma in Computer Science")
        cs_reasons.append(f"Math ({row.get('MAT', 'N/A')}) ≥ B, BI ({row.get('BI', 'N/A')}) ≥ B")
    
    if cs_programs:
        recommendations.append({
            'cluster': 'Computer Science & IT',
            'programs': cs_programs,
            'reasons': cs_reasons,
            'score': 'Layak'
        })
    
    # ---------- ASASI SAINS ----------
    # Syarat: Salah satu Bio/Fizik/Kim ≥ B (75), BM & Math ≥ A- (85)
    science_programs = []
    science_reasons = []
    if (bio >= 75 or fizik >= 75 or kim >= 75) and bm >= 85 and math >= 85:
        science_programs.append("Asasi Sains")
        science_reasons.append(f"Bio/Fizik/Kim ≥ B, BM & Math ≥ A-")
    
    if science_programs:
        recommendations.append({
            'cluster': 'Science',
            'programs': science_programs,
            'reasons': science_reasons,
            'score': 'Layak'
        })
    
    # ---------- DIPLOMA IN BUSINESS STUDIES ----------
    # Syarat: Math ≥ C (60), BI ≥ C (60)
    biz_programs = []
    biz_reasons = []
    if math >= 60 and bi >= 60:
        biz_programs.append("Diploma in Business Studies")
        biz_programs.append("Diploma in International Business")
        biz_reasons.append(f"Math ({row.get('MAT', 'N/A')}) ≥ C, BI ({row.get('BI', 'N/A')}) ≥ C")
    
    if biz_programs:
        recommendations.append({
            'cluster': 'Business & Management',
            'programs': biz_programs,
            'reasons': biz_reasons,
            'score': 'Layak'
        })
    
    return recommendations if recommendations else [{"cluster": "Tiada", "programs": [], "reasons": ["Tiada program yang memenuhi syarat minimum"]}]

# ============================================
# FUNGSI UNTUK PAPAR SUBJEK
# ============================================
def display_subjects(row):
    # Cari semua kolum subjek (yang ada dalam SUBJECT_NAMES dan ada nilai)
    subject_cols = []
    for col in row.index:
        if col in SUBJECT_NAMES:
            val = row.get(col)
            if pd.notna(val) and val != 'NA' and val != '':
                subject_cols.append(col)
    
    # Susun ikut abjad
    subject_cols.sort()
    
    return subject_cols

# ============================================
# FUNGSI UNTUK SEMAK PADANAN DENGAN PILIHAN ASAL
# ============================================
def check_original_choices(row, recommendations):
    original_choices = {
        'PIL1': row.get('PIL1', ''),
        'PIL2': row.get('PIL2', ''),
        'PIL3': row.get('PIL3', '')
    }
    
    # Senarai program yang kita cadangkan
    recommended_programs = []
    for rec in recommendations:
        if rec['cluster'] not in ['Tiada', 'Tidak Layak']:
            recommended_programs.extend(rec['programs'])
    
    # Semak padanan
    matches = {}
    for key, prog in original_choices.items():
        if pd.notna(prog) and prog != '':
            # Cari dalam cadangan
            match_found = any(prog.lower() in rp.lower() for rp in recommended_programs)
            matches[key] = {
                'program': prog,
                'match': match_found
            }
    
    return matches, original_choices

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
            
            subject_cols = display_subjects(pelajar_terpilih)
            
            if subject_cols:
                # Bahagikan kepada 3 column
                cols = st.columns(3)
                for i, subj_code in enumerate(subject_cols):
                    with cols[i % 3]:
                        grade = pelajar_terpilih.get(subj_code, '')
                        st.markdown(f"**{SUBJECT_NAMES.get(subj_code, subj_code)}**")
                        st.info(grade)
            else:
                st.warning("Tiada data subjek SPM untuk pelajar ini.")
            
            # ============================================
            # PAPAR KEPUTUSAN CADANGAN
            # ============================================
            st.markdown("---")
            st.markdown("## 📊 **KEPUTUSAN CADANGAN PROGRAM**")
            
            # Dapatkan cadangan berdasarkan syarat sebenar
            recommendations = recommend_programs(pelajar_terpilih)
            
            # Semak padanan dengan pilihan asal
            matches, original_choices = check_original_choices(pelajar_terpilih, recommendations)
            
            # Paparan utama cadangan
            if recommendations and recommendations[0]['cluster'] != 'Tiada' and recommendations[0]['cluster'] != 'Tidak Layak':
                # Paparan dalam bentuk grid
                for rec in recommendations:
                    with st.expander(f"**{rec['cluster']}** ({rec['score']})"):
                        for prog in rec['programs']:
                            # Tandakan jika program ini dalam pilihan asal
                            in_original = False
                            for key, prog_info in matches.items():
                                if prog_info['match'] and prog.lower() in prog_info['program'].lower():
                                    in_original = True
                                    break
                            
                            if in_original:
                                st.write(f"✅ **{prog}** ⭐ (Dalam pilihan asal)")
                            else:
                                st.write(f"✅ {prog}")
                        
                        if rec['reasons']:
                            st.caption("📌 " + ", ".join(rec['reasons']))
            elif recommendations and recommendations[0]['cluster'] == 'Tidak Layak':
                st.error("❌ " + recommendations[0]['reasons'][0])
            else:
                st.warning("Tiada cadangan program yang memenuhi syarat minimum.")
            
            # ============================================
            # PAPAR PILIHAN PROGRAM ASAL
            # ============================================
            with st.expander("📋 Pilihan Program Asal Pelajar"):
                for key, prog in original_choices.items():
                    if pd.notna(prog) and prog != '':
                        match_status = "✓ Dalam cadangan" if matches[key]['match'] else "✗ Tiada dalam cadangan"
                        st.write(f"**{key}:** {prog} - {match_status}")
                
                if 'KURSUSJAYA' in pelajar_terpilih.index:
                    st.write(f"**Berjaya ditawarkan:** {pelajar_terpilih['KURSUSJAYA']}")
            
            # ============================================
            # PAPAR PREDIKSI MODEL (optional)
            # ============================================
            with st.expander("🔍 Analisis Model (untuk rujukan)"):
                # Sediakan features untuk model
                subject_cols_all = [col for col in pelajar_terpilih.index if col in SUBJECT_NAMES]
                
                all_features = {}
                
                # Demographic features
                all_features['JANTINA'] = 1 if pelajar_terpilih.get('JANTINA') == 'P' else 0
                all_features['LOKASI'] = 1 if pelajar_terpilih.get('LOKASI') == 'BANDAR' else 0
                all_features['ALIRAN'] = 1 if pelajar_terpilih.get('ALIRAN') == 'STEM' else 0
                all_features['PENDAPATAN'] = pelajar_terpilih.get('PENDAPATAN', 0)
                
                # Subject grades
                for subj in subject_cols_all:
                    all_features[subj] = grade_to_numeric(pelajar_terpilih.get(subj, 0))
                
                # Isi feature missing dengan 0
                for f in feature_names:
                    if f not in all_features:
                        all_features[f] = 0
                
                feature_df = pd.DataFrame([all_features])
                feature_df = feature_df[feature_names]
                
                prediction = model.predict(feature_df)[0]
                probability = model.predict_proba(feature_df)[0][1]
                
                st.write(f"**Prediksi model:** {'DITAWARKAN' if prediction == 1 else 'TIDAK DITAWARKAN'}")
                st.write(f"**Kebarangkalian:** {probability:.1%}")
        
        else:
            st.error("❌ Pelajar tidak dijumpai. Sila semak semula NOKP atau nama.")

# Footer
st.markdown("---")
st.markdown("💡 *Sistem ini adalah prototype untuk membantu pegawai MARA membuat keputusan berdasarkan syarat kelayakan program.*")
