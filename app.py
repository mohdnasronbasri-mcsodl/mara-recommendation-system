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

# Load model dan data
@st.cache_resource
def load_model_and_data():
    model = joblib.load('mara_model.pkl')
    df = pd.read_csv('data_lengkap.csv')
    return model, df

model, df = load_model_and_data()
feature_names = list(model.feature_names_in_)

# ============================================
# KAMUS SUBJEK SPM
# ============================================
SUBJECT_NAMES = {
    'BM': 'BAHASA MELAYU', 'BI': 'BAHASA INGGERIS', 'PI': 'PENDIDIKAN ISLAM',
    'PM': 'PENDIDIKAN MORAL', 'SEJ': 'SEJARAH', 'MAT': 'MATEMATIK',
    'M-T': 'MATEMATIK TAMBAHAN', 'FIZ': 'FIZIK', 'KIM': 'KIMIA',
    'BIO': 'BIOLOGI', 'ACC': 'PRINSIP PERAKAUNAN', 'PT': 'PERDAGANGAN',
    'EKO': 'EKONOMI', 'SK': 'SAINS KOMPUTER', 'PQS': 'PENDIDIKAN AL-QURAN DAN AL-SUNNAH',
    'PSI': "PENDIDIKAN SYARI'AH ISLAMIAH", 'TSI': 'TASAWWUR ISLAM',
    'BAT': 'BAHASA ARAB', 'PI': 'PENDIDIKAN ISLAM'
}

# ============================================
# FUNGSI GRADE KE NUMERIK
# ============================================
def grade_to_numeric(grade):
    if pd.isna(grade) or grade == 'NA' or grade == '':
        return 0
    mapping = {'A+':95,'A':90,'A-':85,'B+':80,'B':75,'B-':70,
               'C+':65,'C':60,'C-':55,'D':50,'E':45}
    val = str(grade).strip().upper()
    return mapping.get(val, 0)

# ============================================
# SENARAI SEMUA PROGRAM
# ============================================
ALL_PROGRAMS = [
    {
        'name': 'Diploma in English Communication + Translation Certificate',
        'cluster': 'Language',
        'syarat': {'BI': 75, 'BM': 60, 'SEJ': 40}
    },
    {
        'name': 'Diploma in Halal Industry + Halal Executive Certification',
        'cluster': 'Halal',
        'syarat': {'PI': 75, 'PQS': 75, 'PSI': 75, 'BM': 60, 'BI': 60, 'SEJ': 40},
        'syarat_alternatif': {'islam_score': 75, 'BM': 60, 'BI': 60}
    },
    {
        'name': 'Diploma in Islamic Finance + Associate Qualification',
        'cluster': 'Islamic Finance',
        'syarat': {'MAT': 60, 'PI': 60, 'BM': 60, 'SEJ': 40}
    },
    {
        'name': 'Asasi Kejuruteraan & Teknologi (UTM)',
        'cluster': 'Engineering',
        'syarat': {'M-T': 75, 'FIZ': 75, 'KIM': 75, 'BM': 85, 'MAT': 85, 'SEJ': 40},
        'syarat_alternatif': {'M-T': 75, 'sains_min': 75, 'BM': 85, 'MAT': 85}
    },
    {
        'name': 'Asasi Kejuruteraan & Teknologi (UMP)',
        'cluster': 'Engineering',
        'syarat': {'M-T': 75, 'FIZ': 75, 'KIM': 75, 'BM': 85, 'MAT': 85, 'SEJ': 40},
        'syarat_alternatif': {'M-T': 75, 'sains_min': 75, 'BM': 85, 'MAT': 85}
    },
    {
        'name': 'Diploma in Accounting',
        'cluster': 'Accounting',
        'syarat': {'ACC': 75, 'MAT': 75, 'SEJ': 40}
    },
    {
        'name': 'Diploma in Accounting + SAP',
        'cluster': 'Accounting',
        'syarat': {'ACC': 75, 'MAT': 75, 'SEJ': 40}
    },
    {
        'name': 'Diploma in Computer Science',
        'cluster': 'Computer',
        'syarat': {'MAT': 75, 'BI': 75, 'BM': 60, 'SEJ': 40}
    },
    {
        'name': 'Asasi Sains',
        'cluster': 'Science',
        'syarat': {'BIO': 75, 'FIZ': 75, 'KIM': 75, 'BM': 85, 'MAT': 85, 'SEJ': 40},
        'syarat_alternatif': {'sains_min': 75, 'BM': 85, 'MAT': 85}
    },
    {
        'name': 'Diploma in Business Studies',
        'cluster': 'Business',
        'syarat': {'MAT': 60, 'BI': 60, 'SEJ': 40}
    },
    {
        'name': 'Diploma in International Business',
        'cluster': 'Business',
        'syarat': {'MAT': 60, 'BI': 60, 'SEJ': 40}
    }
]

# ============================================
# FUNGSI HITUNG SKOR KESESUAIAN
# ============================================
def hitung_skor(row, program):
    skor = 0
    total_bobot = 0
    
    # Bobot: demografi 10%, pendapatan 10%, subjek 80%
    
    # Demografi (10%)
    if row.get('JANTINA') in ['L', 'P']:
        skor += 5
    if row.get('LOKASI') in ['BANDAR', 'LUAR BANDAR']:
        skor += 5
    total_bobot += 10
    
    # Pendapatan (10%) - lebih rendah lebih baik untuk B40
    pendapatan = row.get('PENDAPATAN', 5000)
    if pendapatan < 3000:
        skor += 10
    elif pendapatan < 5000:
        skor += 7
    elif pendapatan < 8000:
        skor += 5
    else:
        skor += 3
    total_bobot += 10
    
    # Subjek (80%)
    subjek_bobot = 0
    subjek_skor = 0
    
    for subj, min_nilai in program.get('syarat', {}).items():
        if subj in ['BM', 'BI', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI', 'SEJ']:
            nilai_pelajar = grade_to_numeric(row.get(subj, 0))
            if nilai_pelajar >= min_nilai:
                subjek_skor += min_nilai
                subjek_bobot += min_nilai
            else:
                # Kurang skor kalau tak capai
                subjek_skor += nilai_pelajar * 0.5
                subjek_bobot += min_nilai
    
    if subjek_bobot > 0:
        skor += (subjek_skor / subjek_bobot) * 80
        total_bobot += 80
    
    # Normalisasi ke 0-100
    if total_bobot > 0:
        return round((skor / total_bobot) * 100, 1)
    return 0

# ============================================
# FUNGSI SEMAK KELAYAKAN
# ============================================
def is_eligible(row, program):
    syarat = program.get('syarat', {})
    syarat_alt = program.get('syarat_alternatif', {})
    
    # Semak Sejarah dulu
    if grade_to_numeric(row.get('SEJ', 0)) < 40:
        return False
    
    # Guna syarat alternatif kalau ada
    if syarat_alt:
        if 'islam_score' in syarat_alt:
            islam_score = max([
                grade_to_numeric(row.get('PI', 0)),
                grade_to_numeric(row.get('PQS', 0)),
                grade_to_numeric(row.get('PSI', 0))
            ])
            if (islam_score >= syarat_alt['islam_score'] and
                grade_to_numeric(row.get('BM', 0)) >= syarat_alt.get('BM', 0) and
                grade_to_numeric(row.get('BI', 0)) >= syarat_alt.get('BI', 0)):
                return True
        
        if 'sains_min' in syarat_alt:
            sains_score = max([
                grade_to_numeric(row.get('FIZ', 0)),
                grade_to_numeric(row.get('KIM', 0)),
                grade_to_numeric(row.get('BIO', 0))
            ])
            if (sains_score >= syarat_alt['sains_min'] and
                grade_to_numeric(row.get('BM', 0)) >= syarat_alt.get('BM', 0) and
                grade_to_numeric(row.get('MAT', 0)) >= syarat_alt.get('MAT', 0)):
                return True
    
    # Guna syarat biasa
    for subj, min_nilai in syarat.items():
        if subj in ['BM', 'BI', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI']:
            if grade_to_numeric(row.get(subj, 0)) < min_nilai:
                return False
    
    return True

# ============================================
# SIDEBAR PENCARIAN
# ============================================
st.sidebar.header("🔍 Cari Pelajar")
cari_melalui = st.sidebar.radio("Cari melalui:", ["NOKP", "Nama"])

if cari_melalui == "NOKP":
    nokp_input = st.sidebar.text_input("Masukkan 12 digit NOKP")
else:
    nama_input = st.sidebar.text_input("Masukkan nama penuh")

cari_button = st.sidebar.button("🔍 Cari Pelajar")

# ============================================
# MAIN AREA
# ============================================
if cari_button:
    with st.spinner("Mencari pelajar..."):
        # Cari pelajar
        if cari_melalui == "NOKP" and nokp_input:
            pelajar = df[df['NOKP'].astype(str).str.contains(nokp_input, na=False)]
        else:
            pelajar = df[df['NAMA'].str.contains(nama_input, case=False, na=False)]
        
        if len(pelajar) == 0:
            st.error("❌ Pelajar tidak dijumpai")
        else:
            # Pilih pelajar pertama
            row = pelajar.iloc[0]
            
            # ============================================
            # LAYOUT 2 COLUMN
            # ============================================
            col_kiri, col_kanan = st.columns([1, 2])
            
            with col_kiri:
                # PROFIL PELAJAR (COMPACT)
                st.markdown("### 👤 Profil")
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.9em'>
                <b>NOKP:</b> {row['NOKP']}<br>
                <b>Nama:</b> {row['NAMA']}<br>
                <b>Jantina:</b> {'Perempuan' if row.get('JANTINA')=='P' else 'Lelaki'}<br>
                <b>Lokasi:</b> {row.get('LOKASI', 'N/A')}<br>
                <b>Aliran:</b> {row.get('ALIRAN', 'N/A')}<br>
                <b>Pendapatan:</b> RM {row.get('PENDAPATAN', 0):,.0f}
                </div>
                """, unsafe_allow_html=True)
                
                # SUBJEK SPM (COMPACT, FONT KECIL)
                st.markdown("### 📚 Subjek SPM")
                
                # Cari semua subjek yang ada nilai
                subject_items = []
                for code, name in SUBJECT_NAMES.items():
                    if code in row.index:
                        grade = row.get(code)
                        if pd.notna(grade) and grade != 'NA' and grade != '':
                            subject_items.append(f"<tr><td>{name}</td><td><b>{grade}</b></td></tr>")
                
                # Papar dalam jadual kecil
                if subject_items:
                    st.markdown(f"""
                    <div style='font-size: 0.8em; max-height: 400px; overflow-y: auto'>
                    <table>
                        {''.join(subject_items)}
                    </table>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Tiada data subjek")
            
            with col_kanan:
                # ========================================
                # CADANGAN 5 PROGRAM TERBAIK
                # ========================================
                st.markdown("### 🎯 5 Program Terbaik")
                
                # Dapatkan pilihan asal pelajar
                pilihan_asal = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        pilihan_asal.append(str(row[pil]).strip())
                
                # Kira skor untuk semua program
                program_scores = []
                for prog in ALL_PROGRAMS:
                    if is_eligible(row, prog):
                        skor = hitung_skor(row, prog)
                        # Tambah bonus kalau dalam pilihan asal
                        in_original = any(prog['name'].lower() in p.lower() for p in pilihan_asal)
                        if in_original:
                            skor = min(skor + 15, 100)  # Bonus 15%
                        program_scores.append({
                            'name': prog['name'],
                            'cluster': prog['cluster'],
                            'score': skor,
                            'in_original': in_original
                        })
                
                # Susun ikut skor tertinggi
                program_scores.sort(key=lambda x: x['score'], reverse=True)
                
                # Ambil 5 terbaik
                top5 = program_scores[:5]
                
                # Papar dalam format senarai
                for i, prog in enumerate(top5, 1):
                    # Tentukan warna berdasarkan skor
                    if prog['score'] >= 80:
                        color = "#28a745"  # Hijau
                    elif prog['score'] >= 60:
                        color = "#ffc107"  # Kuning
                    else:
                        color = "#dc3545"  # Merah
                    
                    # Tambah ⭐ kalau dalam pilihan asal
                    star = " ⭐" if prog['in_original'] else ""
                    
                    st.markdown(f"""
                    <div style='margin-bottom: 10px; padding: 8px; border-left: 5px solid {color}; background-color: #f8f9fa; border-radius: 3px;'>
                        <span style='font-size: 1.1em'><b>{i}. {prog['name']}{star}</b></span><br>
                        <span style='font-size: 0.9em; color: {color}'><b>Kesesuaian: {prog['score']}%</b></span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Info tambahan
                with st.expander("📊 Perincian Skor"):
                    st.markdown("""
                    **Komponen skor:**
                    - Demografi: 10%
                    - Pendapatan: 10% (B40 > T20)
                    - Subjek: 80% (berdasarkan syarat program)
                    - Bonus: +15% jika dalam pilihan asal
                    
                    **Kelayakan:**
                    - ≥80%: Sangat sesuai
                    - 60-79%: Sederhana sesuai
                    - <60%: Kurang sesuai
                    """)
                
                # Papar pilihan asal untuk rujukan
                with st.expander("📋 Pilihan Asal Pelajar"):
                    for i, p in enumerate(pilihan_asal, 1):
                        # Cek sama ada dalam top5
                        in_top5 = any(p.lower() in prog['name'].lower() for prog in top5)
                        status = "✓ Dalam cadangan" if in_top5 else "✗ Tiada dalam cadangan"
                        st.write(f"**PIL{i}:** {p} - {status}")
                    
                    if 'KURSUSJAYA' in row.index:
                        st.write(f"**Status sebenar:** {row['KURSUSJAYA']}")
