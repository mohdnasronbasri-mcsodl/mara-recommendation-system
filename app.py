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
    'BAT': 'BAHASA ARAB'
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
        'D': 50, 'E': 45, 'F': 40,
        'G': 30
    }
    val = str(grade).strip().upper()
    if val in mapping:
        return mapping[val]
    for key in mapping:
        if val.startswith(key):
            return mapping[key]
    return 0

# ============================================
# FUNGSI BANTUAN
# ============================================
def count_subjects_ge(row, subjects, min_nilai):
    count = 0
    for subj in subjects:
        if subj in row.index:
            if grade_to_numeric(row[subj]) >= min_nilai:
                count += 1
    return count

def get_best_in_list(row, subject_list):
    best = 0
    for subj in subject_list:
        if subj in row.index:
            best = max(best, grade_to_numeric(row[subj]))
    return best

# ============================================
# SENARAI PROGRAM DENGAN SYARAT TERKINI
# ============================================
ALL_PROGRAMS = [
    # ========== GROUP 1 ==========
    {
        'name': 'Diploma in Integrated Logistics Management + CILT',
        'cluster': 'Logistics',
        'group': 1,
        'syarat': {
            'BM': 60,  # C
            'MAT': 40, # E
            'SEJ': 40, # E
            'BI_min': 40,  # E (tapi ada syarat khas)
            'other_count': 2,
            'other_min': 60,  # C
            'BI_syarat_khas': True  # E/D kena pra-diploma
        }
    },
    {
        'name': 'Diploma in Halal Industry + Halal Executive Certification',
        'cluster': 'Halal',
        'group': 1,
        'syarat': {
            'BM': 60, 'MAT': 40, 'SEJ': 40,
            'BI_min': 40,
            'other_count': 2,
            'other_min': 60,
            'BI_syarat_khas': True
        }
    },
    {
        'name': 'Diploma in Islamic Finance + Associate Qualification',
        'cluster': 'Islamic Finance',
        'group': 1,
        'syarat': {
            'BM': 60, 'MAT': 40, 'SEJ': 40,
            'BI_min': 40,
            'other_count': 2,
            'other_min': 60,
            'BI_syarat_khas': True
        }
    },
    {
        'name': 'Diploma in Business Studies',
        'cluster': 'Business',
        'group': 1,
        'syarat': {
            'BM': 60, 'MAT': 40, 'SEJ': 40,
            'BI_min': 40,
            'other_count': 2,
            'other_min': 60,
            'BI_syarat_khas': True
        }
    },
    {
        'name': 'Diploma in Business Information Technology',
        'cluster': 'Business IT',
        'group': 1,
        'syarat': {
            'BM': 60, 'MAT': 40, 'SEJ': 40,
            'BI_min': 40,
            'other_count': 2,
            'other_min': 60,
            'BI_syarat_khas': True
        }
    },
    {
        'name': 'Diploma in International Business',
        'cluster': 'Business',
        'group': 1,
        'syarat': {
            'BM': 60, 'MAT': 40, 'SEJ': 40,
            'BI_min': 40,
            'other_count': 2,
            'other_min': 60,
            'BI_syarat_khas': True
        }
    },
    {
        'name': 'Diploma in Creative Digital Media Production',
        'cluster': 'Creative Arts',
        'group': 1,
        'syarat': {
            'BM': 60, 'MAT': 40, 'SEJ': 40,
            'BI_min': 40,
            'other_count': 2,
            'other_min': 60,
            'BI_syarat_khas': True
        }
    },
    
    # ========== GROUP 2 ==========
    {
        'name': 'Diploma in Computer Science + SAS Certification',
        'cluster': 'Computer Science',
        'group': 2,
        'syarat': {
            'BM': 60,  # C
            'BI': 75,  # B
            'MAT': 75, # B
            'SEJ': 40, # E
            'other_count': 2,
            'other_min': 60  # C
        }
    },
    {
        'name': 'Diploma in Marketing + CPM Certification',
        'cluster': 'Marketing',
        'group': 2,
        'syarat': {
            'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40,
            'other_count': 2,
            'other_min': 60
        }
    },
    
    # ========== GROUP 3 ==========
    {
        'name': 'Diploma in Computer Science',
        'cluster': 'Computer Science',
        'group': 3,
        'syarat': {
            'BM': 60,  # C
            'MAT': 60, # C
            'SEJ': 40, # E
            'BI_min': 40, # E
            'other_count': 3,
            'other_min': 60  # C
        }
    },
    
    # ========== GROUP 4 ==========
    {
        'name': 'Diploma in English Communication + Translation Certificate',
        'cluster': 'Language',
        'group': 4,
        'syarat': {
            'BM': 60,  # C
            'BI': 75,  # B
            'MAT': 40, # E
            'SEJ': 40, # E
            'other_count': 1,
            'other_min': 60  # C
        }
    },
    
    # ========== GROUP 5 ==========
    {
        'name': 'Diploma in Accounting',
        'cluster': 'Accounting',
        'group': 5,
        'syarat': {
            'BM': 60,  # C
            'MAT': 60, # C
            'SEJ': 40, # E
            'BI_min': 40, # E
            'other_count': 1,
            'other_min': 60  # C
        }
    },
    
    # ========== GROUP 6 ==========
    {
        'name': 'Diploma in Accounting + SAP Certification',
        'cluster': 'Accounting',
        'group': 6,
        'syarat': {
            'BM': 60,  # C
            'BI': 75,  # B
            'MAT': 75, # B
            'SEJ': 40, # E
            'other_count': 1,
            'other_min': 60  # C
        }
    },
    
    # ========== GROUP 7 ==========
    {
        'name': 'Asasi Kejuruteraan & Teknologi (UTM)',
        'cluster': 'Engineering',
        'group': 7,
        'syarat': {
            'BM': 85,  # A-
            'MAT': 85, # A-
            'M-T': 75, # B
            'SEJ': 40, # E
            'sains_min': 75,  # Fizik atau Kimia ≥ B
            'other_count': 2,
            'other_min': 75  # B
        }
    },
    {
        'name': 'Asasi Kejuruteraan & Teknologi (UMP)',
        'cluster': 'Engineering',
        'group': 7,
        'syarat': {
            'BM': 85, 'MAT': 85, 'M-T': 75, 'SEJ': 40,
            'sains_min': 75,
            'other_count': 2,
            'other_min': 75
        }
    }
]

# ============================================
# FUNGSI SEMAK KELAYAKAN (VERSI BARU)
# ============================================
def is_eligible(row, program, debug=False):
    results = []
    syarat = program.get('syarat', {})
    
    # Semak Sejarah
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    if sejarah < syarat.get('SEJ', 0):
        if debug: results.append(f"❌ Sejarah: {sejarah} < {syarat.get('SEJ', 0)}")
        return False, results
    
    # Semak BM
    bm = grade_to_numeric(row.get('BM', 0))
    if bm < syarat.get('BM', 0):
        if debug: results.append(f"❌ BM: {bm} < {syarat.get('BM', 0)}")
        return False, results
    
    # Semak Math
    math = grade_to_numeric(row.get('MAT', 0))
    if math < syarat.get('MAT', 0):
        if debug: results.append(f"❌ MAT: {math} < {syarat.get('MAT', 0)}")
        return False, results
    
    # Semak BI (jika ada syarat khas)
    bi = grade_to_numeric(row.get('BI', 0))
    if 'BI' in syarat:
        if bi < syarat['BI']:
            if debug: results.append(f"❌ BI: {bi} < {syarat['BI']}")
            return False, results
    elif 'BI_min' in syarat:
        if bi < syarat['BI_min']:
            if debug: results.append(f"❌ BI: {bi} < {syarat['BI_min']}")
            return False, results
        # Syarat khas untuk BI E/D
        if syarat.get('BI_syarat_khas', False):
            if bi <= 50:  # E atau D
                if debug: results.append("⚠️ BI E/D - layak tapi perlu program pra-diploma")
    
    # Semak M-T (jika ada)
    if 'M-T' in syarat:
        mt = grade_to_numeric(row.get('M-T', 0))
        if mt < syarat['M-T']:
            if debug: results.append(f"❌ M-T: {mt} < {syarat['M-T']}")
            return False, results
    
    # Semak sains_min (Fizik atau Kimia)
    if 'sains_min' in syarat:
        fizik = grade_to_numeric(row.get('FIZ', 0))
        kim = grade_to_numeric(row.get('KIM', 0))
        if max(fizik, kim) < syarat['sains_min']:
            if debug: results.append(f"❌ Fizik/Kimia: {max(fizik, kim)} < {syarat['sains_min']}")
            return False, results
    
    # Semak other subjects
    if 'other_count' in syarat:
        # Senarai subjek lain (semua kecuali yang wajib)
        wajib = ['BM', 'BI', 'MAT', 'SEJ', 'M-T', 'FIZ', 'KIM']
        other_subjects = [col for col in row.index if col not in wajib and col in SUBJECT_NAMES]
        
        other_pass = 0
        for subj in other_subjects:
            if grade_to_numeric(row[subj]) >= syarat['other_min']:
                other_pass += 1
        
        if other_pass < syarat['other_count']:
            if debug: results.append(f"❌ Hanya {other_pass}/{syarat['other_count']} subjek lain ≥ {syarat['other_min']}")
            return False, results
    
    if debug:
        results.append("✅ Semua syarat dipenuhi")
    
    return True, results

# ============================================
# FUNGSI HITUNG SKOR KESESUAIAN
# ============================================
def hitung_skor(row, program):
    skor = 0
    total_bobot = 0
    
    # Demografi (10%)
    skor += 10
    total_bobot += 10
    
    # Pendapatan (10%)
    pendapatan = row.get('PENDAPATAN', 5000)
    if pendapatan < 3000:
        skor += 10
    elif pendapatan < 5000:
        skor += 8
    elif pendapatan < 8000:
        skor += 6
    else:
        skor += 4
    total_bobot += 10
    
    # Subjek (80%)
    subjek_count = 0
    subjek_total = 0
    
    # Semua subjek dalam syarat
    syarat = program.get('syarat', {})
    all_subjek = []
    for key in syarat:
        if key in ['BM', 'BI', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI', 'SEJ']:
            all_subjek.append(key)
    
    unique_subjek = list(set(all_subjek))
    
    for subj in unique_subjek:
        if subj in ['BM', 'BI', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI', 'SEJ']:
            nilai = grade_to_numeric(row.get(subj, 0))
            if nilai > 0:
                subjek_total += nilai
                subjek_count += 1
    
    if subjek_count > 0:
        purata = subjek_total / subjek_count
        skor += purata * 0.8
        total_bobot += 80
    
    if total_bobot > 0:
        return round((skor / total_bobot) * 100, 1)
    return 50

# ============================================
# SIDEBAR PENCARIAN
# ============================================
st.sidebar.header("🔍 Cari Pelajar")
cari_melalui = st.sidebar.radio("Cari melalui:", ["NOKP", "Nama"])

if cari_melalui == "NOKP":
    nokp_input = st.sidebar.text_input("Masukkan 12 digit NOKP", placeholder="030807060678")
else:
    nama_input = st.sidebar.text_input("Masukkan nama penuh", placeholder="NUR AELYA")

cari_button = st.sidebar.button("🔍 Cari Pelajar")

# ============================================
# MAIN AREA
# ============================================
if cari_button:
    with st.spinner("Mencari pelajar..."):
        if cari_melalui == "NOKP" and nokp_input:
            pelajar = df[df['NOKP'].astype(str).str.contains(nokp_input, na=False)]
        else:
            pelajar = df[df['NAMA'].str.contains(nama_input, case=False, na=False)]
        
        if len(pelajar) == 0:
            st.error("❌ Pelajar tidak dijumpai")
        else:
            row = pelajar.iloc[0]
            
            # DEBUG GRED
            with st.expander("🔍 Gred Penting"):
                penting = ['BM', 'BI', 'MAT', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI']
                data = {}
                for subj in penting:
                    if subj in row.index:
                        data[subj] = f"{row[subj]} ({grade_to_numeric(row[subj])})"
                st.json(data)
            
            # ========================================
            # LAYOUT 2 KOLOM
            # ========================================
            col_kiri, col_kanan = st.columns([1, 2])
            
            with col_kiri:
                st.markdown("### 👤 Profil Pelajar")
                
                # PROFIL DALAM TABLE
                profil_items = []
                profil_items.append(f"<tr><td><b>NOKP</b></td><td>{row['NOKP']}</td></tr>")
                profil_items.append(f"<tr><td><b>Nama</b></td><td>{row['NAMA']}</td></tr>")
                profil_items.append(f"<tr><td><b>Jantina</b></td><td>{'Perempuan' if row.get('JANTINA')=='P' else 'Lelaki'}</td></tr>")
                profil_items.append(f"<tr><td><b>Lokasi</b></td><td>{row.get('LOKASI', 'N/A')}</td></tr>")
                profil_items.append(f"<tr><td><b>Aliran</b></td><td>{row.get('ALIRAN', 'N/A')}</td></tr>")
                profil_items.append(f"<tr><td><b>Pendapatan</b></td><td>RM {row.get('PENDAPATAN', 0):,.0f}</td></tr>")
                
                st.markdown(f"""
                <div style='max-height: 300px; overflow-y: auto; margin-bottom: 20px;'>
                <table>
                    {''.join(profil_items)}
                </table>
                </div>
                """, unsafe_allow_html=True)
                
                # SUBJEK SPM
                st.markdown("### 📚 Subjek SPM")
                
                subject_items = []
                for code, name in SUBJECT_NAMES.items():
                    if code in row.index:
                        grade = row.get(code)
                        if pd.notna(grade) and grade != 'NA' and grade != '':
                            numeric = grade_to_numeric(grade)
                            subject_items.append(f"<tr><td>{name}</td><td><b>{grade}</b> ({numeric})</td></tr>")
                
                if subject_items:
                    items_to_show = ''.join(subject_items[:20])
                    st.markdown(f"""
                    <div style='font-size: 0.9em; max-height: 400px; overflow-y: auto'>
                    <table>
                        {items_to_show}
                    </table>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Tiada data subjek")
            
            with col_kanan:
                st.markdown("### 🎯 5 Program Terbaik")
                
                pilihan_asal = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        pilihan_asal.append(str(row[pil]).strip())
                
                program_scores = []
                
                for prog in ALL_PROGRAMS:
                    eligible, _ = is_eligible(row, prog, debug=False)
                    if eligible:
                        skor = hitung_skor(row, prog)
                        in_original = any(prog['name'].lower() in p.lower() for p in pilihan_asal)
                        if in_original:
                            skor = min(skor + 15, 100)
                        
                        program_scores.append({
                            'name': prog['name'],
                            'cluster': prog['cluster'],
                            'group': prog['group'],
                            'score': skor,
                            'in_original': in_original
                        })
                
                st.caption(f"📊 Program layak: {len(program_scores)} daripada {len(ALL_PROGRAMS)}")
                
                if len(program_scores) == 0:
                    st.warning("⚠️ Tiada program yang layak")
                    
                    with st.expander("🔍 Debug: Semua program"):
                        for prog in ALL_PROGRAMS:
                            eligible, reasons = is_eligible(row, prog, debug=True)
                            status = "✅ LAYAK" if eligible else "❌ TIDAK LAYAK"
                            st.write(f"**{prog['name']}** - {status}")
                            for r in reasons:
                                st.caption(r)
                else:
                    program_scores.sort(key=lambda x: x['score'], reverse=True)
                    top5 = program_scores[:5]
                    
                    for i, prog in enumerate(top5, 1):
                        if prog['score'] >= 80:
                            color = "#28a745"
                        elif prog['score'] >= 60:
                            color = "#ffc107"
                        else:
                            color = "#dc3545"
                        
                        star = " ⭐" if prog['in_original'] else ""
                        
                        st.markdown(f"""
                        <div style='margin-bottom: 10px; padding: 8px; border-left: 5px solid {color}; border-radius: 3px;'>
                            <span style='font-size: 1.1em'><b>{i}. {prog['name']}{star}</b></span><br>
                            <span style='font-size: 0.9em; color: {color}'><b>Kesesuaian: {prog['score']}%</b></span>
                            <span style='font-size: 0.8em; color: gray;'>Kumpulan {prog['group']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("📊 Perincian Skor"):
                        st.markdown("""
                        **Komponen skor:**
                        - Demografi: 10%
                        - Pendapatan: 10% (B40 > T20)
                        - Subjek: 80% (purata subjek berkaitan)
                        - Bonus: +15% jika dalam pilihan asal
                        
                        **Kelayakan:**
                        - ≥80%: Sangat sesuai
                        - 60-79%: Sederhana sesuai
                        - <60%: Kurang sesuai
                        """)
                    
                    with st.expander("📋 Pilihan Asal Pelajar"):
                        for i, p in enumerate(pilihan_asal, 1):
                            in_top5 = any(p.lower() in prog['name'].lower() for prog in top5)
                            status = "✓ Dalam cadangan" if in_top5 else "✗ Tiada dalam cadangan"
                            st.write(f"**PIL{i}:** {p} - {status}")
                        
                        if 'KURSUSJAYA' in row.index:
                            st.write(f"**Status sebenar:** {row['KURSUSJAYA']}")
