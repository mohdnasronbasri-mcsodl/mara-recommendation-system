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
# KAMUS SUBJEK SPM (RINGKAS)
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
    # Handle kes macam 'A' je
    if val in mapping:
        return mapping[val]
    # Handle kes macam 'A+' etc
    for key in mapping:
        if val.startswith(key):
            return mapping[key]
    return 0

# ============================================
# SENARAI SEMUA PROGRAM (DENGAN SYARAT LEBIH FLEKSIBEL)
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
        'syarat': {'BM': 60, 'BI': 60, 'SEJ': 40},
        'syarat_islam': {'PI': 60, 'PQS': 60, 'PSI': 60}  # Salah satu ≥ 60
    },
    {
        'name': 'Diploma in Islamic Finance + Associate Qualification',
        'cluster': 'Islamic Finance',
        'syarat': {'MAT': 60, 'BM': 60, 'SEJ': 40},
        'syarat_islam': {'PI': 50, 'PQS': 50, 'PSI': 50}  # Salah satu ≥ 50
    },
    {
        'name': 'Asasi Kejuruteraan & Teknologi (UTM)',
        'cluster': 'Engineering',
        'syarat': {'M-T': 70, 'BM': 80, 'MAT': 80, 'SEJ': 40},  # Turunkan sikit
        'syarat_sains': {'FIZ': 70, 'KIM': 70}  # Salah satu ≥ 70
    },
    {
        'name': 'Asasi Kejuruteraan & Teknologi (UMP)',
        'cluster': 'Engineering',
        'syarat': {'M-T': 70, 'BM': 80, 'MAT': 80, 'SEJ': 40},
        'syarat_sains': {'FIZ': 70, 'KIM': 70}
    },
    {
        'name': 'Diploma in Accounting',
        'cluster': 'Accounting',
        'syarat': {'ACC': 70, 'MAT': 70, 'SEJ': 40}
    },
    {
        'name': 'Diploma in Accounting + SAP',
        'cluster': 'Accounting',
        'syarat': {'ACC': 70, 'MAT': 70, 'SEJ': 40}
    },
    {
        'name': 'Diploma in Computer Science',
        'cluster': 'Computer',
        'syarat': {'MAT': 70, 'BI': 70, 'BM': 60, 'SEJ': 40}
    },
    {
        'name': 'Asasi Sains',
        'cluster': 'Science',
        'syarat': {'BM': 80, 'MAT': 80, 'SEJ': 40},
        'syarat_sains': {'BIO': 70, 'FIZ': 70, 'KIM': 70}  # Salah satu ≥ 70
    },
    {
        'name': 'Diploma in Business Studies',
        'cluster': 'Business',
        'syarat': {'MAT': 55, 'BI': 55, 'SEJ': 40}  # Turunkan sikit
    },
    {
        'name': 'Diploma in International Business',
        'cluster': 'Business',
        'syarat': {'MAT': 55, 'BI': 55, 'SEJ': 40}
    }
]

# ============================================
# FUNGSI SEMAK KELAYAKAN (DENGAN DEBUG)
# ============================================
def is_eligible(row, program, debug=False):
    results = []
    
    # Semak Sejarah
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    if sejarah < 40:
        if debug: results.append(f"❌ Sejarah: {sejarah} < 40")
        return False, results
    
    # Semak syarat asas
    syarat = program.get('syarat', {})
    for subj, min_nilai in syarat.items():
        if subj in ['BM', 'BI', 'MAT', 'M-T', 'ACC']:
            nilai = grade_to_numeric(row.get(subj, 0))
            if nilai < min_nilai:
                if debug: results.append(f"❌ {subj}: {nilai} < {min_nilai}")
                return False, results
            else:
                if debug: results.append(f"✅ {subj}: {nilai} ≥ {min_nilai}")
    
    # Semak syarat sains (jika ada)
    if 'syarat_sains' in program:
        sains_ok = False
        for subj, min_nilai in program['syarat_sains'].items():
            nilai = grade_to_numeric(row.get(subj, 0))
            if nilai >= min_nilai:
                sains_ok = True
                if debug: results.append(f"✅ Sains ({subj}): {nilai} ≥ {min_nilai}")
                break
        if not sains_ok:
            if debug: results.append(f"❌ Tiada sains mencapai syarat")
            return False, results
    
    # Semak syarat islam (jika ada)
    if 'syarat_islam' in program:
        islam_ok = False
        for subj, min_nilai in program['syarat_islam'].items():
            nilai = grade_to_numeric(row.get(subj, 0))
            if nilai >= min_nilai:
                islam_ok = True
                if debug: results.append(f"✅ Islam ({subj}): {nilai} ≥ {min_nilai}")
                break
        if not islam_ok:
            if debug: results.append(f"❌ Tiada subjek islam mencapai syarat")
            return False, results
    
    if debug and not results:
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
    
    # Pendapatan (10%) - B40 dapat bonus
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
    all_syarat = []
    all_syarat.extend(program.get('syarat', {}).keys())
    if 'syarat_sains' in program:
        all_syarat.extend(program['syarat_sains'].keys())
    if 'syarat_islam' in program:
        all_syarat.extend(program['syarat_islam'].keys())
    
    unique_subjek = list(set(all_syarat))
    
    for subj in unique_subjek:
        if subj in ['BM', 'BI', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI', 'SEJ']:
            nilai = grade_to_numeric(row.get(subj, 0))
            if nilai > 0:
                subjek_total += nilai
                subjek_count += 1
    
    if subjek_count > 0:
        purata = subjek_total / subjek_count
        skor += purata * 0.8  # 80% dari purata subjek
        total_bobot += 80
    
    # Normalisasi
    if total_bobot > 0:
        return round((skor / total_bobot) * 100, 1)
    return 50  # default

# ============================================
# SIDEBAR PENCARIAN
# ============================================
st.sidebar.header("🔍 Cari Pelajar")
cari_melalui = st.sidebar.radio("Cari melalui:", ["NOKP", "Nama"])

if cari_melalui == "NOKP":
    nokp_input = st.sidebar.text_input("Masukkan 12 digit NOKP", placeholder="Contoh: 030807060678")
else:
    nama_input = st.sidebar.text_input("Masukkan nama penuh", placeholder="Contoh: NUR AELYA")

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
            # DEBUG: TUNJUK GRED PENTING
            # ============================================
            with st.expander("🔍 DEBUG: Gred Penting"):
                penting = ['BM', 'BI', 'MAT', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI']
                data = {}
                for subj in penting:
                    if subj in row.index:
                        data[subj] = f"{row[subj]} ({grade_to_numeric(row[subj])})"
                st.json(data)
            
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
                
                # SUBJEK SPM (RINGKAS)
                st.markdown("### 📚 Subjek SPM")
                
                subject_items = []
                for code, name in SUBJECT_NAMES.items():
                    if code in row.index:
                        grade = row.get(code)
                        if pd.notna(grade) and grade != 'NA' and grade != '':
                            numeric = grade_to_numeric(grade)
                            subject_items.append(f"<tr><td>{name}</td><td><b>{grade}</b> ({numeric})</td></tr>")
                
                if subject_items:
                    st.markdown(f"""
                    <div style='font-size: 0.8em; max-height: 400px; overflow-y: auto'>
                    <table>
                        {''.join(subject_items[:20])}  {/* Had 20 subjek */}
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
                    eligible, debug_results = is_eligible(row, prog, debug=False)
                    if eligible:
                        skor = hitung_skor(row, prog)
                        # Tambah bonus kalau dalam pilihan asal
                        in_original = any(prog['name'].lower() in p.lower() for p in pilihan_asal)
                        if in_original:
                            skor = min(skor + 15, 100)
                        
                        program_scores.append({
                            'name': prog['name'],
                            'cluster': prog['cluster'],
                            'score': skor,
                            'in_original': in_original,
                            'debug': debug_results
                        })
                    else:
                        # Untuk debug, kita boleh tengok kenapa tak layak
                        pass
                
                # DEBUG: TUNJUK BILANGAN PROGRAM LAYAK
                st.caption(f"📊 Program layak: {len(program_scores)} daripada {len(ALL_PROGRAMS)}")
                
                if len(program_scores) == 0:
                    st.warning("⚠️ Tiada program yang layak berdasarkan syarat semasa. Cuba semak subjek pelajar.")
                    
                    # Cadangan manual untuk debug
                    with st.expander("🔍 Debug: Semua program"):
                        for prog in ALL_PROGRAMS:
                            eligible, reasons = is_eligible(row, prog, debug=True)
                            status = "✅ LAYAK" if eligible else "❌ TIDAK LAYAK"
                            st.write(f"**{prog['name']}** - {status}")
                            for r in reasons:
                                st.caption(r)
                else:
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
                        - Subjek: 80% (purata subjek berkaitan)
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
