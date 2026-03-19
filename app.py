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
# SENARAI PROGRAM (AKAN DIUPDATE)
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
        'syarat_islam': {'PI': 60, 'PQS': 60, 'PSI': 60}
    },
    {
        'name': 'Diploma in Islamic Finance + Associate Qualification',
        'cluster': 'Islamic Finance',
        'syarat': {'MAT': 60, 'BM': 60, 'SEJ': 40},
        'syarat_islam': {'PI': 50, 'PQS': 50, 'PSI': 50}
    },
    {
        'name': 'Asasi Kejuruteraan & Teknologi (UTM)',
        'cluster': 'Engineering',
        'syarat': {'M-T': 70, 'BM': 80, 'MAT': 80, 'SEJ': 40},
        'syarat_sains': {'FIZ': 70, 'KIM': 70}
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
        'syarat_sains': {'BIO': 70, 'FIZ': 70, 'KIM': 70}
    },
    {
        'name': 'Diploma in Business Studies',
        'cluster': 'Business',
        'syarat': {'MAT': 55, 'BI': 55, 'SEJ': 40}
    },
    {
        'name': 'Diploma in International Business',
        'cluster': 'Business',
        'syarat': {'MAT': 55, 'BI': 55, 'SEJ': 40}
    }
]

# ============================================
# FUNGSI SEMAK KELAYAKAN
# ============================================
def is_eligible(row, program, debug=False):
    results = []
    
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    if sejarah < 40:
        if debug: results.append(f"❌ Sejarah: {sejarah} < 40")
        return False, results
    
    syarat = program.get('syarat', {})
    for subj, min_nilai in syarat.items():
        if subj in ['BM', 'BI', 'MAT', 'M-T', 'ACC']:
            nilai = grade_to_numeric(row.get(subj, 0))
            if nilai < min_nilai:
                if debug: results.append(f"❌ {subj}: {nilai} < {min_nilai}")
                return False, results
            else:
                if debug: results.append(f"✅ {subj}: {nilai} ≥ {min_nilai}")
    
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
# FUNGSI HITUNG SKOR
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
