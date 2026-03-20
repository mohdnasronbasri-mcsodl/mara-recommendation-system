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
# FUNGSI SEMAK KELAYAKAN (VERSI BETUL)
# ============================================
def is_eligible(row, program):
    syarat = program.get('syarat', {})
    
    # Semak Sejarah
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    if sejarah < syarat.get('SEJ', 40):
        return False
    
    # Semak BM
    bm = grade_to_numeric(row.get('BM', 0))
    if bm < syarat.get('BM', 60):
        return False
    
    # Semak Math
    math = grade_to_numeric(row.get('MAT', 0))
    if math < syarat.get('MAT', 0):
        return False
    
    # Semak BI
    bi = grade_to_numeric(row.get('BI', 0))
    if 'BI' in syarat:
        if bi < syarat['BI']:
            return False
    elif 'BI_min' in syarat:
        if bi < syarat['BI_min']:
            return False
    
    # Semak M-T (jika ada)
    if 'M-T' in syarat:
        mt = grade_to_numeric(row.get('M-T', 0))
        if mt < syarat['M-T']:
            return False
    
    # Semak sains_min (Fizik atau Kimia)
    if 'sains_min' in syarat:
        fizik = grade_to_numeric(row.get('FIZ', 0))
        kim = grade_to_numeric(row.get('KIM', 0))
        if max(fizik, kim) < syarat['sains_min']:
            return False
    
    # Semak other subjects
    if 'other_count' in syarat:
        wajib = ['BM', 'BI', 'MAT', 'SEJ', 'M-T', 'FIZ', 'KIM']
        other_subjects = [col for col in row.index if col not in wajib and col in SUBJECT_NAMES]
        other_pass = 0
        for subj in other_subjects:
            if grade_to_numeric(row[subj]) >= syarat['other_min']:
                other_pass += 1
        if other_pass < syarat['other_count']:
            return False
    
    return True

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
    
    base_score = skor / total_bobot * 100 if total_bobot > 0 else 50
    
    # Bonus group priority
    priority_bonus = {7: 20, 6: 15, 2: 12, 3: 10, 4: 8, 5: 5, 1: 0}
    bonus = priority_bonus.get(program.get('group', 1), 0)
    base_score = min(base_score + bonus, 100)
    
    return round(base_score, 1)

# ============================================
# FUNGSI PENJELASAN (XAI)
# ============================================
def generate_explanation(row, program):
    group = program.get('group', 0)
    reasons = []
    
    if group == 7:
        if grade_to_numeric(row.get('M-T', 0)) >= 75:
            reasons.append(f"Add Math {row.get('M-T', '')} (≥B)")
        if grade_to_numeric(row.get('FIZ', 0)) >= 75:
            reasons.append(f"Fizik {row.get('FIZ', '')} (≥B)")
        if grade_to_numeric(row.get('KIM', 0)) >= 75:
            reasons.append(f"Kimia {row.get('KIM', '')} (≥B)")
        if reasons:
            return "Layak Asasi: " + ", ".join(reasons[:3])
        return "Layak Asasi (syarat minimum dipenuhi)"
    
    elif group == 6:
        if grade_to_numeric(row.get('ACC', 0)) >= 75:
            reasons.append(f"ACC {row.get('ACC', '')} (≥B)")
        if grade_to_numeric(row.get('MAT', 0)) >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if reasons:
            return "Layak Accounting + SAP: " + ", ".join(reasons)
        return "Layak Accounting + SAP"
    
    elif group == 2:
        if grade_to_numeric(row.get('MAT', 0)) >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if grade_to_numeric(row.get('BI', 0)) >= 75:
            reasons.append(f"BI {row.get('BI', '')} (≥B)")
        if reasons:
            return "Layak CS/MK + Sijil: " + ", ".join(reasons)
        return "Layak CS/MK + Sijil"
    
    elif group == 3:
        if grade_to_numeric(row.get('MAT', 0)) >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        return "Layak CS Asas: " + ", ".join(reasons) if reasons else "Layak CS Asas"
    
    elif group == 4:
        if grade_to_numeric(row.get('BI', 0)) >= 75:
            reasons.append(f"BI {row.get('BI', '')} (≥B)")
        return "Layak English: " + ", ".join(reasons) if reasons else "Layak English"
    
    elif group == 5:
        if grade_to_numeric(row.get('MAT', 0)) >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        return "Layak Perakaunan Asas: " + ", ".join(reasons) if reasons else "Layak Perakaunan Asas"
    
    else:  # Group 1
        if grade_to_numeric(row.get('BM', 0)) >= 60:
            reasons.append(f"BM {row.get('BM', '')} (≥C)")
        other_count = 0
        for subj in ['M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO']:
            if grade_to_numeric(row.get(subj, 0)) >= 60:
                other_count += 1
        if other_count >= 2:
            reasons.append(f"{other_count} subjek lain ≥C")
        return "Layak Program Umum: " + ", ".join(reasons) if reasons else "Layak Program Umum"

# ============================================
# SENARAI PROGRAM DENGAN SYARAT TERKINI
# ============================================
ALL_PROGRAMS = [
    # ========== GROUP 1 ==========
    {
        'name': 'Diploma in Integrated Logistics Management + Chartered Institute of Logistics and Transport',
        'group': 1,
        'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}
    },
    {
        'name': 'Diploma in Halal Industry + Halal Executive Certification',
        'group': 1,
        'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}
    },
    {
        'name': 'Diploma in Islamic Finance + Associate Qualification in Islamic Finance',
        'group': 1,
        'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}
    },
    {
        'name': 'Diploma in Business Studies',
        'group': 1,
        'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}
    },
    {
        'name': 'Diploma in Business Information Technology',
        'group': 1,
        'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}
    },
    {
        'name': 'Diploma in International Business',
        'group': 1,
        'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}
    },
    {
        'name': 'Diploma in Creative Digital Media Production',
        'group': 1,
        'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}
    },
    
    # ========== GROUP 2 ==========
    {
        'name': 'Diploma in Computer Science + SAS@Certified Specialist: Visual Business Analytics Certification',
        'group': 2,
        'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 2, 'other_min': 60}
    },
    {
        'name': 'Diploma in Marketing + Certified Professional Marketer (Asia) Certification',
        'group': 2,
        'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 2, 'other_min': 60}
    },
    
    # ========== GROUP 3 ==========
    {
        'name': 'Diploma in Computer Science',
        'group': 3,
        'syarat': {'BM': 60, 'MAT': 60, 'SEJ': 40, 'BI_min': 40, 'other_count': 3, 'other_min': 60}
    },
    
    # ========== GROUP 4 ==========
    {
        'name': 'Diploma in English Communication + Sijil Penterjemahan Bahasa ITBM',
        'group': 4,
        'syarat': {'BM': 60, 'BI': 75, 'MAT': 40, 'SEJ': 40, 'other_count': 1, 'other_min': 60}
    },
    
    # ========== GROUP 5 ==========
    {
        'name': 'Diploma in Accounting',
        'group': 5,
        'syarat': {'BM': 60, 'MAT': 60, 'SEJ': 40, 'BI_min': 40, 'other_count': 1, 'other_min': 60}
    },
    
    # ========== GROUP 6 ==========
    {
        'name': 'Diploma in Accounting + SAP S/4HANA Financial Accounting Associates Certification',
        'group': 6,
        'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 1, 'other_min': 60}
    },
    
    # ========== GROUP 7 ==========
    {
        'name': 'Asasi Kejuruteraan & Teknologi - Universiti Teknologi Malaysia',
        'group': 7,
        'syarat': {'BM': 85, 'MAT': 85, 'M-T': 75, 'SEJ': 40, 'sains_min': 75, 'other_count': 2, 'other_min': 75}
    },
    {
        'name': 'Asasi Kejuruteraan & Teknologi - Universiti Malaysia Pahang Al-Sultan Abdullah',
        'group': 7,
        'syarat': {'BM': 85, 'MAT': 85, 'M-T': 75, 'SEJ': 40, 'sains_min': 75, 'other_count': 2, 'other_min': 75}
    }
]

# ============================================
# FUNGSI CHECK PROGRAM DITAWAR
# ============================================
def check_offered_program(program_ditawar, pilihan_asal):
    if program_ditawar == 'TIDAK DITAWARKAN' or pd.isna(program_ditawar):
        return None
    
    for i, p in enumerate(pilihan_asal, 1):
        if program_ditawar.lower() in p.lower():
            return {'type': 'success', 'message': f"✅ Program Ditawar: {program_ditawar} (Pilihan {i})"}
    
    return {
        'type': 'info',
        'message': f"✅ Program Ditawar: {program_ditawar}",
        'note': "📝 Nota: Program ini mungkin dalam pilihan 4-12 dalam senarai penuh UPUOnline."
    }

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
        if cari_melalui == "NOKP" and (not nokp_input or nokp_input.strip() == ""):
            st.error("❌ Sila masukkan NOKP")
            st.stop()
        elif cari_melalui == "Nama" and (not nama_input or nama_input.strip() == ""):
            st.error("❌ Sila masukkan nama")
            st.stop()
        
        if cari_melalui == "NOKP":
            pelajar = df[df['NOKP'].astype(str).str.contains(nokp_input, na=False)]
        else:
            df['NAMA'] = df['NAMA'].fillna('')
            pelajar = df[df['NAMA'].str.contains(nama_input, case=False, na=False)]
        
        if len(pelajar) == 0:
            st.error("❌ Pelajar tidak dijumpai")
        else:
            row = pelajar.iloc[0]
            
            # ========================================
            # LAYOUT 2 KOLOM
            # ========================================
            col_kiri, col_kanan = st.columns([1, 2])
            
            with col_kiri:
                st.markdown("### 👤 Profil Pelajar")
                
                # Profil dalam table
                st.markdown(f"""
| | |
|---|---|
| **NOKP** | {row['NOKP']} |
| **Nama** | {row['NAMA']} |
| **Jantina** | {'Perempuan' if row.get('JANTINA')=='P' else 'Lelaki'} |
| **Lokasi** | {row.get('LOKASI', 'N/A')} |
| **Aliran** | {row.get('ALIRAN', 'N/A')} |
| **Pendapatan** | RM {row.get('PENDAPATAN', 0):,.0f} |
                """)
                
                # SUBJEK SPM
                st.markdown("### 📚 Subjek SPM")
                
                subject_data = []
                for code, name in SUBJECT_NAMES.items():
                    if code in row.index:
                        grade = row.get(code)
                        if pd.notna(grade) and grade != 'NA' and grade != '':
                            subject_data.append({"Subjek": name, "Gred": grade})
                
                if subject_data:
                    df_subjects = pd.DataFrame(subject_data)
                    st.dataframe(df_subjects, use_container_width=True, hide_index=True)
                else:
                    st.info("Tiada data subjek")
                
                # PERINCIAN SKOR
                st.markdown("### 📊 Perincian Skor")
                st.markdown("""
**Komponen Skor:**
- Demografi: 10%
- Pendapatan: 10% (B40 > T20)
- Subjek: 80% (purata subjek berkaitan)
- Bonus Kumpulan: Asasi +20%, Accounting+SAP +15%, dsb
- Bonus: +15% jika dalam pilihan asal

**Kelayakan:**
- ≥80%: Sangat sesuai
- 60-79%: Sederhana sesuai
- <60%: Kurang sesuai
                """)
            
            with col_kanan:
                st.markdown("### 🎯 Cadangan Program")
                
                pilihan_asal = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        pilihan_asal.append(str(row[pil]).strip())
                
                # Kumpul SEMUA program
                all_programs = []
                
                for prog in ALL_PROGRAMS:
                    eligible = is_eligible(row, prog)
                    score = hitung_skor(row, prog) if eligible else 0
                    in_original = any(prog['name'].lower() in p.lower() for p in pilihan_asal)
                    if in_original and eligible:
                        score = min(score + 15, 100)
                    
                    all_programs.append({
                        'name': prog['name'],
                        'group': prog['group'],
                        'score': score,
                        'eligible': eligible,
                        'in_original': in_original,
                        'explanation': generate_explanation(row, prog) if eligible else ""
                    })
                
                # Susun ikut skor tertinggi
                all_programs.sort(key=lambda x: -x['score'])
                
                st.caption(f"📊 Jumlah program: {len(all_programs)}")
                
                for i, prog in enumerate(all_programs, 1):
                    if prog['eligible']:
                        if prog['score'] >= 80:
                            color = "#28a745"
                        elif prog['score'] >= 60:
                            color = "#ffc107"
                        else:
                            color = "#dc3545"
                        
                        star = " ⭐" if prog['in_original'] else ""
                        
                        st.markdown(f"""
                        <div style='margin-bottom: 12px; padding: 10px; border-left: 5px solid {color}; border-radius: 5px; background-color: white; border: 1px solid #e0e0e0;'>
                            <span style='font-size: 1em;'><b>{i}. {prog['name']}{star}</b></span><br>
                            <span style='font-size: 0.85em; color: {color};'><b>Kesesuaian: {prog['score']}%</b></span><br>
                            <span style='font-size: 0.75em; color: #555;'><i>✓ {prog['explanation']}</i></span><br>
                            <span style='font-size: 0.7em; color: gray;'>Kumpulan {prog['group']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Program tak layak
                        st.markdown(f"""
                        <div style='margin-bottom: 8px; padding: 8px; border-left: 5px solid #dc3545; border-radius: 5px; background-color: white; border: 1px solid #e0e0e0;'>
                            <span style='font-size: 0.9em;'><b>{i}. {prog['name']}</b></span><br>
                            <span style='font-size: 0.75em; color: #dc3545;'><b>❌ Tidak layak</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # PILIHAN ASAL
                st.markdown("### 📋 Pilihan Asal Pelajar")
                choice_rows = []
                for i, p in enumerate(pilihan_asal, 1):
                    in_list = any(p.lower() in prog['name'].lower() for prog in all_programs if prog['eligible'])
                    status = "✅ Dalam cadangan" if in_list else "❌ Tidak dalam cadangan"
                    choice_rows.append([f"PIL{i}", p, status])
                
                if choice_rows:
                    df_choices = pd.DataFrame(choice_rows, columns=["Pilihan", "Program", "Status"])
                    st.dataframe(df_choices, use_container_width=True, hide_index=True)
                
                # PROGRAM DITAWAR
                if 'KURSUSJAYA' in row.index and pd.notna(row['KURSUSJAYA']):
                    program_ditawar = str(row['KURSUSJAYA']).strip()
                    if program_ditawar != 'TIDAK DITAWARKAN':
                        offered_info = check_offered_program(program_ditawar, pilihan_asal)
                        if offered_info:
                            if offered_info['type'] == 'success':
                                st.success(offered_info['message'])
                            else:
                                st.info(f"{offered_info['message']}\n\n{offered_info.get('note', '')}")
