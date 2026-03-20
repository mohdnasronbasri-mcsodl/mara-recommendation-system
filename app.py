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
# FUNGSI XAI MENGIKUT GROUP
# ============================================
def generate_explanation(row, program):
    group = program.get('group', 0)
    syarat = program.get('syarat', {})
    reasons = []
    
    if group == 7:
        mt = grade_to_numeric(row.get('M-T', 0))
        fizik = grade_to_numeric(row.get('FIZ', 0))
        kim = grade_to_numeric(row.get('KIM', 0))
        bm = grade_to_numeric(row.get('BM', 0))
        math = grade_to_numeric(row.get('MAT', 0))
        
        if mt >= 75:
            reasons.append(f"Add Math {row.get('M-T', '')} (≥B)")
        if fizik >= 75:
            reasons.append(f"Fizik {row.get('FIZ', '')} (≥B)")
        if kim >= 75:
            reasons.append(f"Kimia {row.get('KIM', '')} (≥B)")
        if bm >= 85:
            reasons.append(f"BM {row.get('BM', '')} (≥A-)")
        if math >= 85:
            reasons.append(f"Math {row.get('MAT', '')} (≥A-)")
        
        if reasons:
            return "Eligible for Foundation: " + ", ".join(reasons[:3])
        else:
            return "Eligible for Foundation (Minimum requirements met)"
    
    elif group == 6:
        acc = grade_to_numeric(row.get('ACC', 0))
        math = grade_to_numeric(row.get('MAT', 0))
        bi = grade_to_numeric(row.get('BI', 0))
        
        if acc >= 75:
            reasons.append(f"ACC {row.get('ACC', '')} (≥B)")
        if math >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if bi >= 75:
            reasons.append(f"BI {row.get('BI', '')} (≥B)")
        
        return "Eligible for Accounting + SAP: " + ", ".join(reasons[:3]) if reasons else "Eligible for Accounting + SAP"
    
    elif group == 2:
        math = grade_to_numeric(row.get('MAT', 0))
        bi = grade_to_numeric(row.get('BI', 0))
        
        if math >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if bi >= 75:
            reasons.append(f"BI {row.get('BI', '')} (≥B)")
        
        return "Eligible for DCS/DMK + Cert: " + ", ".join(reasons) if reasons else "Eligible for DCS/DMK + Cert"
    
    elif group == 3:
        math = grade_to_numeric(row.get('MAT', 0))
        if math >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        
        other_count = 0
        for subj in ['M-T', 'FIZ', 'KIM', 'BIO', 'ACC']:
            if grade_to_numeric(row.get(subj, 0)) >= 60:
                other_count += 1
        
        if other_count > 0:
            reasons.append(f"{other_count} other subjects ≥C")
        
        return "Eligible for CS Basic: " + ", ".join(reasons) if reasons else "Eligible for CS Basic"
    
    elif group == 4:
        bi = grade_to_numeric(row.get('BI', 0))
        if bi >= 75:
            reasons.append(f"BI {row.get('BI', '')} (≥B)")
        
        return "Eligible for English: " + ", ".join(reasons) if reasons else "Eligible for English Communication"
    
    elif group == 5:
        math = grade_to_numeric(row.get('MAT', 0))
        if math >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        
        return "Eligible for Accounting Basic: " + ", ".join(reasons) if reasons else "Eligible for Accounting Basic"
    
    else:
        bm = grade_to_numeric(row.get('BM', 0))
        if bm >= 60:
            reasons.append(f"BM {row.get('BM', '')} (≥C)")
        
        other_count = 0
        for subj in ['M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO']:
            if grade_to_numeric(row.get(subj, 0)) >= 60:
                other_count += 1
        
        if other_count >= 2:
            reasons.append(f"{other_count} other subjects ≥C")
        
        return "Eligible for General Programs: " + ", ".join(reasons) if reasons else "Eligible for General Programs"

# ============================================
# SENARAI PROGRAM (sama seperti sebelum)
# ============================================
ALL_PROGRAMS = [
    # GROUP 7
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Teknologi Malaysia', 'cluster': 'Engineering', 'group': 7, 'priority': 1,
     'syarat': {'BM': 85, 'MAT': 85, 'M-T': 75, 'SEJ': 40, 'sains_min': 75, 'other_count': 2, 'other_min': 75}},
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Malaysia Pahang Al-Sultan Abdullah', 'cluster': 'Engineering', 'group': 7, 'priority': 1,
     'syarat': {'BM': 85, 'MAT': 85, 'M-T': 75, 'SEJ': 40, 'sains_min': 75, 'other_count': 2, 'other_min': 75}},
    # GROUP 6
    {'name': 'Diploma in Accounting + SAP S/4HANA Financial Accounting Associates Certification', 'cluster': 'Accounting', 'group': 6, 'priority': 2,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 1, 'other_min': 60}},
    # GROUP 2
    {'name': 'Diploma in Computer Science + SAS@Certified Specialist: Visual Business Analytics Certification', 'cluster': 'Computer Science', 'group': 2, 'priority': 3,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Marketing + Certified Professional Marketer (Asia) Certification', 'cluster': 'Marketing', 'group': 2, 'priority': 3,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 2, 'other_min': 60}},
    # GROUP 3
    {'name': 'Diploma in Computer Science', 'cluster': 'Computer Science', 'group': 3, 'priority': 4,
     'syarat': {'BM': 60, 'MAT': 60, 'SEJ': 40, 'BI_min': 40, 'other_count': 3, 'other_min': 60}},
    # GROUP 4
    {'name': 'Diploma in English Communication + Sijil Penterjemahan Bahasa ITBM', 'cluster': 'Language', 'group': 4, 'priority': 5,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 40, 'SEJ': 40, 'other_count': 1, 'other_min': 60}},
    # GROUP 5
    {'name': 'Diploma in Accounting', 'cluster': 'Accounting', 'group': 5, 'priority': 6,
     'syarat': {'BM': 60, 'MAT': 60, 'SEJ': 40, 'BI_min': 40, 'other_count': 1, 'other_min': 60}},
    # GROUP 1
    {'name': 'Diploma in Integrated Logistics Management + Chartered Institute of Logistics and Transport', 'cluster': 'Logistics', 'group': 1, 'priority': 7,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Halal Industry + Halal Executive Certification', 'cluster': 'Halal', 'group': 1, 'priority': 7,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Islamic Finance + Associate Qualification in Islamic Finance', 'cluster': 'Islamic Finance', 'group': 1, 'priority': 7,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Business Studies', 'cluster': 'Business', 'group': 1, 'priority': 7,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Business Information Technology', 'cluster': 'Business IT', 'group': 1, 'priority': 7,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in International Business', 'cluster': 'Business', 'group': 1, 'priority': 7,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Creative Digital Media Production', 'cluster': 'Creative Arts', 'group': 1, 'priority': 7,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
]

# ============================================
# FUNGSI SEMAK KELAYAKAN
# ============================================
def is_eligible(row, program, debug=False):
    syarat = program.get('syarat', {})
    
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    if sejarah < syarat.get('SEJ', 0):
        return False
    
    bm = grade_to_numeric(row.get('BM', 0))
    if bm < syarat.get('BM', 0):
        return False
    
    math = grade_to_numeric(row.get('MAT', 0))
    if math < syarat.get('MAT', 0):
        return False
    
    bi = grade_to_numeric(row.get('BI', 0))
    if 'BI' in syarat:
        if bi < syarat['BI']:
            return False
    elif 'BI_min' in syarat:
        if bi < syarat['BI_min']:
            return False
    
    if 'M-T' in syarat:
        mt = grade_to_numeric(row.get('M-T', 0))
        if mt < syarat['M-T']:
            return False
    
    if 'sains_min' in syarat:
        fizik = grade_to_numeric(row.get('FIZ', 0))
        kim = grade_to_numeric(row.get('KIM', 0))
        if max(fizik, kim) < syarat['sains_min']:
            return False
    
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
    
    skor += 10
    total_bobot += 10
    
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
    
    priority_bonus = {7: 20, 6: 15, 2: 12, 3: 10, 4: 8, 5: 5, 1: 0}
    bonus = priority_bonus.get(program.get('group', 1), 0)
    base_score = min(base_score + bonus, 100)
    
    return round(base_score, 1)

# ============================================
# FUNGSI CHECK PROGRAM DITAWAR
# ============================================
def check_offered_program(program_ditawar, pilihan_asal):
    if program_ditawar == 'TIDAK DITAWARKAN' or pd.isna(program_ditawar):
        return None
    
    for i, p in enumerate(pilihan_asal, 1):
        if program_ditawar.lower() in p.lower():
            return {'type': 'success', 'message': f"✅ Program Offered: {program_ditawar} (Choice {i})"}
    
    return {
        'type': 'info',
        'message': f"✅ Program Offered: {program_ditawar}",
        'note': "📝 Note: This program may be among choices 4-12 in the full UPUOnline list."
    }

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("🔍 Search Student")
cari_melalui = st.sidebar.radio("Search by:", ["NOKP", "Name"])

if cari_melalui == "NOKP":
    nokp_input = st.sidebar.text_input("Enter 12-digit IC Number", placeholder="030807060678")
else:
    nama_input = st.sidebar.text_input("Enter full name", placeholder="NUR AELYA")

cari_button = st.sidebar.button("🔍 Search Student")

# ============================================
# MAIN AREA
# ============================================
if cari_button:
    with st.spinner("Searching for student..."):
        if cari_melalui == "NOKP" and (not nokp_input or nokp_input.strip() == ""):
            st.error("❌ Please enter IC Number")
            st.stop()
        elif cari_melalui == "Name" and (not nama_input or nama_input.strip() == ""):
            st.error("❌ Please enter name")
            st.stop()
        
        if cari_melalui == "NOKP":
            pelajar = df[df['NOKP'].astype(str).str.contains(nokp_input, na=False)]
        else:
            df['NAMA'] = df['NAMA'].fillna('')
            pelajar = df[df['NAMA'].str.contains(nama_input, case=False, na=False)]
        
        if len(pelajar) == 0:
            st.error("❌ Student not found")
        else:
            row = pelajar.iloc[0]
            
            col_kiri, col_kanan = st.columns([1, 2])
            
            with col_kiri:
                st.markdown("### 👤 Student Profile")
                
                # Profil dalam table
                profil_html = '<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 20px;"><table style="width:100%; border-collapse: collapse;">'
                profil_html += f"<tr><td style='padding: 6px;'><b>NOKP</b></td><td style='padding: 6px;'>{row['NOKP']}</td></tr>"
                profil_html += f"<tr><td style='padding: 6px;'><b>Nama</b></td><td style='padding: 6px;'>{row['NAMA']}</td></tr>"
                profil_html += f"<tr><td style='padding: 6px;'><b>Jantina</b></td><td style='padding: 6px;'>{'Perempuan' if row.get('JANTINA')=='P' else 'Lelaki'}</td></tr>"
                profil_html += f"<tr><td style='padding: 6px;'><b>Lokasi</b></td><td style='padding: 6px;'>{row.get('LOKASI', 'N/A')}</td></tr>"
                profil_html += f"<tr><td style='padding: 6px;'><b>Aliran</b></td><td style='padding: 6px;'>{row.get('ALIRAN', 'N/A')}</td></tr>"
                profil_html += f"<tr><td style='padding: 6px;'><b>Pendapatan</b></td><td style='padding: 6px;'>RM {row.get('PENDAPATAN', 0):,.0f}</td></tr>"
                profil_html += "</table></div>"
                st.markdown(profil_html, unsafe_allow_html=True)
                
                # SUBJEK SPM
                st.markdown("### 📚 SPM Subjects")
                subject_rows = []
                for code, name in SUBJECT_NAMES.items():
                    if code in row.index:
                        grade = row.get(code)
                        if pd.notna(grade) and grade != 'NA' and grade != '':
                            subject_rows.append(f"<tr><td style='padding: 4px;'>{name}</td><td style='text-align:center; padding: 4px;'><b>{grade}</b></td></tr>")
                
                if subject_rows:
                    subjects_html = f"""
                    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 8px; max-height: 400px; overflow-y: auto;'>
                    <table style='width:100%; border-collapse: collapse;'>
                        <tr><th style='text-align:left; padding: 6px;'>Subject</th><th style='text-align:center; padding: 6px;'>Grade</th></tr>
                        {''.join(subject_rows[:20])}
                    </table>
                    </div>
                    """
                    st.markdown(subjects_html, unsafe_allow_html=True)
                else:
                    st.info("No subject data found")
                
                # PERINCIAN SKOR
                st.markdown("### 📊 Score Details")
                st.markdown("""
                **Score Components:**
                - Demographic: 10%
                - Income: 10% (B40 higher score)
                - Subjects: 80% (average of relevant subjects)
                - Group Priority: Bonus (Foundation +20%, etc)
                - Bonus: +15% if in student's original choices
                
                **Eligibility:**
                - ≥80%: Highly Suitable
                - 60-79%: Moderately Suitable
                - <60%: Less Suitable
                """)
            
            with col_kanan:
                st.markdown("### 🎯 Recommended Programs (Sorted by Suitability)")
                
                pilihan_asal = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        pilihan_asal.append(str(row[pil]).strip())
                
                # Kumpul SEMUA program yang layak
                eligible_programs = []
                not_eligible_programs = []
                
                for prog in ALL_PROGRAMS:
                    if is_eligible(row, prog):
                        skor = hitung_skor(row, prog)
                        in_original = any(prog['name'].lower() in p.lower() for p in pilihan_asal)
                        if in_original:
                            skor = min(skor + 15, 100)
                        
                        eligible_programs.append({
                            'name': prog['name'],
                            'group': prog['group'],
                            'score': skor,
                            'in_original': in_original,
                            'explanation': generate_explanation(row, prog)
                        })
                    else:
                        # Kumpul program tak layak untuk Not Recommended
                        reasons = []
                        syarat = prog.get('syarat', {})
                        if grade_to_numeric(row.get('SEJ', 0)) < syarat.get('SEJ', 0):
                            reasons.append(f"Sejarah: {row.get('SEJ', 'N/A')} (need ≥ {syarat.get('SEJ', 0)})")
                        if grade_to_numeric(row.get('BM', 0)) < syarat.get('BM', 0):
                            reasons.append(f"BM: {row.get('BM', 'N/A')} (need ≥ {syarat.get('BM', 0)})")
                        if grade_to_numeric(row.get('MAT', 0)) < syarat.get('MAT', 0):
                            reasons.append(f"Math: {row.get('MAT', 'N/A')} (need ≥ {syarat.get('MAT', 0)})")
                        if 'BI' in syarat and grade_to_numeric(row.get('BI', 0)) < syarat['BI']:
                            reasons.append(f"BI: {row.get('BI', 'N/A')} (need ≥ {syarat['BI']})")
                        if 'M-T' in syarat and grade_to_numeric(row.get('M-T', 0)) < syarat['M-T']:
                            reasons.append(f"Add Math: {row.get('M-T', 'N/A')} (need ≥ {syarat['M-T']})")
                        
                        if reasons:
                            not_eligible_programs.append({
                                'name': prog['name'],
                                'group': prog['group'],
                                'reasons': reasons[:2]
                            })
                
                # Sort eligible programs by score
                eligible_programs.sort(key=lambda x: -x['score'])
                
                st.caption(f"📊 Eligible Programs: {len(eligible_programs)} out of {len(ALL_PROGRAMS)}")
                
                if len(eligible_programs) == 0:
                    st.warning("⚠️ No suitable programs found.")
                else:
                    for i, prog in enumerate(eligible_programs, 1):
                        if prog['score'] >= 80:
                            color = "#28a745"
                        elif prog['score'] >= 60:
                            color = "#ffc107"
                        else:
                            color = "#dc3545"
                        
                        star = " ⭐" if prog['in_original'] else ""
                        
                        st.markdown(f"""
                        <div style='margin-bottom: 12px; padding: 10px; border-left: 5px solid {color}; border-radius: 5px; background-color: #f0f2f6;'>
                            <span style='font-size: 1em;'><b>{i}. {prog['name']}{star}</b></span><br>
                            <span style='font-size: 0.85em; color: {color};'><b>Suitability: {prog['score']}%</b></span><br>
                            <span style='font-size: 0.75em; color: #555;'><i>✓ {prog['explanation']}</i></span><br>
                            <span style='font-size: 0.7em; color: gray;'>Group {prog['group']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # PILIHAN ASAL
                    st.markdown("### 📋 Student's Original Choices")
                    table_rows = []
                    for i, p in enumerate(pilihan_asal, 1):
                        in_list = any(p.lower() in prog['name'].lower() for prog in eligible_programs)
                        status = "✅" if in_list else "❌"
                        table_rows.append(f"<tr><td style='text-align:center'>{i}</td><td>{p}</td><td style='text-align:center'>{status}</td></tr>")
                    
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 8px; border-radius: 8px; max-height: 200px; overflow-y: auto;'>
                    <table style='width:100%; border-collapse: collapse;'>
                        <tr><th style='text-align:center'>Choice</th><th>Program</th><th style='text-align:center'>In List?</th></tr>
                        {''.join(table_rows)}
                    </table>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # PROGRAM DITAWAR
                    if 'KURSUSJAYA' in row.index and pd.notna(row['KURSUSJAYA']):
                        program_ditawar = str(row['KURSUSJAYA']).strip()
                        offered_info = check_offered_program(program_ditawar, pilihan_asal)
                        if offered_info:
                            if offered_info['type'] == 'success':
                                st.success(offered_info['message'])
                            else:
                                st.info(f"{offered_info['message']}\n\n{offered_info.get('note', '')}")
                    
                    # NOT RECOMMENDED
                    st.markdown("### ❌ Not Recommended Programs")
                    st.markdown("Programs where the student does not meet the minimum requirements:")
                    
                    if not_eligible_programs:
                        for prog in not_eligible_programs[:8]:
                            st.markdown(f"""
                            <div style='margin-bottom: 8px; padding: 8px; border-left: 5px solid #6c757d; border-radius: 5px; background-color: #f0f2f6;'>
                                <span style='font-size: 0.9em;'><b>{prog['name']}</b></span><br>
                                <span style='font-size: 0.75em; color: #dc3545;'><b>❌ Not eligible:</b> {', '.join(prog['reasons'])}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("✅ All programs are eligible for this student.")
