import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="MARA Program Recommendation",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 MARA Program Recommendation System")

# Load model and data
@st.cache_resource
def load_model_and_data():
    model = joblib.load('mara_model.pkl')
    df = pd.read_csv('data_lengkap.csv')
    return model, df

model, df = load_model_and_data()
feature_names = list(model.feature_names_in_)

# ============================================
# SPM SUBJECT DICTIONARY
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
# GRADE TO NUMERIC FUNCTION
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
# CHECK ELIGIBILITY
# ============================================
def is_eligible(row, program):
    syarat = program.get('syarat', {})
    
    # History
    sejarah = grade_to_numeric(row.get('SEJ', 0))
    if sejarah < syarat.get('SEJ', 40):
        return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ {syarat.get('SEJ', 40)})"
    
    # BM
    bm = grade_to_numeric(row.get('BM', 0))
    if bm < syarat.get('BM', 60):
        return False, f"BM: {row.get('BM', 'N/A')} (need ≥ {syarat.get('BM', 60)})"
    
    # Math
    math = grade_to_numeric(row.get('MAT', 0))
    if math < syarat.get('MAT', 0):
        return False, f"Math: {row.get('MAT', 'N/A')} (need ≥ {syarat.get('MAT', 0)})"
    
    # English
    bi = grade_to_numeric(row.get('BI', 0))
    if 'BI' in syarat:
        if bi < syarat['BI']:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ {syarat['BI']})"
    elif 'BI_min' in syarat:
        if bi < syarat['BI_min']:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ {syarat['BI_min']})"
    
    # Additional Math
    if 'M-T' in syarat:
        mt = grade_to_numeric(row.get('M-T', 0))
        if mt < syarat['M-T']:
            return False, f"Add Math: {row.get('M-T', 'N/A')} (need ≥ {syarat['M-T']})"
    
    # Science (Physics or Chemistry)
    if 'sains_min' in syarat:
        fizik = grade_to_numeric(row.get('FIZ', 0))
        kim = grade_to_numeric(row.get('KIM', 0))
        if max(fizik, kim) < syarat['sains_min']:
            return False, f"Physics/Chemistry: best = {max(fizik, kim)} (need ≥ {syarat['sains_min']})"
    
    # Other subjects
    if 'other_count' in syarat:
        wajib = ['BM', 'BI', 'MAT', 'SEJ', 'M-T', 'FIZ', 'KIM']
        other_subjects = [col for col in row.index if col not in wajib and col in SUBJECT_NAMES]
        other_pass = 0
        for subj in other_subjects:
            if grade_to_numeric(row[subj]) >= syarat['other_min']:
                other_pass += 1
        if other_pass < syarat['other_count']:
            return False, f"Only {other_pass}/{syarat['other_count']} other subjects ≥ {syarat['other_min']}"
    
    return True, ""

# ============================================
# CALCULATE SCORE
# ============================================
def calculate_score(row, program):
    score = 0
    total_weight = 0
    
    # Demographic (10%)
    score += 10
    total_weight += 10
    
    # Income (10%)
    income = row.get('PENDAPATAN', 5000)
    if income < 3000:
        score += 10
    elif income < 5000:
        score += 8
    elif income < 8000:
        score += 6
    else:
        score += 4
    total_weight += 10
    
    # Subjects (80%)
    subject_count = 0
    subject_total = 0
    syarat = program.get('syarat', {})
    all_subjects = []
    for key in syarat:
        if key in ['BM', 'BI', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI', 'SEJ']:
            all_subjects.append(key)
    
    unique_subjects = list(set(all_subjects))
    
    for subj in unique_subjects:
        if subj in ['BM', 'BI', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PI', 'PQS', 'PSI', 'SEJ']:
            value = grade_to_numeric(row.get(subj, 0))
            if value > 0:
                subject_total += value
                subject_count += 1
    
    if subject_count > 0:
        average = subject_total / subject_count
        score += average * 0.8
        total_weight += 80
    
    base_score = score / total_weight * 100 if total_weight > 0 else 50
    
    # Group priority bonus
    priority_bonus = {7: 20, 6: 15, 2: 12, 3: 10, 4: 8, 5: 5, 1: 0}
    bonus = priority_bonus.get(program.get('group', 1), 0)
    base_score = min(base_score + bonus, 100)
    
    return round(base_score, 1)

# ============================================
# GENERATE EXPLANATION
# ============================================
def generate_explanation(row, program):
    group = program.get('group', 0)
    reasons = []
    
    if group == 7:
        if grade_to_numeric(row.get('M-T', 0)) >= 75:
            reasons.append(f"Add Math {row.get('M-T', '')} (≥B)")
        if grade_to_numeric(row.get('FIZ', 0)) >= 75:
            reasons.append(f"Physics {row.get('FIZ', '')} (≥B)")
        if grade_to_numeric(row.get('KIM', 0)) >= 75:
            reasons.append(f"Chemistry {row.get('KIM', '')} (≥B)")
        if reasons:
            return "Eligible for Foundation: " + ", ".join(reasons[:3])
        return "Eligible for Foundation (minimum requirements met)"
    
    elif group == 6:
        if grade_to_numeric(row.get('ACC', 0)) >= 75:
            reasons.append(f"ACC {row.get('ACC', '')} (≥B)")
        if grade_to_numeric(row.get('MAT', 0)) >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if reasons:
            return "Eligible for Accounting + SAP: " + ", ".join(reasons)
        return "Eligible for Accounting + SAP"
    
    elif group == 2:
        if grade_to_numeric(row.get('MAT', 0)) >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if grade_to_numeric(row.get('BI', 0)) >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        if reasons:
            return "Eligible for CS/Marketing + Certification: " + ", ".join(reasons)
        return "Eligible for CS/Marketing + Certification"
    
    elif group == 3:
        if grade_to_numeric(row.get('MAT', 0)) >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        return "Eligible for CS Basic: " + ", ".join(reasons) if reasons else "Eligible for CS Basic"
    
    elif group == 4:
        if grade_to_numeric(row.get('BI', 0)) >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        return "Eligible for English Communication: " + ", ".join(reasons) if reasons else "Eligible for English Communication"
    
    elif group == 5:
        if grade_to_numeric(row.get('MAT', 0)) >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        return "Eligible for Accounting Basic: " + ", ".join(reasons) if reasons else "Eligible for Accounting Basic"
    
    else:  # Group 1
        if grade_to_numeric(row.get('BM', 0)) >= 60:
            reasons.append(f"BM {row.get('BM', '')} (≥C)")
        other_count = 0
        for subj in ['M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO']:
            if grade_to_numeric(row.get(subj, 0)) >= 60:
                other_count += 1
        if other_count >= 2:
            reasons.append(f"{other_count} other subjects ≥C")
        return "Eligible for General Programs: " + ", ".join(reasons) if reasons else "Eligible for General Programs"

# ============================================
# ALL PROGRAMS
# ============================================
ALL_PROGRAMS = [
    # GROUP 1
    {'name': 'Diploma in Integrated Logistics Management + Chartered Institute of Logistics and Transport', 'group': 1,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Halal Industry + Halal Executive Certification', 'group': 1,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Islamic Finance + Associate Qualification in Islamic Finance', 'group': 1,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Business Studies', 'group': 1,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Business Information Technology', 'group': 1,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in International Business', 'group': 1,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Creative Digital Media Production', 'group': 1,
     'syarat': {'BM': 60, 'MAT': 40, 'SEJ': 40, 'BI_min': 40, 'other_count': 2, 'other_min': 60}},
    
    # GROUP 2
    {'name': 'Diploma in Computer Science + SAS@Certified Specialist: Visual Business Analytics Certification', 'group': 2,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 2, 'other_min': 60}},
    {'name': 'Diploma in Marketing + Certified Professional Marketer (Asia) Certification', 'group': 2,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 2, 'other_min': 60}},
    
    # GROUP 3
    {'name': 'Diploma in Computer Science', 'group': 3,
     'syarat': {'BM': 60, 'MAT': 60, 'SEJ': 40, 'BI_min': 40, 'other_count': 3, 'other_min': 60}},
    
    # GROUP 4
    {'name': 'Diploma in English Communication + Sijil Penterjemahan Bahasa ITBM', 'group': 4,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 40, 'SEJ': 40, 'other_count': 1, 'other_min': 60}},
    
    # GROUP 5
    {'name': 'Diploma in Accounting', 'group': 5,
     'syarat': {'BM': 60, 'MAT': 60, 'SEJ': 40, 'BI_min': 40, 'other_count': 1, 'other_min': 60}},
    
    # GROUP 6
    {'name': 'Diploma in Accounting + SAP S/4HANA Financial Accounting Associates Certification', 'group': 6,
     'syarat': {'BM': 60, 'BI': 75, 'MAT': 75, 'SEJ': 40, 'other_count': 1, 'other_min': 60}},
    
    # GROUP 7
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Teknologi Malaysia', 'group': 7,
     'syarat': {'BM': 85, 'MAT': 85, 'M-T': 75, 'SEJ': 40, 'sains_min': 75, 'other_count': 2, 'other_min': 75}},
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Malaysia Pahang Al-Sultan Abdullah', 'group': 7,
     'syarat': {'BM': 85, 'MAT': 85, 'M-T': 75, 'SEJ': 40, 'sains_min': 75, 'other_count': 2, 'other_min': 75}},
]

# ============================================
# CHECK OFFERED PROGRAM
# ============================================
def check_offered_program(program_ditawar, original_choices):
    if program_ditawar == 'TIDAK DITAWARKAN' or pd.isna(program_ditawar):
        return None
    
    for i, p in enumerate(original_choices, 1):
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
search_by = st.sidebar.radio("Search by:", ["NOKP", "Name"])

if search_by == "NOKP":
    nokp_input = st.sidebar.text_input("Enter 12-digit IC Number", placeholder="030807060678")
else:
    name_input = st.sidebar.text_input("Enter full name", placeholder="NUR AELYA")

search_button = st.sidebar.button("🔍 Search Student")

# ============================================
# MAIN AREA
# ============================================
if search_button:
    with st.spinner("Searching for student..."):
        if search_by == "NOKP" and (not nokp_input or nokp_input.strip() == ""):
            st.error("❌ Please enter IC Number")
            st.stop()
        elif search_by == "Name" and (not name_input or name_input.strip() == ""):
            st.error("❌ Please enter name")
            st.stop()
        
        if search_by == "NOKP":
            student = df[df['NOKP'].astype(str).str.contains(nokp_input, na=False)]
        else:
            df['NAMA'] = df['NAMA'].fillna('')
            student = df[df['NAMA'].str.contains(name_input, case=False, na=False)]
        
        if len(student) == 0:
            st.error("❌ Student not found")
        else:
            row = student.iloc[0]
            
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.markdown("### 👤 Student Profile")
                
                # Profile without labels (just data)
                st.markdown(f"""
                <div style='background-color: #FFFFFF; padding: 10px; border-radius: 8px; margin-bottom: 20px;'>
                <table style='width:100%; border-collapse: collapse; background-color: transparent;'>
                    <tr><td style='padding: 6px; color: #000000;'>{row['NOKP']}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'>{row['NAMA']}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'>{'Female' if row.get('JANTINA')=='P' else 'Male'}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'>{row.get('LOKASI', 'N/A')}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'>{row.get('ALIRAN', 'N/A')}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'>RM {row.get('PENDAPATAN', 0):,.0f}</td></tr>
                </table>
                </div>
                """, unsafe_allow_html=True)
                
                # SPM Subjects
                st.markdown("### 📚 SPM Subjects")
                
                subject_data = []
                for code, name in SUBJECT_NAMES.items():
                    if code in row.index:
                        grade = row.get(code)
                        if pd.notna(grade) and grade != 'NA' and grade != '':
                            subject_data.append({"Subject": name, "Grade": grade})
                
                if subject_data:
                    df_subjects = pd.DataFrame(subject_data)
                    # Center the Grade column
                    st.dataframe(df_subjects.style.set_properties(**{'text-align': 'center'}, subset=['Grade']), 
                                use_container_width=True, hide_index=True)
                else:
                    st.info("No subject data found")
                
                # Score Details
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
            
            with col_right:
                # Get original choices
                original_choices = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        original_choices.append(str(row[pil]).strip())
                
                # ========================================
                # ORIGINAL CHOICES (ABOVE)
                # ========================================
                st.markdown("### 📋 Student's Original Choices")
                choice_rows = []
                for i, p in enumerate(original_choices, 1):
                    choice_rows.append([f"Choice {i}", p])
                
                if choice_rows:
                    df_choices = pd.DataFrame(choice_rows, columns=["Choice", "Program"])
                    st.dataframe(df_choices, use_container_width=True, hide_index=True)
                else:
                    st.info("No original choices recorded")
                
                # ========================================
                # PROGRAM RECOMMENDATIONS (BELOW)
                # ========================================
                st.markdown("### 🎯 Program Recommendations")
                
                # Evaluate ALL programs
                all_programs = []
                
                for prog in ALL_PROGRAMS:
                    eligible, reason = is_eligible(row, prog)
                    score = calculate_score(row, prog) if eligible else 0
                    in_original = any(prog['name'].lower() in p.lower() for p in original_choices)
                    if in_original and eligible:
                        score = min(score + 15, 100)
                    
                    all_programs.append({
                        'name': prog['name'],
                        'group': prog['group'],
                        'score': score,
                        'eligible': eligible,
                        'in_original': in_original,
                        'explanation': generate_explanation(row, prog) if eligible else "",
                        'reason': reason if not eligible else ""
                    })
                
                # Sort by score (highest first)
                all_programs.sort(key=lambda x: -x['score'])
                
                st.caption(f"📊 Total programs: {len(all_programs)}")
                
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
                        <div style='margin-bottom: 12px; padding: 10px; border-left: 5px solid {color}; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0;'>
                            <span style='font-size: 1em; color: #000000;'><b>{i}. {prog['name']}{star}</b></span><br>
                            <span style='font-size: 0.85em; color: {color};'><b>Suitability: {prog['score']}%</b></span><br>
                            <span style='font-size: 0.75em; color: #555555;'><i>✓ {prog['explanation']}</i></span><br>
                            <span style='font-size: 0.7em; color: #888888;'>Group {prog['group']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Not eligible with detailed reason
                        st.markdown(f"""
                        <div style='margin-bottom: 8px; padding: 8px; border-left: 5px solid #dc3545; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0;'>
                            <span style='font-size: 0.9em; color: #000000;'><b>{i}. {prog['name']}</b></span><br>
                            <span style='font-size: 0.75em; color: #dc3545;'><b>❌ Not eligible:</b> {prog['reason']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Offered Program
                if 'KURSUSJAYA' in row.index and pd.notna(row['KURSUSJAYA']):
                    program_offered = str(row['KURSUSJAYA']).strip()
                    if program_offered != 'TIDAK DITAWARKAN':
                        offered_info = check_offered_program(program_offered, original_choices)
                        if offered_info:
                            if offered_info['type'] == 'success':
                                st.success(offered_info['message'])
                            else:
                                st.info(f"{offered_info['message']}\n\n{offered_info.get('note', '')}")
