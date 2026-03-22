import streamlit as st
import pandas as pd
import joblib
import numpy as np

# NO plotly import - removed!

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

# Subject weights per group (for academic score calculation)
GROUP_SUBJECT_WEIGHTS = {
    7: {'M-T': 1.5, 'FIZ': 1.5, 'KIM': 1.5, 'MAT': 1.2, 'BM': 1.2, 'BI': 0.8, 'default': 0.8},
    6: {'ACC': 1.3, 'MAT': 1.3, 'BI': 1.3, 'default': 0.7},
    5: {'MAT': 1.2, 'ACC': 1.2, 'default': 0.8},
    4: {'BI': 1.5, 'BM': 1.2, 'default': 0.7},
    3: {'MAT': 1.3, 'SK': 1.3, 'BI': 1.1, 'default': 0.8},
    2: {'MAT': 1.3, 'BI': 1.3, 'SK': 1.2, 'default': 0.8},
    1: {'BM': 1.2, 'MAT': 1.0, 'BI': 1.0, 'SEJ': 0.8, 'default': 0.8}
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
# XAI: DETAILED SCORE BREAKDOWN
# ============================================
def calculate_detailed_score(row, program):
    """
    Calculate suitability score with detailed breakdown for XAI.
    """
    
    group = program.get('group', 1)
    weights = GROUP_SUBJECT_WEIGHTS.get(group, GROUP_SUBJECT_WEIGHTS[1])
    
    # ============================================
    # 1. ACADEMIC SCORE (80% of final)
    # ============================================
    
    # Determine relevant subjects for this program group
    if group == 7:  # Engineering
        relevant_subjects = ['M-T', 'FIZ', 'KIM', 'MAT', 'BM', 'BI']
    elif group == 6:  # Accounting + SAP
        relevant_subjects = ['ACC', 'MAT', 'BI', 'BM']
    elif group == 5:  # Accounting Basic
        relevant_subjects = ['MAT', 'ACC', 'BM']
    elif group == 4:  # English Communication
        relevant_subjects = ['BI', 'BM', 'SEJ']
    elif group in [2, 3]:  # Computer Science
        relevant_subjects = ['MAT', 'BI', 'SK', 'BM']
    else:  # Group 1
        relevant_subjects = ['BM', 'MAT', 'BI', 'SEJ']
    
    academic_breakdown = []
    academic_total = 0
    academic_weight = 0
    
    for subj in relevant_subjects:
        if subj in row.index:
            grade_value = grade_to_numeric(row.get(subj, 0))
            weight = weights.get(subj, weights.get('default', 1.0))
            
            if grade_value > 0:
                subject_score = (grade_value / 100) * weight * 100
                academic_total += subject_score
                academic_weight += weight
                
                academic_breakdown.append({
                    'subject': SUBJECT_NAMES.get(subj, subj),
                    'code': subj,
                    'grade': row.get(subj, 'N/A'),
                    'grade_value': grade_value,
                    'weight': weight,
                    'contribution': round(subject_score, 1)
                })
    
    # Calculate academic score (max 100)
    if academic_weight > 0:
        academic_raw = academic_total / academic_weight
    else:
        academic_raw = 50
    
    academic_score = min(academic_raw, 100)
    
    # ============================================
    # 2. DEMOGRAPHIC SCORE (10% of final)
    # ============================================
    
    demographic_breakdown = []
    demographic_total = 0
    
    # Location score (max 50 points within demographic)
    location = row.get('LOKASI', 'URBAN')
    if location == 'RURAL':
        location_score = 50
        location_note = "Rural location: priority given to support rural students"
    else:
        location_score = 30
        location_note = "Urban location: base score (30)"
    
    demographic_total += location_score
    demographic_breakdown.append({
        'factor': 'Location',
        'value': location,
        'score': location_score,
        'max_score': 50,
        'note': location_note
    })
    
    # Income score (max 50 points within demographic)
    income = row.get('PENDAPATAN', 5000)
    if income < 3000:  # B40
        income_score = 50
        income_category = 'B40 (Low Income)'
        income_note = 'B40: maximum priority to support students from low-income families'
    elif income < 5000:  # M40 lower
        income_score = 40
        income_category = 'M40 (Lower Middle)'
        income_note = 'M40: high priority'
    elif income < 8000:  # M40 upper
        income_score = 30
        income_category = 'M40 (Upper Middle)'
        income_note = 'M40: moderate priority'
    else:  # T20
        income_score = 20
        income_category = 'T20 (High Income)'
        income_note = 'T20: base score'
    
    demographic_total += income_score
    demographic_breakdown.append({
        'factor': 'Income',
        'value': f"RM {income:,.0f} ({income_category})",
        'score': income_score,
        'max_score': 50,
        'note': income_note
    })
    
    # Demographic score is out of 100
    demographic_score = demographic_total
    
    # ============================================
    # 3. PREFERENCE ALIGNMENT (10% of final, max 15 points)
    # ============================================
    
    # Get original choices
    original_choices = []
    for pil in ['PIL1', 'PIL2', 'PIL3']:
        if pil in row.index and pd.notna(row[pil]):
            original_choices.append(str(row[pil]).strip())
    
    # Check if this program matches any original choice
    program_name = program['name']
    matched = False
    choice_number = None
    
    for i, choice in enumerate(original_choices, 1):
        if program_name.lower() in choice.lower() or choice.lower() in program_name.lower():
            matched = True
            choice_number = i
            break
    
    # Preference bonus (max 15 points)
    if matched:
        if choice_number == 1:
            preference_bonus = 15
            preference_note = f"✓ This program matches your 1st choice! Maximum bonus (+15)"
        elif choice_number == 2:
            preference_bonus = 12
            preference_note = f"✓ This program matches your 2nd choice (+12)"
        else:
            preference_bonus = 10
            preference_note = f"✓ This program matches your 3rd choice (+10)"
    else:
        preference_bonus = 0
        preference_note = "✗ This program is not in your original choices"
    
    # ============================================
    # 4. CALCULATE TOTAL SCORE
    # ============================================
    
    total_score = (academic_score * 0.8) + (demographic_score * 0.1) + preference_bonus
    total_score = min(total_score, 100)
    
    # ============================================
    # 5. RETURN DETAILED BREAKDOWN
    # ============================================
    
    return {
        'total_score': round(total_score, 1),
        'academic_score': round(academic_score, 1),
        'demographic_score': round(demographic_score, 1),
        'preference_bonus': preference_bonus,
        'breakdown': {
            'academic': {
                'subjects': academic_breakdown,
                'calculation': f"Total weighted: {academic_total:.1f} / Weight sum: {academic_weight:.1f} = {academic_score:.1f}%"
            },
            'demographic': {
                'components': demographic_breakdown,
                'calculation': f"{demographic_breakdown[0]['factor']} ({demographic_breakdown[0]['score']}) + {demographic_breakdown[1]['factor']} ({demographic_breakdown[1]['score']}) = {demographic_score}%"
            },
            'preference': {
                'matched': matched,
                'choice_number': choice_number,
                'bonus': preference_bonus,
                'note': preference_note,
                'original_choices': original_choices
            }
        },
        'weight_formula': f"({academic_score:.1f} × 0.8) + ({demographic_score:.1f} × 0.1) + {preference_bonus} = {round(total_score, 1)}%",
        'formula_explanation': "Suitability = (Academic × 80%) + (Demographic × 10%) + Preference Bonus (max 15)"
    }

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
# CALCULATE SCORE (BASIC)
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
        return {
            'type': 'info',
            'message': "❌ Student was not offered any program.",
            'note': "This may be due to quota limitations or the student not meeting the required criteria."
        }
    
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
                
                st.markdown(f"""
                <div style='background-color: #FFFFFF; padding: 10px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e0e0e0;'>
                <table style='width:100%; border-collapse: collapse; background-color: transparent;'>
                    <tr><td style='padding: 6px; color: #000000;'><b>NOKP</b></td><td style='padding: 6px; color: #000000;'>{row['NOKP']}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'><b>Name</b></td><td style='padding: 6px; color: #000000;'>{row['NAMA']}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'><b>Gender</b></td><td style='padding: 6px; color: #000000;'>{'Female' if row.get('JANTINA')=='P' else 'Male'}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'><b>Location</b></td><td style='padding: 6px; color: #000000;'>{row.get('LOKASI', 'N/A')}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'><b>Academic Stream</b></td><td style='padding: 6px; color: #000000;'>{row.get('ALIRAN', 'N/A')}</td></tr>
                    <tr><td style='padding: 6px; color: #000000;'><b>Parental Income</b></td><td style='padding: 6px; color: #000000;'>RM {row.get('PENDAPATAN', 0):,.0f}</td></tr>
                 </table>
                </div>
                """, unsafe_allow_html=True)
                
                # Not Offered Note
                if 'KURSUSJAYA' in row.index and pd.notna(row['KURSUSJAYA']):
                    program_offered = str(row['KURSUSJAYA']).strip()
                    if program_offered == 'TIDAK DITAWARKAN':
                        st.info("❌ **Student was not offered any program.**\n\nThis may be due to quota limitations or the student not meeting the required criteria.")
                
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
                    st.dataframe(df_subjects, use_container_width=True, hide_index=True)
                else:
                    st.info("No subject data found")
                
                # Score Details Info
                with st.expander("ℹ️ How Score is Calculated"):
                    st.markdown("""
                    ### Suitability Score Formula
                    
                    **Total = (Academic × 80%) + (Demographic × 10%) + Preference Bonus**
                    
                    ---
                    **1. Academic Score (80%)**
                    - Based on SPM grades in relevant subjects
                    - Different subjects have different weights per program
                    - Maximum: 100%
                    
                    **2. Demographic Score (10%)**
                    - Location: Rural (50) / Urban (30)
                    - Income: B40 (50), M40 lower (40), M40 upper (30), T20 (20)
                    - Maximum: 100%
                    
                    **3. Preference Bonus (10%)**
                    - Matches 1st choice: +15
                    - Matches 2nd choice: +12
                    - Matches 3rd choice: +10
                    - No match: 0
                    
                    ---
                    **Eligibility Levels:**
                    - 🟢 **≥80%**: Highly Suitable
                    - 🟡 **60-79%**: Moderately Suitable
                    - 🔴 **<60%**: Less Suitable
                    """)
            
            with col_right:
                # Get original choices
                original_choices = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        original_choices.append(str(row[pil]).strip())
                
                # ========================================
                # ORIGINAL CHOICES
                # ========================================
                st.markdown("### 📋 Student's Original Choices")
                
                # Evaluate ALL programs first to check which ones are in recommendations
                all_programs_for_check = []
                for prog in ALL_PROGRAMS:
                    eligible, _ = is_eligible(row, prog)
                    score = calculate_score(row, prog) if eligible else 0
                    all_programs_for_check.append({
                        'name': prog['name'],
                        'eligible': eligible,
                        'score': score
                    })
                
                # Sort to get top recommendations
                all_programs_for_check.sort(key=lambda x: -x['score'])
                top_recommendations = [p['name'] for p in all_programs_for_check[:10]]
                
                choice_rows = []
                for i, p in enumerate(original_choices, 1):
                    in_list = any(p.lower() in rec.lower() for rec in top_recommendations)
                    status = "✅" if in_list else "❌"
                    choice_rows.append([f"Choice {i}", p, status])
                
                if choice_rows:
                    df_choices = pd.DataFrame(choice_rows, columns=["Choice", "Program", "In Top 10 Recommendations?"])
                    st.dataframe(df_choices, use_container_width=True, hide_index=True)
                else:
                    st.info("No original choices recorded")
                
                # ========================================
                # PROGRAM RECOMMENDATIONS WITH XAI
                # ========================================
                st.markdown("### 🎯 Program Recommendations")
                st.caption("Click on any program to see detailed score breakdown")
                
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
                
                st.caption(f"📊 Showing {len([p for p in all_programs if p['eligible']])} eligible programs out of {len(all_programs)} total")
                
                # Display recommendations with expanders for XAI
                for i, prog in enumerate(all_programs, 1):
                    if prog['eligible']:
                        # Get detailed score breakdown for XAI
                        detailed = calculate_detailed_score(row, prog)
                        
                        if prog['score'] >= 80:
                            color = "#28a745"
                            level = "🟢 Highly Suitable"
                        elif prog['score'] >= 60:
                            color = "#ffc107"
                            level = "🟡 Moderately Suitable"
                        else:
                            color = "#dc3545"
                            level = "🔴 Less Suitable"
                        
                        star = " ⭐" if prog['in_original'] else ""
                        
                        # Create expander for detailed breakdown
                        with st.expander(f"{i}. {prog['name']}{star} - {level} ({prog['score']}%)"):
                            # Main score display with formula
                            st.markdown(f"""
                            <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 8px;'>
                                <h4 style='margin: 0; color: {color};'>🎯 Total Suitability Score: {detailed['total_score']}%</h4>
                                <p style='margin: 5px 0 0 0; font-size: 0.85em;'><b>Formula:</b> {detailed['weight_formula']}</p>
                                <p style='margin: 2px 0 0 0; font-size: 0.75em; color: #666;'>{detailed['formula_explanation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Three columns for the three components
                            col_a, col_b, col_c = st.columns(3)
                            
                            # Column 1: Academic Performance (80%)
                            with col_a:
                                st.markdown(f"""
                                <div style='background-color: #e8f4fd; padding: 10px; border-radius: 8px; height: 100%;'>
                                    <h4 style='margin: 0 0 8px 0;'>📚 Academic</h4>
                                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: #1e88e5;'>{detailed['academic_score']}%</p>
                                    <p style='font-size: 0.7em; color: #666; margin-bottom: 8px;'>Weight: 80%</p>
                                    <hr style='margin: 8px 0;'>
                                """, unsafe_allow_html=True)
                                
                                # Show subject breakdown
                                for subj in detailed['breakdown']['academic']['subjects']:
                                    if subj['grade_value'] > 0:
                                        st.markdown(f"""
                                        <div style='display: flex; justify-content: space-between; font-size: 0.75em; margin-bottom: 4px;'>
                                            <span><b>{subj['subject']}</b> ({subj['grade']})</span>
                                            <span><b>{subj['contribution']:.1f}</b> (×{subj['weight']})</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <hr style='margin: 8px 0;'>
                                    <div style='font-size: 0.65em; color: #666;'>{detailed['breakdown']['academic']['calculation']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Column 2: Demographic Context (10%)
                            with col_b:
                                st.markdown(f"""
                                <div style='background-color: #fef4e8; padding: 10px; border-radius: 8px; height: 100%;'>
                                    <h4 style='margin: 0 0 8px 0;'>🏠 Demographic</h4>
                                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: #fb8c00;'>{detailed['demographic_score']}%</p>
                                    <p style='font-size: 0.7em; color: #666; margin-bottom: 8px;'>Weight: 10%</p>
                                    <hr style='margin: 8px 0;'>
                                """, unsafe_allow_html=True)
                                
                                for comp in detailed['breakdown']['demographic']['components']:
                                    st.markdown(f"""
                                    <div style='margin-bottom: 8px;'>
                                        <div style='display: flex; justify-content: space-between; font-size: 0.75em;'>
                                            <span><b>{comp['factor']}:</b></span>
                                            <span><b>{comp['score']}</b> / {comp['max_score']}</span>
                                        </div>
                                        <div style='font-size: 0.65em; color: #666;'>{comp['note']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div style='font-size: 0.65em; color: #666; margin-top: 8px;'>
                                        {detailed['breakdown']['demographic']['calculation']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Column 3: Preference Alignment (10%)
                            with col_c:
                                pref_bonus = detailed['preference_bonus']
                                if pref_bonus > 0:
                                    bg_color = "#e8f5e9"
                                    border_color = "#4caf50"
                                else:
                                    bg_color = "#fff3e0"
                                    border_color = "#ff9800"
                                
                                st.markdown(f"""
                                <div style='background-color: {bg_color}; padding: 10px; border-radius: 8px; border-left: 4px solid {border_color}; height: 100%;'>
                                    <h4 style='margin: 0 0 8px 0;'>⭐ Preference</h4>
                                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: {border_color};'>{pref_bonus} / 15</p>
                                    <p style='font-size: 0.7em; color: #666; margin-bottom: 8px;'>Weight: 10% (max 15 points)</p>
                                    <hr style='margin: 8px 0;'>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div style='font-size: 0.75em; margin-bottom: 8px;'>
                                        {detailed['breakdown']['preference']['note']}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if detailed['breakdown']['preference']['original_choices']:
                                    st.markdown("**Your original choices:**")
                                    for j, choice in enumerate(detailed['breakdown']['preference']['original_choices'], 1):
                                        st.markdown(f"&nbsp;&nbsp;{j}. {choice[:50]}...")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Progress bar showing total composition
                            st.markdown("---")
                            st.markdown("### 📊 Score Composition")
                            
                            # Calculate contributions
                            academic_contrib = detailed['academic_score'] * 0.8
                            demo_contrib = detailed['demographic_score'] * 0.1
                            pref_contrib = detailed['preference_bonus']
                            
                            # Simple bar using st.progress
                            st.markdown(f"""
                            <div style='margin: 10px 0;'>
                                <div style='display: flex; height: 30px; border-radius: 5px; overflow: hidden;'>
                                    <div style='background-color: #1e88e5; width: {academic_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Academic {academic_contrib:.1f}%</div>
                                    <div style='background-color: #fb8c00; width: {demo_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Demo {demo_contrib:.1f}%</div>
                                    <div style='background-color: #4caf50; width: {pref_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Pref {pref_contrib:.1f}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.caption(f"Total: {detailed['total_score']}% = Academic ({detailed['academic_score']}% × 0.8) + Demographic ({detailed['demographic_score']}% × 0.1) + Preference Bonus ({pref_bonus})")
                            
                            # Brief explanation
                            st.info(f"💡 **Why this score?** {generate_explanation(row, prog)}")
                    
                    else:
                        # Not eligible - keep original display
                        st.markdown(f"""
                        <div style='margin-bottom: 8px; padding: 8px; border-left: 5px solid #dc3545; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0;'>
                            <span style='font-size: 0.9em; color: #000000;'><b>{i}. {prog['name']}</b></span><br>
                            <span style='font-size: 0.75em; color: #dc3545;'><b>❌ Not eligible:</b> {prog['reason']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Offered Program Display
                if 'KURSUSJAYA' in row.index and pd.notna(row['KURSUSJAYA']):
                    program_offered = str(row['KURSUSJAYA']).strip()
                    if program_offered != 'TIDAK DITAWARKAN':
                        offered_info = check_offered_program(program_offered, original_choices)
                        if offered_info:
                            if offered_info['type'] == 'success':
                                st.success(offered_info['message'])
                            else:
                                st.info(f"{offered_info['message']}\n\n{offered_info.get('note', '')}")
