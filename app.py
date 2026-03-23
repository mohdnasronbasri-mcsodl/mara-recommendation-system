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

# Display MARA logo and title
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://photos.smugmug.com/REKABENTUK/LOGO/KOLEKSI-LOGO/i-j2SVDpd/0/Kkb8LcCXXZcpDqr9scB36rcD6FvXG2DM2gdpwB5kT/X2/logo%20mara%202021-01-X2.png", width=60)
with col_title:
    st.title("MARA Program Recommendation System")

# Load model and data
@st.cache_resource
def load_model_and_data():
    model = joblib.load('mara_model.pkl')
    df = pd.read_csv('data_lengkap.csv')
    return model, df

model, df = load_model_and_data()

# ============================================
# SPM SUBJECT DICTIONARY
# ============================================
SUBJECT_NAMES = {
    'BM': 'BAHASA MELAYU', 'BI': 'BAHASA INGGERIS', 'PI': 'PENDIDIKAN ISLAM',
    'PM': 'PENDIDIKAN MORAL', 'SEJ': 'SEJARAH', 'MAT': 'MATEMATIK',
    'M-T': 'MATEMATIK TAMBAHAN', 'FIZ': 'FIZIK', 'KIM': 'KIMIA',
    'BIO': 'BIOLOGI', 'ACC': 'PRINSIP PERAKAUNAN', 'PT': 'PERDAGANGAN',
    'EKO': 'EKONOMI', 'SK': 'SAINS KOMPUTER', 'PQS': 'PENDIDIKAN AL-QURAN DAN AL-SUNNAH',
    'PSI': "PENDIDIKAN SYARI'AH ISLAMIAH", 'TSI': 'TASAWWUR ISLAM', 'BAT': 'BAHASA ARAB',
    'AWB': 'AL-ADAB WA AL-BALAGHAH','BC': 'BAHASA CINA', 'RGD': 'REKA BENTUK GRAFIK DIGITAL', 'MUL': 'PRODUKSI MULTIMEDIA'
}

# Subject weights per group
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
# HELPER FUNCTION: COUNT SUBJECTS WITH MIN GRADE
# ============================================
def count_subjects_with_grade(row, subjects, min_grade):
    count = 0
    for subj in subjects:
        if subj in row.index:
            grade_val = grade_to_numeric(row.get(subj, 0))
            if grade_val >= min_grade:
                count += 1
    return count

# ============================================
# CHECK ELIGIBILITY
# ============================================
def is_eligible(row, program):
    group = program.get('group', 1)

    def get_grade(subject):
        return grade_to_numeric(row.get(subject, 0))

    # GROUP 1
    if group == 1:
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('MAT') < 40:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ E / 40)"
        if get_grade('BI') < 40:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"

        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'TSI', 'AWB', 'BC', 'RGD', 'MUL']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)

        if subjects_with_C < 3:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 3, including BM)"
        return True, ""

    # GROUP 2
    elif group == 2:
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('MAT') < 75:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ B / 75)"
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"

        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'TSI', 'AWB', 'BC', 'RGD', 'MUL']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)

        if subjects_with_C < 5:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 5, including BM)"
        return True, ""

    # GROUP 3
    elif group == 3:
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('MAT') < 60:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ C / 60)"
        if get_grade('BI') < 40:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"

        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'TSI', 'AWB', 'BC', 'RGD', 'MUL']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)

        if subjects_with_C < 5:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 5, including BM and Mathematics)"
        return True, ""

    # GROUP 4
    elif group == 4:
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        if get_grade('MAT') < 40:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"

        other_subjects = ['M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'TSI', 'AWB', 'BC', 'RGD', 'MUL']
        other_with_C = count_subjects_with_grade(row, other_subjects, 60)

        if other_with_C < 1:
            return False, f"Need at least 1 other subject with grade ≥ C (you have {other_with_C})"
        return True, ""

    # GROUP 5
    elif group == 5:
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('MAT') < 60:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ C / 60)"
        if get_grade('BI') < 40:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"

        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'TSI', 'AWB', 'BC', 'RGD', 'MUL']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)

        if subjects_with_C < 5:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 5, including BM)"
        return True, ""

    # GROUP 6
    elif group == 6:
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('MAT') < 75:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ B / 75)"
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"

        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'TSI', 'AWB', 'BC', 'RGD', 'MUL']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)

        if subjects_with_C < 3:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 3, including BM)"
        return True, ""

    # GROUP 7
    elif group == 7:
        if get_grade('BM') < 85:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ A- / 85)"
        if get_grade('MAT') < 85:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ A- / 85)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        if get_grade('M-T') < 75:
            return False, f"Additional Mathematics: {row.get('M-T', 'N/A')} (need ≥ B / 75)"

        fizik = get_grade('FIZ')
        kim = get_grade('KIM')
        if max(fizik, kim) < 75:
            return False, f"Physics/Chemistry: best = {max(fizik, kim)} (need at least one ≥ B / 75)"

        subjects_to_check_b = ['BI', 'M-T', 'FIZ', 'KIM', 'MAT', 'BM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'SEJ', 'TSI', 'AWB', 'BC', 'RGD', 'MUL']
        subjects_with_B = count_subjects_with_grade(row, subjects_to_check_b, 75)

        if subjects_with_B < 5:
            return False, f"Only {subjects_with_B} subjects with grade ≥ B (need at least 5)"
        return True, ""

    return False, "Program requirements not defined"

# ============================================
# CALCULATE SCORE
# ============================================
def calculate_score(row, program):
    score = 0
    total_weight = 0
    group = program.get('group', 1)

    score += 10
    total_weight += 10

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

    subject_count = 0
    subject_total = 0

    if group == 7:
        relevant = ['BM', 'MAT', 'M-T', 'FIZ', 'KIM', 'BI', 'SEJ']
    elif group == 6:
        relevant = ['BM', 'MAT', 'BI', 'ACC', 'SEJ']
    elif group == 5:
        relevant = ['BM', 'MAT', 'BI', 'ACC', 'SEJ']
    elif group == 4:
        relevant = ['BM', 'BI', 'MAT', 'SEJ']
    elif group in [2, 3]:
        relevant = ['BM', 'MAT', 'BI', 'SK', 'SEJ']
    else:
        relevant = ['BM', 'MAT', 'BI', 'SEJ']

    for subj in relevant:
        if subj in row.index:
            value = grade_to_numeric(row.get(subj, 0))
            if value > 0:
                subject_total += value
                subject_count += 1

    if subject_count > 0:
        average = subject_total / subject_count
        score += average * 0.8
        total_weight += 80

    base_score = score / total_weight * 100 if total_weight > 0 else 50

    priority_bonus = {7: 20, 6: 15, 2: 12, 3: 10, 4: 8, 5: 5, 1: 0}
    bonus = priority_bonus.get(group, 0)
    base_score = min(base_score + bonus, 100)

    return round(base_score, 1)

# ============================================
# XAI: DETAILED SCORE BREAKDOWN
# ============================================
def calculate_detailed_score(row, program):
    group = program.get('group', 1)
    weights = GROUP_SUBJECT_WEIGHTS.get(group, GROUP_SUBJECT_WEIGHTS[1])

    if group == 7:
        relevant_subjects = ['M-T', 'FIZ', 'KIM', 'MAT', 'BM', 'BI', 'SEJ']
    elif group == 6:
        relevant_subjects = ['ACC', 'MAT', 'BI', 'BM', 'SEJ']
    elif group == 5:
        relevant_subjects = ['ACC', 'MAT', 'BM', 'BI', 'SEJ']
    elif group == 4:
        relevant_subjects = ['BI', 'BM', 'SEJ']
    elif group in [2, 3]:
        relevant_subjects = ['MAT', 'BI', 'SK', 'BM', 'SEJ']
    else:
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
                    'grade': row.get(subj, 'N/A'),
                    'grade_value': grade_value,
                    'weight': weight,
                    'contribution': round(subject_score, 1)
                })

    if academic_weight > 0:
        academic_raw = academic_total / academic_weight
    else:
        academic_raw = 50

    academic_score = min(academic_raw, 100)

    # Demographic Score
    demographic_total = 0

    location = row.get('LOKASI', 'URBAN')
    if location == 'RURAL':
        location_score = 50
    else:
        location_score = 30
    demographic_total += location_score

    income = row.get('PENDAPATAN', 5000)
    if income < 3000:
        income_score = 50
    elif income < 5000:
        income_score = 40
    elif income < 8000:
        income_score = 30
    else:
        income_score = 20
    demographic_total += income_score

    demographic_score = demographic_total

    # Preference Alignment
    original_choices = []
    for pil in ['PIL1', 'PIL2', 'PIL3']:
        if pil in row.index and pd.notna(row[pil]):
            original_choices.append(str(row[pil]).strip())

    program_name = program['name']
    matched = False
    choice_number = None

    for i, choice in enumerate(original_choices, 1):
        if program_name.lower() in choice.lower() or choice.lower() in program_name.lower():
            matched = True
            choice_number = i
            break

    if matched:
        if choice_number == 1:
            preference_bonus = 15
        elif choice_number == 2:
            preference_bonus = 12
        else:
            preference_bonus = 10
    else:
        preference_bonus = 0

    total_score = (academic_score * 0.8) + (demographic_score * 0.1) + preference_bonus
    total_score = min(total_score, 100)

    return {
        'academic_score': round(academic_score, 1),
        'demographic_score': round(demographic_score, 1),
        'preference_bonus': preference_bonus,
        'total_score': round(total_score, 1),
        'original_choices': original_choices,
        'breakdown': {
            'academic': {
                'subjects': academic_breakdown,
                'calculation': f"Total: {academic_total:.1f} / Weight: {academic_weight:.1f} = {academic_score:.1f}%"
            }
        },
        'weight_formula': f"({academic_score:.1f} × 0.8) + ({demographic_score:.1f} × 0.1) + {preference_bonus} = {round(total_score, 1)}%",
        'formula_explanation': "Suitability = (Academic × 80%) + (Demographic × 10%) + Preference Bonus (max 15)"
    }

# ============================================
# GENERATE EXPLANATION
# ============================================
def generate_explanation(row, program):
    group = program.get('group', 0)
    reasons = []

    def get_grade(subject):
        return grade_to_numeric(row.get(subject, 0))

    if group == 7:
        if get_grade('M-T') >= 75:
            reasons.append(f"Add Math {row.get('M-T', '')} (≥B)")
        if get_grade('FIZ') >= 75:
            reasons.append(f"Physics {row.get('FIZ', '')} (≥B)")
        if get_grade('KIM') >= 75:
            reasons.append(f"Chemistry {row.get('KIM', '')} (≥B)")
        if reasons:
            return "Eligible for Engineering Foundation: " + ", ".join(reasons[:3])
        return "Eligible for Engineering Foundation"
    elif group == 6:
        if get_grade('ACC') >= 75:
            reasons.append(f"ACC {row.get('ACC', '')} (≥B)")
        if get_grade('MAT') >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if get_grade('BI') >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        if reasons:
            return "Eligible for Accounting + SAP: " + ", ".join(reasons)
        return "Eligible for Accounting + SAP"
    elif group == 5:
        if get_grade('ACC') >= 60:
            reasons.append(f"ACC {row.get('ACC', '')} (≥C)")
        if get_grade('MAT') >= 40:
            reasons.append(f"Math {row.get('MAT', '')} (≥E)")
        if get_grade('BI') >= 40:
            reasons.append(f"English {row.get('BI', '')} (≥E)")
        if reasons:
            return "Eligible for Accounting Basic: " + ", ".join(reasons)
        return "Eligible for Accounting Basic"
    elif group == 4:
        if get_grade('BI') >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        return "Eligible for English Communication: " + ", ".join(reasons) if reasons else "Eligible for English Communication"
    elif group == 3:
        if get_grade('MAT') >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        subjects_with_C = count_subjects_with_grade(row, ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'TSI', 'AWB', 'BC', 'RGD', 'MUL'], 60)
        if subjects_with_C >= 5:
            reasons.append(f"{subjects_with_C} subjects ≥C")
        return "Eligible for Computer Science Basic: " + ", ".join(reasons) if reasons else "Eligible for CS Basic"
    elif group == 2:
        if get_grade('MAT') >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if get_grade('BI') >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        subjects_with_C = count_subjects_with_grade(row, ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'TSI', 'AWB', 'BC', 'RGD', 'MUL'], 60)
        if subjects_with_C >= 5:
            reasons.append(f"{subjects_with_C} subjects ≥C")
        return "Eligible for  Computer Science/Marketing + Certification: " + ", ".join(reasons) if reasons else "Eligible for CS/Marketing + Certification"
    else:
        if get_grade('BM') >= 60:
            reasons.append(f"BM {row.get('BM', '')} (≥C)")
        subjects_with_C = count_subjects_with_grade(row, ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'TSI', 'AWB', 'BC', 'RGD', 'MUL'], 60)
        if subjects_with_C >= 3:
            reasons.append(f"{subjects_with_C} subjects ≥C (including BM)")
        return "Eligible for General Programs: " + ", ".join(reasons) if reasons else "Eligible for General Programs"

# ============================================
# ALL PROGRAMS
# ============================================
ALL_PROGRAMS = [
    {'name': 'Diploma in Integrated Logistics Management + Chartered Institute of Logistics and Transport', 'group': 1},
    {'name': 'Diploma in Halal Industry + Halal Executive Certification', 'group': 1},
    {'name': 'Diploma in Islamic Finance + Associate Qualification in Islamic Finance', 'group': 1},
    {'name': 'Diploma in Business Studies', 'group': 1},
    {'name': 'Diploma in Business Information Technology', 'group': 1},
    {'name': 'Diploma in International Business', 'group': 1},
    {'name': 'Diploma in Creative Digital Media Production', 'group': 1},
    {'name': 'Diploma in Computer Science + SAS@Certified Specialist: Visual Business Analytics Certification', 'group': 2},
    {'name': 'Diploma in Marketing + Certified Professional Marketer (Asia) Certification', 'group': 2},
    {'name': 'Diploma in Computer Science', 'group': 3},
    {'name': 'Diploma in English Communication + Sijil Penterjemahan Bahasa ITBM', 'group': 4},
    {'name': 'Diploma in Accounting', 'group': 5},
    {'name': 'Diploma in Accounting + SAP S/4HANA Financial Accounting Associates Certification', 'group': 6},
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Teknologi Malaysia', 'group': 7},
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Malaysia Pahang Al-Sultan Abdullah', 'group': 7},
]

# ============================================
# CHECK OFFERED PROGRAM
# ============================================
def check_offered_program(program_ditawar, original_choices):
    if program_ditawar == 'TIDAK DITAWARKAN' or pd.isna(program_ditawar):
        return "❌ Student was not offered any program."

    for i, p in enumerate(original_choices, 1):
        if program_ditawar.lower() in p.lower():
            return f"✅ Program Offered: {program_ditawar} (Choice {i})"

    return f"✅ Offered: {program_ditawar}"

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
            # Convert NOKP to string with 12-digit format
            df['NOKP_STR'] = df['NOKP'].astype(str).str.zfill(12)
            
            # Pad search input to 12 digits
            search_nokp = nokp_input.strip().zfill(12)
            
            student = df[df['NOKP_STR'].str.contains(search_nokp, na=False)]
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
                <table style='width:100%;'>
                    <tr><td style='padding: 6px;'><b>NOKP</b></td><td style='padding: 6px;'>{row['NOKP']}</td>
                    </tr>
                    <tr><td style='padding: 6px;'><b>Name</b></td><td style='padding: 6px;'>{row['NAMA']}</td>
                    </tr>
                    <tr><td style='padding: 6px;'><b>Gender</b></td><td style='padding: 6px;'>{'Female' if row.get('JANTINA')=='P' else 'Male'}</td>
                    </tr>
                    <tr><td style='padding: 6px;'><b>Location</b></td><td style='padding: 6px;'>{row.get('LOKASI', 'N/A')}</td>
                    </tr>
                    <tr><td style='padding: 6px;'><b>Academic Stream</b></td><td style='padding: 6px;'>{row.get('ALIRAN', 'N/A')}</td>
                    </tr>
                    <tr><td style='padding: 6px;'><b>Parental Income</b></td><td style='padding: 6px;'>RM {row.get('PENDAPATAN', 0):,.0f}</td>
                    </tr>
                 </table>
                </div>
                """, unsafe_allow_html=True)

                # ============================================
                # OFFERED PROGRAM STATUS (with explanation)
                # ============================================
                # Get original choices first (need to define here for offered status)
                original_choices_for_status = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        original_choices_for_status.append(str(row[pil]).strip())
                
                if 'KURSUSJAYA' in row.index and pd.notna(row['KURSUSJAYA']):
                    program_offered = str(row['KURSUSJAYA']).strip()
                    
                    # Check if student was offered
                    if program_offered != 'TIDAK DITAWARKAN':
                        # Student was offered - find choice number if available
                        offered_choice = None
                        for i, p in enumerate(original_choices_for_status, 1):
                            if program_offered.lower() in p.lower():
                                offered_choice = i
                                break
                        
                        if offered_choice:
                            st.success(f"✅ **Program Offered:** {program_offered} (Choice {offered_choice})")
                        else:
                            st.success(f"✅ **Program Offered:** {program_offered}")
                    
                    else:
                        # Student was NOT offered - check if eligible for any program
                        # Need to evaluate eligibility first
                        temp_eligible_list = []
                        for prog in ALL_PROGRAMS:
                            eligible, _ = is_eligible(row, prog)
                            temp_eligible_list.append(eligible)
                        eligible_for_any = any(temp_eligible_list)
                        
                        if eligible_for_any:
                            st.info(f"""
                            ❌ **Not Offered**
                            
                            Student met the eligibility requirements but was not offered a place.
                            This may be due to:
                            • Limited quota availability
                            • Program capacity being fully filled
                            • Competitive selection among eligible candidates
                            
                            *The system recommendations below show programs the student is eligible for.*
                            """)
                        else:
                            st.warning(f"""
                            ❌ **Not Offered**
                            
                            Student does not meet the minimum requirements for any program.
                            """)

                # SPM Subjects
                st.markdown("### 📚 SPM Subjects")
                subject_data = []
                for code, name in SUBJECT_NAMES.items():
                    if code in row.index:
                        grade = row.get(code)
                        if pd.notna(grade) and grade != 'NA' and grade != '':
                            subject_data.append({"Subject": name, "Grade": grade})

                if subject_data:
                    st.dataframe(pd.DataFrame(subject_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No subject data found")

                with st.expander("ℹ️ How Score is Calculated"):
                    st.markdown("""
                    **Total Suitability = (Academic × 80%) + (Demographic × 10%) + Preference Bonus**

                    **Academic Score (Eligibility Level):** Based on SPM grades in relevant subjects
                    
                    **Demographic Score:** Location (Rural 50/Urban 30) + Income (B40 50, M40 40/30, T20 20)
                    
                    **Preference Bonus:** 1st choice +15, 2nd +12, 3rd +10

                    **Suitability Levels:** 
                    🟢 ≥80% Highly Suitable  
                    🟡 60-79% Moderately Suitable 
                    🔴 <60% Less Suitable
                    """)

            with col_right:
                original_choices = []
                for pil in ['PIL1', 'PIL2', 'PIL3']:
                    if pil in row.index and pd.notna(row[pil]):
                        original_choices.append(str(row[pil]).strip())

                # Build recommendations with scores
                all_programs_with_scores = []
                for prog in ALL_PROGRAMS:
                    eligible, reason = is_eligible(row, prog)
                    if eligible:
                        detailed = calculate_detailed_score(row, prog)
                        in_original = any(prog['name'].lower() in p.lower() for p in original_choices)
                        all_programs_with_scores.append({
                            'name': prog['name'],
                            'group': prog['group'],
                            'eligible': True,
                            'in_original': in_original,
                            'academic_score': detailed['academic_score'],
                            'demographic_score': detailed['demographic_score'],
                            'preference_bonus': detailed['preference_bonus'],
                            'total_score': detailed['total_score'],
                            'detailed': detailed,
                            'explanation': generate_explanation(row, prog),
                            'reason': ""
                        })
                    else:
                        all_programs_with_scores.append({
                            'name': prog['name'],
                            'group': prog['group'],
                            'eligible': False,
                            'in_original': False,
                            'academic_score': 0,
                            'demographic_score': 0,
                            'preference_bonus': 0,
                            'total_score': 0,
                            'detailed': None,
                            'explanation': "",
                            'reason': reason
                        })

                # SORT by Total Suitability Score
                all_programs_with_scores.sort(key=lambda x: -x['total_score'])

                # Original choices table
                st.markdown("### 📋 Student's Original Choices")
                top_recommendations = [p['name'] for p in all_programs_with_scores[:10] if p['eligible']]
                choice_rows = []
                for i, p in enumerate(original_choices, 1):
                    in_list = any(p.lower() in rec.lower() for rec in top_recommendations)
                    choice_rows.append([f"Choice {i}", p, "✅" if in_list else "❌"])

                if choice_rows:
                    st.dataframe(pd.DataFrame(choice_rows, columns=["Choice", "Program", "In Top 10?"]), use_container_width=True, hide_index=True)
                else:
                    st.info("No original choices recorded")

                # ========================================
                # PROGRAM RECOMMENDATIONS
                # ========================================
                st.markdown("### 🎯 Program Recommendations")
                st.caption("Click on any program to see detailed score breakdown")

                eligible_count = len([p for p in all_programs_with_scores if p['eligible']])
                st.caption(f"📊 Showing {eligible_count} eligible programs out of {len(all_programs_with_scores)} total")

                # Display recommendations
                for i, prog in enumerate(all_programs_with_scores, 1):
                    if prog['eligible']:
                        star = "⭐ " if prog['in_original'] else ""

                        # Header with scores (always visible)
                        st.markdown(f"""
                        <div style='border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 8px; background-color: #ffffff;'>
                            <div style='padding: 12px 15px; background-color: #f8f9fa; border-radius: 8px;'>
                                <div style='font-weight: bold; font-size: 1em;'>{i}. {star}{prog['name']}</div>
                                <div style='font-size: 0.85em; margin-top: 4px;'>
                                    <span style='color: #1e88e5;'>🎯 Total: {prog['total_score']}%</span>
                                    <span style='color: #666; margin-left: 12px;'>📊 Eligibility: {prog['academic_score']}%</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Expander for detailed content
                        with st.expander(f"📋 View Details", expanded=False):
                            # Determine suitability level text
                            if prog['total_score'] >= 80:
                                level_text = "🟢 Highly Suitable"
                            elif prog['total_score'] >= 60:
                                level_text = "🟡 Moderately Suitable"
                            else:
                                level_text = "🔴 Less Suitable"

                            st.markdown(f"**{level_text}**")

                            detailed = prog['detailed']
                            
                            # Calculate contribution scores
                            academic_contrib = prog['academic_score'] * 0.8
                            demo_contrib = prog['demographic_score'] * 0.1
                            pref_contrib = prog['preference_bonus']

                            # Formula explanation
                            st.markdown(f"""
                            <div style='margin-bottom: 12px; padding: 8px; background-color: #f8f9fa; border-radius: 6px;'>
                                <p style='margin: 0; font-size: 0.85em;'><b>Formula:</b> {detailed['weight_formula']}</p>
                                <p style='margin: 2px 0 0 0; font-size: 0.75em; color: #666;'>{detailed['formula_explanation']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # ============================================
                            # 1. ACADEMIC CARD
                            # ============================================
                            st.markdown(f"""
                            <div style='background-color: #e8f4fd; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                                    <span style='font-weight: bold; font-size: 1.1em;'>📚 Academic</span>
                                    <span style='font-size: 1.3em; font-weight: bold; color: #1e88e5;'>{prog['academic_score']}%</span>
                                </div>
                                <div style='font-size: 0.75em; color: #666; margin-bottom: 8px;'>Weight: 80%</div>
                                <hr style='margin: 6px 0;'>
                            """, unsafe_allow_html=True)
                            
                            # Subject breakdown with smaller font
                            for subj in detailed['breakdown']['academic']['subjects']:
                                if subj['grade_value'] > 0:
                                    st.markdown(f"""
                                    <div style='display: flex; justify-content: space-between; font-size: 0.75em; margin-bottom: 3px;'>
                                        <span><b>{subj['subject']}</b> ({subj['grade']})</span>
                                        <span><b>{subj['contribution']:.1f}</b> (×{subj['weight']})</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <hr style='margin: 6px 0;'>
                                <div style='font-size: 0.7em; color: #666;'>{detailed['breakdown']['academic']['calculation']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # ============================================
                            # 2. DEMOGRAPHIC CARD
                            # ============================================
                            location = row.get('LOKASI', 'URBAN')
                            if location == 'RURAL':
                                location_display = "LUAR BANDAR"
                            else:
                                location_display = "BANDAR"

                            income = row.get('PENDAPATAN', 5000)
                            if income < 3000:
                                income_category = "B40"
                            elif income < 5000:
                                income_category = "M40"
                            elif income < 8000:
                                income_category = "M40"
                            else:
                                income_category = "T20"

                            st.markdown(f"""
                            <div style='background-color: #fef4e8; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                                    <span style='font-weight: bold; font-size: 1.1em;'>🏠 Demographic</span>
                                    <span style='font-size: 1.3em; font-weight: bold; color: #fb8c00;'>{prog['demographic_score']}%</span>
                                </div>
                                <div style='font-size: 0.75em; color: #666; margin-bottom: 8px;'>Weight: 10%</div>
                                <hr style='margin: 6px 0;'>
                                <div style='font-size: 0.75em; margin-bottom: 3px;'>
                                    <span><b>Location:</b> {location_display}</span>
                                </div>
                                <div style='font-size: 0.75em; margin-bottom: 3px;'>
                                    <span><b>Income:</b> RM {income:,.0f} ({income_category})</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # ============================================
                            # 3. PREFERENCE CARD (with smaller font for original choices)
                            # ============================================
                            pref_bonus = prog['preference_bonus']
                            if pref_bonus > 0:
                                bg_color = "#e8f5e9"
                                border_color = "#4caf50"
                                match_text = "✓ Matches your original choice"
                            else:
                                bg_color = "#fff3e0"
                                border_color = "#ff9800"
                                match_text = "✗ Not in your original choices"

                            st.markdown(f"""
                            <div style='background-color: {bg_color}; padding: 12px; border-radius: 8px; border-left: 4px solid {border_color}; margin-bottom: 10px;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                                    <span style='font-weight: bold; font-size: 1.1em;'>⭐ Preference</span>
                                    <span style='font-size: 1.3em; font-weight: bold; color: {border_color};'>{pref_bonus} / 15</span>
                                </div>
                                <div style='font-size: 0.75em; color: #666; margin-bottom: 8px;'>Weight: 10% (max 15 points)</div>
                                <hr style='margin: 6px 0;'>
                                <div style='font-size: 0.75em;'>{match_text}</div>
                            """, unsafe_allow_html=True)
                            
                            # Original choices with smaller font
                            original_choices_list = detailed.get('original_choices', [])
                            if original_choices_list:
                                st.markdown("<span style='font-size: 0.75em; font-weight: bold;'>Your original choices:</span>", unsafe_allow_html=True)
                                for j, choice in enumerate(original_choices_list, 1):
                                    st.markdown(f"<span style='font-size: 0.7em;'>&nbsp;&nbsp;{j}. {choice}</span>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)

                            # ============================================
                            # SCORE COMPOSITION
                            # ============================================
                            st.markdown("---")
                            st.markdown("### 📊 Score Composition")
                            
                            academic_contrib = prog['academic_score'] * 0.8
                            demo_contrib = prog['demographic_score'] * 0.1
                            pref_contrib = prog['preference_bonus']

                            st.markdown(f"""
                            <div style='margin: 8px 0;'>
                                <div style='display: flex; height: 28px; border-radius: 5px; overflow: hidden;'>
                                    <div style='background-color: #1e88e5; width: 33%; text-align: center; color: white; font-size: 0.7em; line-height: 28px;'>Academic {academic_contrib:.1f}%</div>
                                    <div style='background-color: #fb8c00; width: 34%; text-align: center; color: white; font-size: 0.7em; line-height: 28px;'>Demo {demo_contrib:.1f}%</div>
                                    <div style='background-color: #4caf50; width: 33%; text-align: center; color: white; font-size: 0.7em; line-height: 28px;'>Pref {pref_contrib:.1f}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.caption(f"Total: {prog['total_score']}% = Academic ({prog['academic_score']}% × 0.8) + Demographic ({prog['demographic_score']}% × 0.1) + Preference Bonus ({pref_contrib})")
                            
                            # Brief explanation
                            st.info(f"💡 {prog['explanation']}")

                    else:
                        # Not eligible
                        st.markdown(f"""
                        <div style='margin-bottom: 8px; padding: 10px; border-left: 5px solid #dc3545; border-radius: 6px; background-color: #ffffff; border: 1px solid #e0e0e0;'>
                            <b>{i}. {prog['name']}</b><br>
                            <span style='color: #dc3545; font-size: 0.85em;'>❌ Not eligible: {prog['reason']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Offered Program Display (at bottom - keep existing)
                if 'KURSUSJAYA' in row.index and pd.notna(row['KURSUSJAYA']):
                    program_offered = str(row['KURSUSJAYA']).strip()
                    if program_offered != 'TIDAK DITAWARKAN':
                        offered_msg = check_offered_program(program_offered, original_choices)
                        st.success(offered_msg)
