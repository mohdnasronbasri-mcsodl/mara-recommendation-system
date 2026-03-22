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
# HELPER FUNCTION: COUNT SUBJECTS WITH MIN GRADE
# ============================================
def count_subjects_with_grade(row, subjects, min_grade):
    """Count how many subjects have grade >= min_grade"""
    count = 0
    for subj in subjects:
        if subj in row.index:
            grade_val = grade_to_numeric(row.get(subj, 0))
            if grade_val >= min_grade:
                count += 1
    return count

# ============================================
# CHECK ELIGIBILITY (UPDATED WITH NEW REQUIREMENTS)
# ============================================
def is_eligible(row, program):
    """
    Check if student meets minimum requirements for a program.
    Returns (True/False, reason_if_not_eligible)
    """
    group = program.get('group', 1)
    
    # Helper function to get numeric grade
    def get_grade(subject):
        return grade_to_numeric(row.get(subject, 0))
    
    # ============================================
    # GROUP 1: General Programs
    # ============================================
    if group == 1:
        # Check BM at least C
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        
        # Check Math, English, History at least E
        if get_grade('MAT') < 40:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ E / 40)"
        if get_grade('BI') < 40:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        
        # Count subjects with C
        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)
        
        if subjects_with_C < 3:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 3, including BM)"
        
        return True, ""
    
    # ============================================
    # GROUP 2: CS/Marketing + Certification
    # ============================================
    elif group == 2:
        # Check BM at least C
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        
        # Check Math and English at least B
        if get_grade('MAT') < 75:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ B / 75)"
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        
        # Check History at least E
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        
        # Count subjects with C
        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)
        
        if subjects_with_C < 5:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 5, including BM)"
        
        return True, ""
    
    # ============================================
    # GROUP 3: Computer Science (Basic)
    # ============================================
    elif group == 3:
        # Check BM and Math at least C
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('MAT') < 60:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ C / 60)"
        
        # Check English and History at least E
        if get_grade('BI') < 40:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        
        # Count subjects with C
        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)
        
        if subjects_with_C < 5:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 5, including BM and Mathematics)"
        
        return True, ""
    
    # ============================================
    # GROUP 4: English Communication
    # ============================================
    elif group == 4:
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        if get_grade('MAT') < 40:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        
        # Count other subjects with C
        other_subjects = ['M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT']
        other_with_C = count_subjects_with_grade(row, other_subjects, 60)
        
        if other_with_C < 1:
            return False, f"Need at least 1 other subject with grade ≥ C (you have {other_with_C})"
        
        return True, ""
    
    # ============================================
    # GROUP 5: Accounting (Basic)
    # ============================================
    elif group == 5:
        # Check BM at least C
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        
        # Check English at least B
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        
        # Check Math and History at least E
        if get_grade('MAT') < 40:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ E / 40)"
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        
        # Count subjects with C
        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)
        
        if subjects_with_C < 5:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 5, including BM)"
        
        return True, ""
    
    # ============================================
    # GROUP 6: Accounting + SAP Certification
    # ============================================
    elif group == 6:
        # Check BM at least C
        if get_grade('BM') < 60:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ C / 60)"
        
        # Check Math and English at least B
        if get_grade('MAT') < 75:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ B / 75)"
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        
        # Check History at least E
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        
        # Count subjects with C
        subjects_to_check = ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT']
        subjects_with_C = count_subjects_with_grade(row, subjects_to_check, 60)
        
        if subjects_with_C < 3:
            return False, f"Only {subjects_with_C} subjects with grade ≥ C (need at least 3, including BM)"
        
        return True, ""
    
    # ============================================
    # GROUP 7: Engineering Foundation
    # ============================================
    elif group == 7:
        # Check BM and Math at least A-
        if get_grade('BM') < 85:
            return False, f"BM: {row.get('BM', 'N/A')} (need ≥ A- / 85)"
        if get_grade('MAT') < 85:
            return False, f"Mathematics: {row.get('MAT', 'N/A')} (need ≥ A- / 85)"
        
        # Check History at least E
        if get_grade('SEJ') < 40:
            return False, f"History: {row.get('SEJ', 'N/A')} (need ≥ E / 40)"
        
        # Check English at least B
        if get_grade('BI') < 75:
            return False, f"English: {row.get('BI', 'N/A')} (need ≥ B / 75)"
        
        # Check Additional Mathematics at least B
        if get_grade('M-T') < 75:
            return False, f"Additional Mathematics: {row.get('M-T', 'N/A')} (need ≥ B / 75)"
        
        # Check Physics OR Chemistry at least B
        fizik = get_grade('FIZ')
        kim = get_grade('KIM')
        if max(fizik, kim) < 75:
            return False, f"Physics/Chemistry: best = {max(fizik, kim)} (need at least one ≥ B / 75)"
        
        # Count subjects with B (need at least 5)
        subjects_to_check_b = ['BI', 'M-T', 'FIZ', 'KIM', 'MAT', 'BM', 'BIO', 'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT', 'SEJ']
        subjects_with_B = count_subjects_with_grade(row, subjects_to_check_b, 75)
        
        if subjects_with_B < 5:
            return False, f"Only {subjects_with_B} subjects with grade ≥ B (need at least 5)"
        
        return True, ""
    
    # Default: not eligible
    return False, "Program requirements not defined"

# ============================================
# CALCULATE SCORE (BASIC) - NOT USED FOR SORTING
# ============================================
def calculate_score(row, program):
    score = 0
    total_weight = 0
    group = program.get('group', 1)
    
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
    
    # Define relevant subjects based on group
    if group == 7:
        relevant = ['BM', 'MAT', 'M-T', 'FIZ', 'KIM', 'BI']
    elif group == 6:
        relevant = ['BM', 'MAT', 'BI', 'ACC']
    elif group == 5:
        relevant = ['BM', 'MAT', 'BI', 'ACC']
    elif group == 4:
        relevant = ['BM', 'BI', 'MAT']
    elif group in [2, 3]:
        relevant = ['BM', 'MAT', 'BI', 'SK']
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
    
    # Group priority bonus
    priority_bonus = {7: 20, 6: 15, 2: 12, 3: 10, 4: 8, 5: 5, 1: 0}
    bonus = priority_bonus.get(group, 0)
    base_score = min(base_score + bonus, 100)
    
    return round(base_score, 1)

# ============================================
# XAI: DETAILED SCORE BREAKDOWN
# ============================================
def calculate_detailed_score(row, program):
    """
    Calculate suitability score with detailed breakdown for XAI.
    Returns academic_score, demographic_score, preference_bonus, total_score
    """
    
    group = program.get('group', 1)
    weights = GROUP_SUBJECT_WEIGHTS.get(group, GROUP_SUBJECT_WEIGHTS[1])
    
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
    
    # Demographic Score
    demographic_breakdown = []
    demographic_total = 0
    
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
    
    income = row.get('PENDAPATAN', 5000)
    if income < 3000:
        income_score = 50
        income_category = 'B40 (Low Income)'
        income_note = 'B40: maximum priority to support students from low-income families'
    elif income < 5000:
        income_score = 40
        income_category = 'M40 (Lower Middle)'
        income_note = 'M40: high priority'
    elif income < 8000:
        income_score = 30
        income_category = 'M40 (Upper Middle)'
        income_note = 'M40: moderate priority'
    else:
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
    
    # Total Score
    total_score = (academic_score * 0.8) + (demographic_score * 0.1) + preference_bonus
    total_score = min(total_score, 100)
    
    return {
        'academic_score': round(academic_score, 1),
        'demographic_score': round(demographic_score, 1),
        'preference_bonus': preference_bonus,
        'total_score': round(total_score, 1),
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
        if get_grade('BM') >= 85:
            reasons.append(f"BM {row.get('BM', '')} (≥A-)")
        if get_grade('MAT') >= 85:
            reasons.append(f"Math {row.get('MAT', '')} (≥A-)")
        if reasons:
            return "Eligible for Engineering Foundation: " + ", ".join(reasons[:4])
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
        if get_grade('BI') >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        if reasons:
            return "Eligible for Accounting Basic: " + ", ".join(reasons)
        return "Eligible for Accounting Basic"
    
    elif group == 4:
        if get_grade('BI') >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        if get_grade('BM') >= 60:
            reasons.append(f"BM {row.get('BM', '')} (≥C)")
        if reasons:
            return "Eligible for English Communication: " + ", ".join(reasons)
        return "Eligible for English Communication"
    
    elif group == 3:
        if get_grade('MAT') >= 60:
            reasons.append(f"Math {row.get('MAT', '')} (≥C)")
        if get_grade('SK') >= 60:
            reasons.append(f"Computer Science {row.get('SK', '')} (≥C)")
        subjects_with_C = count_subjects_with_grade(row, ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC'], 60)
        if subjects_with_C >= 5:
            reasons.append(f"{subjects_with_C} subjects with grade ≥C")
        if reasons:
            return "Eligible for CS Basic: " + ", ".join(reasons)
        return "Eligible for CS Basic"
    
    elif group == 2:
        if get_grade('MAT') >= 75:
            reasons.append(f"Math {row.get('MAT', '')} (≥B)")
        if get_grade('BI') >= 75:
            reasons.append(f"English {row.get('BI', '')} (≥B)")
        if get_grade('SK') >= 60:
            reasons.append(f"Computer Science {row.get('SK', '')} (≥C)")
        subjects_with_C = count_subjects_with_grade(row, ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC'], 60)
        if subjects_with_C >= 5:
            reasons.append(f"{subjects_with_C} subjects with grade ≥C")
        if reasons:
            return "Eligible for CS/Marketing + Certification: " + ", ".join(reasons)
        return "Eligible for CS/Marketing + Certification"
    
    else:  # Group 1
        if get_grade('BM') >= 60:
            reasons.append(f"BM {row.get('BM', '')} (≥C)")
        subjects_with_C = count_subjects_with_grade(row, ['BM', 'MAT', 'BI', 'SEJ', 'M-T', 'FIZ', 'KIM', 'BIO', 'ACC'], 60)
        if subjects_with_C >= 3:
            reasons.append(f"{subjects_with_C} subjects with grade ≥C (including BM)")
        return "Eligible for General Programs: " + ", ".join(reasons) if reasons else "Eligible for General Programs"

# ============================================
# ALL PROGRAMS (UPDATED WITH GROUP NUMBERS)
# ============================================
ALL_PROGRAMS = [
    # GROUP 1 - General Programs
    {'name': 'Diploma in Integrated Logistics Management + Chartered Institute of Logistics and Transport', 'group': 1},
    {'name': 'Diploma in Halal Industry + Halal Executive Certification', 'group': 1},
    {'name': 'Diploma in Islamic Finance + Associate Qualification in Islamic Finance', 'group': 1},
    {'name': 'Diploma in Business Studies', 'group': 1},
    {'name': 'Diploma in Business Information Technology', 'group': 1},
    {'name': 'Diploma in International Business', 'group': 1},
    {'name': 'Diploma in Creative Digital Media Production', 'group': 1},
    
    # GROUP 2 - CS/Marketing + Certification
    {'name': 'Diploma in Computer Science + SAS@Certified Specialist: Visual Business Analytics Certification', 'group': 2},
    {'name': 'Diploma in Marketing + Certified Professional Marketer (Asia) Certification', 'group': 2},
    
    # GROUP 3 - Computer Science (Basic)
    {'name': 'Diploma in Computer Science', 'group': 3},
    
    # GROUP 4 - English Communication
    {'name': 'Diploma in English Communication + Sijil Penterjemahan Bahasa ITBM', 'group': 4},
    
    # GROUP 5 - Accounting (Basic)
    {'name': 'Diploma in Accounting', 'group': 5},
    
    # GROUP 6 - Accounting + SAP Certification
    {'name': 'Diploma in Accounting + SAP S/4HANA Financial Accounting Associates Certification', 'group': 6},
    
    # GROUP 7 - Engineering Foundation
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Teknologi Malaysia', 'group': 7},
    {'name': 'Asasi Kejuruteraan & Teknologi - Universiti Malaysia Pahang Al-Sultan Abdullah', 'group': 7},
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
                    <tr><td style='padding: 6px;'><b>NOKP</b></td><td style='padding: 6px;'>{row['NOKP']}</td>
                    <tr><td style='padding: 6px;'><b>Name</b></td><td style='padding: 6px;'>{row['NAMA']}</td>
                    <tr><td style='padding: 6px;'><b>Gender</b></td><td style='padding: 6px;'>{'Female' if row.get('JANTINA')=='P' else 'Male'}</td>
                    <tr><td style='padding: 6px;'><b>Location</b></td><td style='padding: 6px;'>{row.get('LOKASI', 'N/A')}</td>
                    <tr><td style='padding: 6px;'><b>Academic Stream</b></td><td style='padding: 6px;'>{row.get('ALIRAN', 'N/A')}</td>
                    <tr><td style='padding: 6px;'><b>Parental Income</b></td><td style='padding: 6px;'>RM {row.get('PENDAPATAN', 0):,.0f}</td>
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
                            numeric = grade_to_numeric(grade)
                            if numeric >= 85:
                                grade_display = f"🟢 {grade}"
                            elif numeric >= 75:
                                grade_display = f"🔵 {grade}"
                            elif numeric >= 60:
                                grade_display = f"🟡 {grade}"
                            else:
                                grade_display = f"🔴 {grade}"
                            subject_data.append({"Subject": name, "Grade": grade_display})
                
                if subject_data:
                    df_subjects = pd.DataFrame(subject_data)
                    st.dataframe(df_subjects, use_container_width=True, hide_index=True)
                else:
                    st.info("No subject data found")
                
                # Score Details Info
                with st.expander("ℹ️ How Score is Calculated"):
                    st.markdown("""
                    ### Suitability Score Formula
                    
                    **Total Suitability = (Academic × 80%) + (Demographic × 10%) + Preference Bonus**
                    
                    ---
                    **1. Academic Score (Eligibility Level)**
                    - Based on SPM grades in relevant subjects
                    - Different subjects have different weights per program
                    - Maximum: 100%
                    
                    **2. Demographic Score**
                    - Location: Rural (50) / Urban (30)
                    - Income: B40 (50), M40 lower (40), M40 upper (30), T20 (20)
                    - Maximum: 100%
                    
                    **3. Preference Bonus**
                    - Matches 1st choice: +15
                    - Matches 2nd choice: +12
                    - Matches 3rd choice: +10
                    - No match: 0
                    
                    ---
                    **Suitability Levels:**
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
                
                # Evaluate ALL programs to build recommendation list with detailed scores
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
                
                # SORT by Total Suitability Score (highest to lowest)
                all_programs_with_scores.sort(key=lambda x: -x['total_score'])
                
                # Get top recommendations for original choices check
                top_recommendations = [p['name'] for p in all_programs_with_scores[:10] if p['eligible']]
                
                # Display original choices table
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
                # PROGRAM RECOMMENDATIONS
                # ========================================
                st.markdown("### 🎯 Program Recommendations")
                st.caption("Click on any program to see detailed score breakdown")
                
                eligible_count = len([p for p in all_programs_with_scores if p['eligible']])
                st.caption(f"📊 Showing {eligible_count} eligible programs out of {len(all_programs_with_scores)} total")
                
                # Display recommendations with expanders
                for i, prog in enumerate(all_programs_with_scores, 1):
                    if prog['eligible']:
                        star = "⭐ " if prog['in_original'] else ""
                        
                        # Create expander with custom HTML header
                        with st.expander(""):
                            # Custom header with name left, scores right
                            st.markdown(f"""
                            <div style='display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 10px;'>
                                <span style='font-weight: bold; font-size: 1em;'>{i}. {star}{prog['name']}</span>
                                <div style='text-align: right;'>
                                    <span style='color: #1e88e5; font-weight: bold;'>🎯 {prog['total_score']}%</span><br>
                                    <span style='font-size: 0.75em; color: #666;'>📊 Eligibility: {prog['academic_score']}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Determine suitability level based on TOTAL score
                            if prog['total_score'] >= 80:
                                level_emoji = "🟢"
                                level_text = "Highly Suitable"
                            elif prog['total_score'] >= 60:
                                level_emoji = "🟡"
                                level_text = "Moderately Suitable"
                            else:
                                level_emoji = "🔴"
                                level_text = "Less Suitable"
                            
                            # Display status
                            st.markdown(f"""
                            <div style='margin-bottom: 15px;'>
                                <span style='font-size: 1.1em; font-weight: bold;'>{level_emoji} {level_text}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Get detailed breakdown
                            detailed = prog['detailed']
                            
                            # Formula explanation
                            st.markdown(f"""
                            <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 8px;'>
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
                                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: #1e88e5;'>{prog['academic_score']}%</p>
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
                                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: #fb8c00;'>{prog['demographic_score']}%</p>
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
                                pref_bonus = prog['preference_bonus']
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
                            academic_contrib = prog['academic_score'] * 0.8
                            demo_contrib = prog['demographic_score'] * 0.1
                            pref_contrib = prog['preference_bonus']
                            
                            # Simple bar using HTML/CSS
                            st.markdown(f"""
                            <div style='margin: 10px 0;'>
                                <div style='display: flex; height: 30px; border-radius: 5px; overflow: hidden;'>
                                    <div style='background-color: #1e88e5; width: {academic_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Academic {academic_contrib:.1f}%</div>
                                    <div style='background-color: #fb8c00; width: {demo_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Demo {demo_contrib:.1f}%</div>
                                    <div style='background-color: #4caf50; width: {pref_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Pref {pref_contrib:.1f}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.caption(f"Total: {prog['total_score']}% = Academic ({prog['academic_score']}% × 0.8) + Demographic ({prog['demographic_score']}% × 0.1) + Preference Bonus ({pref_bonus})")
                            
                            # Brief explanation
                            st.info(f"💡 **Why this score?** {prog['explanation']}")
                    
                    else:
                        # Not eligible
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
