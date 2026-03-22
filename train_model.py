# ============================================
# TRAIN MARA RECOMMENDATION MODEL
# ============================================
# Run this script FIRST to train and save the model
# Output: mara_model.pkl

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING MARA RECOMMENDATION MODEL")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n1. Loading data...")
df = pd.read_csv('data_lengkap.csv')
print(f"   Loaded {len(df)} records")

# ============================================
# 2. CREATE TARGET VARIABLE
# ============================================
print("\n2. Creating target variable...")
target_col = 'KURSUSJAYA'
df['TARGET'] = (df[target_col] != 'TIDAK DITAWARKAN').astype(int)
print(f"   Class distribution:")
print(f"   - Not Offered (0): {(df['TARGET']==0).sum()}")
print(f"   - Offered (1): {(df['TARGET']==1).sum()}")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
print("\n3. Feature engineering...")

# Grade mapping
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

# Convert SPM subjects to numeric
subject_columns = ['BM', 'BI', 'SEJ', 'MAT', 'M-T', 'FIZ', 'KIM', 'BIO', 
                   'ACC', 'PT', 'EKO', 'SK', 'PI', 'PQS', 'PSI', 'BAT']

for col in subject_columns:
    if col in df.columns:
        df[f'{col}_NUM'] = df[col].apply(grade_to_numeric)

# Create feature set
feature_cols = []
for col in subject_columns:
    if f'{col}_NUM' in df.columns:
        feature_cols.append(f'{col}_NUM')

# Add demographic features
if 'JANTINA' in df.columns:
    df['JANTINA_BINARY'] = (df['JANTINA'] == 'P').astype(int)
    feature_cols.append('JANTINA_BINARY')

if 'LOKASI' in df.columns:
    df['LOKASI_RURAL'] = (df['LOKASI'] == 'RURAL').astype(int)
    feature_cols.append('LOKASI_RURAL')

if 'ALIRAN' in df.columns:
    df['ALIRAN_STEM'] = (df['ALIRAN'] == 'STEM').astype(int)
    feature_cols.append('ALIRAN_STEM')

if 'PENDAPATAN' in df.columns:
    # Normalize income (divide by 10000)
    df['PENDAPATAN_NORM'] = df['PENDAPATAN'].fillna(5000) / 10000
    feature_cols.append('PENDAPATAN_NORM')

# Create cluster eligibility scores (simplified)
# Engineering eligibility
if 'M-T_NUM' in df.columns and 'FIZ_NUM' in df.columns:
    df['ENGINEERING_SCORE'] = (
        (df['M-T_NUM'] / 100) * 0.35 +
        (df['FIZ_NUM'] / 100) * 0.35 +
        (df['MAT_NUM'] / 100) * 0.30
    )
    feature_cols.append('ENGINEERING_SCORE')

# Accounting eligibility
if 'ACC_NUM' in df.columns and 'MAT_NUM' in df.columns:
    df['ACCOUNTING_SCORE'] = (
        (df['ACC_NUM'] / 100) * 0.50 +
        (df['MAT_NUM'] / 100) * 0.50
    )
    feature_cols.append('ACCOUNTING_SCORE')

# Computer Science eligibility
if 'MAT_NUM' in df.columns and 'BI_NUM' in df.columns:
    df['CS_SCORE'] = (
        (df['MAT_NUM'] / 100) * 0.40 +
        (df['BI_NUM'] / 100) * 0.30 +
        (df['BM_NUM'] / 100) * 0.30
    )
    feature_cols.append('CS_SCORE')

print(f"   Created {len(feature_cols)} features")

# ============================================
# 4. PREPARE DATA FOR MODEL
# ============================================
print("\n4. Preparing data...")

# Handle missing values
X = df[feature_cols].fillna(0)
y = df['TARGET']

# Remove rows with all zeros (no subject data)
X = X[(X.sum(axis=1) > 0)]
y = y[X.index]

print(f"   Final samples: {len(X)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# ============================================
# 5. TRAIN RANDOM FOREST MODEL
# ============================================
print("\n5. Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ============================================
# 6. EVALUATE MODEL
# ============================================
print("\n6. Evaluating model...")

y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"   Test Accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

# ============================================
# 7. SAVE MODEL
# ============================================
print("\n7. Saving model...")

# Save model
joblib.dump(model, 'mara_model.pkl')
print("   ✅ Model saved as 'mara_model.pkl'")

# Save feature names for reference
with open('feature_names.txt', 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")
print("   ✅ Feature names saved as 'feature_names.txt'")

# Save model info
model_info = {
    'accuracy': accuracy,
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
    'feature_importance': feature_importance.to_dict()
}
joblib.dump(model_info, 'model_info.pkl')
print("   ✅ Model info saved as 'model_info.pkl'")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nYou can now run: streamlit run app.py")
