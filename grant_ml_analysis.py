import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re

# Load the socio data with grant info
socio = pd.read_csv("Datasets/chi_data.csv")

# Load grant data and aggregate by community area
grant_data = pd.read_csv("Datasets/NOF_Small_Projects_2025.csv")

print("Data cleaning in progress...")
print(f"Original grant data columns: {grant_data.columns.tolist()}")

# Clean the grant amount columns - remove dollar signs, commas, and convert to numeric
def clean_currency_amount(amount):
    if pd.isna(amount):
        return 0.0
    if isinstance(amount, (int, float)):
        return float(amount)
    # Remove dollar signs, commas, and any non-numeric characters except decimal points
    cleaned = re.sub(r'[^\d.]', '', str(amount))
    return float(cleaned) if cleaned else 0.0

# Apply cleaning to amount columns
grant_data['INCENTIVE_AMOUNT_CLEAN'] = grant_data['INCENTIVE AMOUNT'].apply(clean_currency_amount)
grant_data['TOTAL_PROJECT_COST_CLEAN'] = grant_data['TOTAL PROJECT COST'].apply(clean_currency_amount)

print(f"Sample cleaned incentive amounts: {grant_data['INCENTIVE_AMOUNT_CLEAN'].head()}")
print(f"Sample cleaned project costs: {grant_data['TOTAL_PROJECT_COST_CLEAN'].head()}")

# Clean and aggregate grant data by community area
grant_data['COMMUNITY AREA'] = grant_data['COMMUNITY AREA'].astype(str).str.strip().str.lower()

# Calculate grant metrics
grants_by_ca = grant_data.groupby('COMMUNITY AREA').agg({
    'PROJECT NAME': 'count',
    'INCENTIVE_AMOUNT_CLEAN': 'sum',
    'TOTAL_PROJECT_COST_CLEAN': 'sum'
}).reset_index()

grants_by_ca.columns = ['community_area', 'num_grants', 'total_grant_amount', 'total_project_cost']

print(f"\nGrant summary by community area:")
print(f"Total grants: {grants_by_ca['num_grants'].sum()}")
print(f"Total grant amount: ${grants_by_ca['total_grant_amount'].sum():,.2f}")
print(f"Communities with grants: {len(grants_by_ca)}")

# Clean community area names for merging
grants_by_ca['community_area'] = grants_by_ca['community_area'].str.strip().str.lower()
socio['community_area'] = socio['community_area'].astype(str).str.strip().str.lower()

print(f"\nSocio data community areas sample: {socio['community_area'].head().tolist()}")
print(f"Grant data community areas sample: {grants_by_ca['community_area'].head().tolist()}")

# Merge socio data with grant data
merged = socio.merge(
    grants_by_ca[['community_area', 'num_grants', 'total_grant_amount', 'total_project_cost']],
    on='community_area',
    how='left'
)

print(f"\nAfter merge - dataset shape: {merged.shape}")
print(f"Neighborhoods with grants: {merged['num_grants'].notna().sum()}")

# Fill NaN values for neighborhoods with no grants
merged['num_grants'] = merged['num_grants'].fillna(0)
merged['total_grant_amount'] = merged['total_grant_amount'].fillna(0)
merged['total_project_cost'] = merged['total_project_cost'].fillna(0)

# Ensure total_hh is numeric
merged['total_hh'] = pd.to_numeric(merged['total_hh'], errors='coerce')

# Create target variable: grants per 1,000 households
merged['grants_per_1k_hh'] = (merged['num_grants'] / merged['total_hh']) * 1000

# Alternative target: grant amount per household
merged['grant_amount_per_hh'] = merged['total_grant_amount'] / merged['total_hh']

print(f"\nTarget variable summary:")
print(f"Grants per 1k HH - Mean: {merged['grants_per_1k_hh'].mean():.2f}, Max: {merged['grants_per_1k_hh'].max():.2f}")
print(f"Grant amount per HH - Mean: ${merged['grant_amount_per_hh'].mean():.2f}")

print("=" * 60)
print("MACHINE LEARNING ANALYSIS: PREDICTING GRANT ACCESS")
print("=" * 60)

# Selecting features for prediction
feature_cols = [
    'white_hh_share', 'black_hh_share', 'latino_hh_share', 'asian_hh_share',
    'ami_shr', 'total_hh', 'white_own_shr', 'black_own_shr', 'latino_own_shr'
]

# Check if all feature columns exist and are numeric
print("\nChecking feature columns:")
for col in feature_cols:
    if col in merged.columns:
        dtype = merged[col].dtype
        missing = merged[col].isna().sum()
        print(f"  {col}: {dtype}, missing: {missing}")
    else:
        print(f"  {col}: COLUMN NOT FOUND")

# Convert feature columns to numeric, coercing errors
for col in feature_cols:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Using grants per 1k households as primary target
target_col = 'grants_per_1k_hh'

# Removing any rows with missing data
df_clean = merged[feature_cols + [target_col]].dropna()

print(f"\nDataset: {len(df_clean)} neighborhoods with complete data")
print(f"Features: {len(feature_cols)}")
print(f"Target: {target_col}")

if len(df_clean) == 0:
    print("ERROR: No data remaining after cleaning. Check your feature columns.")
    exit()

X = df_clean[feature_cols]
y = df_clean[target_col]

# Splitting data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} neighborhoods")
print(f"Test set: {len(X_test)} neighborhoods")

# Model 1: Linear Regression
print("\n" + "=" * 60)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 60)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions on test set
y_pred_lr = lr_model.predict(X_test)

# Evaluating model performance
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)

print(f"\nR² Score: {lr_r2:.3f}")
print(f"  -> {lr_r2*100:.1f}% of variance in grant access is explained by demographics")
print(f"\nRMSE: {lr_rmse:.2f} grants per 1k households")
print(f"MAE: {lr_mae:.2f} grants per 1k households")

# Feature importance for linear regression (coefficients)
print("\nFeature Importance (Linear Regression Coefficients):")
feature_importance_lr = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for idx, row in feature_importance_lr.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"  {row['Feature']:25s}: {row['Coefficient']:7.3f} {direction}")

# Model 2: Random Forest
print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print(f"\nR² Score: {rf_r2:.3f}")
print(f"  -> {rf_r2*100:.1f}% of variance explained")
print(f"\nRMSE: {rf_rmse:.2f} grants per 1k households")
print(f"MAE: {rf_mae:.2f} grants per 1k households")

# Feature importance for random forest
print("\nFeature Importance (Random Forest):")
feature_importance_rf = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance_rf.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"  {row['Feature']:25s}: {row['Importance']:.3f} {bar}")

# Cross-validation to check model stability
print("\n" + "=" * 60)
print("CROSS-VALIDATION (5-FOLD)")
print("=" * 60)

cv_scores_lr = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

print(f"\nLinear Regression:")
print(f"  Mean R²: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std() * 2:.3f})")

print(f"\nRandom Forest:")
print(f"  Mean R²: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std() * 2:.3f})")

# Comparison with loan predictability
print("\n" + "=" * 60)
print("COMPARISON WITH LOAN ANALYSIS")
print("=" * 60)

# Loan R² values from your friend's analysis
loan_rf_r2 = 0.684
loan_lr_r2 = 0.472

print(f"\nGrant Predictability (Random Forest): {rf_r2:.3f}")
print(f"Loan Predictability (Random Forest): {loan_rf_r2:.3f}")
print(f"Difference: {rf_r2 - loan_rf_r2:.3f}")

print(f"\nGrant Predictability (Linear): {lr_r2:.3f}")
print(f"Loan Predictability (Linear): {loan_lr_r2:.3f}")
print(f"Difference: {lr_r2 - loan_lr_r2:.3f}")

# Create visualizations directory if it doesn't exist
import os
os.makedirs('visualizations', exist_ok=True)

# Visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Viz 1: Feature Importance Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Linear regression coefficients
feature_importance_lr_plot = feature_importance_lr.copy()
colors_lr = ['steelblue' if x > 0 else 'coral' for x in feature_importance_lr_plot['Coefficient']]
ax1.barh(feature_importance_lr_plot['Feature'], feature_importance_lr_plot['Coefficient'], color=colors_lr, alpha=0.8)
ax1.set_xlabel('Coefficient', fontsize=12)
ax1.set_title('Linear Regression: Feature Coefficients\n(Grant Access)', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Random forest importance
ax2.barh(feature_importance_rf['Feature'], feature_importance_rf['Importance'], color='green', alpha=0.8)
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Random Forest: Feature Importance\n(Grant Access)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/grant_ml_viz_1_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/grant_ml_viz_1_feature_importance.png")

# Viz 2: Predictions vs Actual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Linear regression predictions
ax1.scatter(y_test, y_pred_lr, alpha=0.6, s=100, edgecolors='black', linewidth=0.5, color='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Grants per 1k HH', fontsize=12)
ax1.set_ylabel('Predicted Grants per 1k HH', fontsize=12)
ax1.set_title(f'Linear Regression (R² = {lr_r2:.3f})', fontsize=14, fontweight='bold')

# Random forest predictions
ax2.scatter(y_test, y_pred_rf, alpha=0.3, s=100, edgecolors='black', linewidth=0.5, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Grants per 1k HH', fontsize=12)
ax2.set_ylabel('Predicted Grants per 1k HH', fontsize=12)
ax2.set_title(f'Random Forest (R² = {rf_r2:.3f})', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/grant_ml_viz_2_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/grant_ml_viz_2_predictions.png")

# Viz 3: Comparison with Loans
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

categories = ['Linear Regression', 'Random Forest']
grant_r2 = [lr_r2, rf_r2]
loan_r2 = [loan_lr_r2, loan_rf_r2]

x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, grant_r2, width, label='Grants', color='green', alpha=0.7)
ax.bar(x + width/2, loan_r2, width, label='Loans', color='red', alpha=0.7)

ax.set_xlabel('Model Type', fontsize=12)
ax.set_ylabel('R² Score (Predictability)', fontsize=12)
ax.set_title('Grant vs Loan Predictability from Demographics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/grant_loan_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/grant_loan_comparison.png")

print("\n" + "=" * 60)
print("KEY FINDINGS - GRANT ANALYSIS")
print("=" * 60)

# Interpretation based on results
if rf_r2 < 0.3:
    predictability_desc = "WEAKLY predictable"
    implication = "Grant distribution appears relatively EQUITABLE across demographics"
elif rf_r2 < 0.6:
    predictability_desc = "MODERATELY predictable" 
    implication = "Some demographic patterns in grant access, but not strongly systematic"
else:
    predictability_desc = "HIGHLY predictable"
    implication = "Grant distribution shows SYSTEMATIC disparities by demographics"

print(f"""
1. Grant predictability from demographics:
   - Random Forest R²: {rf_r2:.3f} → {predictability_desc}
   - Linear Regression R²: {lr_r2:.3f}

2. Comparison with loans:
   - Grant predictability: {rf_r2:.3f}
   - Loan predictability: {loan_rf_r2:.3f}
   - Difference: {rf_r2 - loan_rf_r2:.3f} 
     ({'GRANTS more equitable' if rf_r2 < loan_rf_r2 else 'LOANS more equitable'})

3. Most Important Grant Predictors:
   - {feature_importance_rf.iloc[0]['Feature']}: {feature_importance_rf.iloc[0]['Importance']:.1%}
   - {feature_importance_rf.iloc[1]['Feature']}: {feature_importance_rf.iloc[1]['Importance']:.1%}
   - {feature_importance_rf.iloc[2]['Feature']}: {feature_importance_rf.iloc[2]['Importance']:.1%}

4. Key Implications:
   - {implication}
   - Grant programs appear {'MORE equitable' if rf_r2 < loan_rf_r2 else 'LESS equitable'} than loan programs
   - This suggests {'structural barriers affect loans more than grants' if rf_r2 < loan_rf_r2 else 'both grants and loans show systematic disparities'}
""")

print("=" * 60)
print("GRANT ANALYSIS COMPLETE")
print("=" * 60)
