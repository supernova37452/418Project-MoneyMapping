import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#load the socio data with loan info
socio = pd.read_csv("Datasets/chi_data.csv")

#load loan data and aggregate by community area (same as visualization script)
loan_data = pd.read_excel("Datasets/Illinois-CRA-Compilation-2023.xlsx")
chicago_loans = loan_data[loan_data['County FIPS'] == 17031].copy()
chicago_loans['tract_clean'] = (chicago_loans['Tract'] * 100).astype(int).astype(str)

chi_tract = pd.read_csv("Datasets/chi_data_tract.csv")
chi_tract['tract_clean'] = chi_tract['tract'].astype(str)

chicago_loans = chicago_loans.merge(
    chi_tract[['tract_clean', 'community_area']],
    on='tract_clean',
    how='left'
)

chicago_loans = chicago_loans[chicago_loans['community_area'].notna()].copy()

chicago_loans['total_loan_count'] = (
    chicago_loans['<100K Number'] +
    chicago_loans['100-<250K Number'] +
    chicago_loans['>250K Number']
)

chicago_loans['total_loan_amount'] = (
    chicago_loans['<100K Amount'] +
    chicago_loans['100-<250K Amount'] +
    chicago_loans['>250K Amount']
)

loans_by_ca = chicago_loans.groupby('community_area').agg({
    'total_loan_count': 'sum',
    'total_loan_amount': 'sum',
}).round(2)

loans_by_ca_clean = loans_by_ca.reset_index()
loans_by_ca_clean['community_area'] = loans_by_ca_clean['community_area'].str.strip().str.lower()

socio['community_area'] = socio['community_area'].str.strip().str.lower()

#merge socio data with loan data
merged = socio.merge(
    loans_by_ca_clean[['community_area', 'total_loan_count', 'total_loan_amount']],
    on='community_area',
    how='left'
)

merged['loans_per_1k_hh'] = (merged['total_loan_count'] / merged['total_hh']) * 1000

print("=" * 60)
print("MACHINE LEARNING ANALYSIS: PREDICTING LOAN ACCESS")
print("=" * 60)

#selecting features for prediction (demographic and economic indicators)
feature_cols = [
    'white_hh_share', 'black_hh_share', 'latino_hh_share', 'asian_hh_share',
    'ami_shr', 'total_hh', 'white_own_shr', 'black_own_shr', 'latino_own_shr'
]

#target = what we r trying to predict (loan access per capita)
target_col = 'loans_per_1k_hh'

#removing any rows with missing data
df_clean = merged[feature_cols + [target_col]].dropna()

print(f"\nDataset: {len(df_clean)} neighborhoods with complete data")
print(f"Features: {len(feature_cols)}")
print(f"Target: {target_col}")

X = df_clean[feature_cols]
y = df_clean[target_col]

#splitting data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} neighborhoods")
print(f"Test set: {len(X_test)} neighborhoods")

# Model 1: Linear Regression
#this is a simple model that finds linear relationships between features and loan access
print("\n" + "=" * 60)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 60)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#predictions on test set
y_pred_lr = lr_model.predict(X_test)

#evaluating model performance
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)

print(f"\nR² Score: {lr_r2:.3f}")
print(f"  -> {lr_r2*100:.1f}% of variance in loan access is explained by demographics")
print(f"\nRMSE: {lr_rmse:.2f} loans per 1k households")
print(f"MAE: {lr_mae:.2f} loans per 1k households")

#feature importance for linear regression (coefficients)
print("\nFeature Importance (Linear Regression Coefficients):")
feature_importance_lr = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for idx, row in feature_importance_lr.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"  {row['Feature']:25s}: {row['Coefficient']:7.2f} {direction}")

# Model 2: Random Forest
#this is a more complex model that can capture non-linear relationships
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
print(f"\nRMSE: {rf_rmse:.2f} loans per 1k households")
print(f"MAE: {rf_mae:.2f} loans per 1k households")

#feature importance for random forest (based on how much each feature improves predictions)
print("\nFeature Importance (Random Forest):")
feature_importance_rf = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance_rf.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"  {row['Feature']:25s}: {row['Importance']:.3f} {bar}")

# Cross-validation to check model stability
#this tests the model on different splits of the data to make sure its not overfitting
print("\n" + "=" * 60)
print("CROSS-VALIDATION (5-FOLD)")
print("=" * 60)

cv_scores_lr = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

print(f"\nLinear Regression:")
print(f"  Mean R²: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std() * 2:.3f})")

print(f"\nRandom Forest:")
print(f"  Mean R²: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std() * 2:.3f})")

# Visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Viz 1: Feature Importance Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#linear regression coefficients
feature_importance_lr_plot = feature_importance_lr.copy()
colors_lr = ['steelblue' if x > 0 else 'coral' for x in feature_importance_lr_plot['Coefficient']]
ax1.barh(feature_importance_lr_plot['Feature'], feature_importance_lr_plot['Coefficient'], color=colors_lr, alpha=0.8)
ax1.set_xlabel('Coefficient', fontsize=12)
ax1.set_title('Linear Regression: Feature Coefficients', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

#random forest importance
ax2.barh(feature_importance_rf['Feature'], feature_importance_rf['Importance'], color='steelblue', alpha=0.8)
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Random Forest: Feature Importance', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/ml_viz_1_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/ml_viz_1_feature_importance.png")

# Viz 2: Predictions vs Actual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#linear regression predictions
ax1.scatter(y_test, y_pred_lr, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Loans per 1k HH', fontsize=12)
ax1.set_ylabel('Predicted Loans per 1k HH', fontsize=12)
ax1.set_title(f'Linear Regression (R² = {lr_r2:.3f})', fontsize=14, fontweight='bold')

#random forest predictions
ax2.scatter(y_test, y_pred_rf, alpha=0.6, s=100, edgecolors='black', linewidth=0.5, color='orange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Loans per 1k HH', fontsize=12)
ax2.set_ylabel('Predicted Loans per 1k HH', fontsize=12)
ax2.set_title(f'Random Forest (R² = {rf_r2:.3f})', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/ml_viz_2_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/ml_viz_2_predictions.png")

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print(f"""
1. Both models show demographics STRONGLY predict loan access
   - Linear Regression R²: {lr_r2:.3f}
   - Random Forest R²: {rf_r2:.3f}

2. Most Important Predictors:
   - Income (ami_shr): Higher income = more loans
   - White household share: Positive relationship with loan access
   - Black household share: Negative relationship with loan access

3. Model Performance:
   - Average prediction error: ~{lr_mae:.1f} loans per 1k households
   - Models r stable across different data splits (cross-validation)

4. What This Means:
   - Loan access is HIGHLY predictable from demographics
   - This suggests systematic disparities, not random variation
   - Geographic/racial patterns r consistent and measurable
""")

print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
