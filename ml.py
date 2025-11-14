"""
MoneyMapping ML analysis
File: MoneyMapping_ML.py

Description
-----------
A self-contained Python script / notebook-like file to:
 - load SBIF project dataset and socioeconomic dataset
 - robustly detect columns and merge datasets on community area
 - preprocess monetary fields and location fields
 - perform regression (Linear Regression + Random Forest) to predict incentive/grant amount
 - compute feature importances, evaluation metrics, and residuals
 - aggregate by community and run KMeans clustering
 - save outputs (CSVs + plots) to /mnt/data

Usage
-----
Run in a Jupyter cell or as a script. Requires:
 - pandas, numpy, scikit-learn, matplotlib
Optional (recommended):
 - jupyterlab/notebook to run interactively

Files
-----
Expected inputs (place in same working directory or provide full paths):
 - SBIF_Applicants_Small_Business_Projects_2025.csv
 - Socioecon_Community_2008_2012.csv

Outputs (saved to /mnt/data):
 - merged_money_mapping.csv
 - regression_metrics.csv
 - feature_importances.csv
 - community_clusters.csv
 - residuals_by_project.csv
 - plots: actual_vs_pred.png, feature_importance.png, clusters_map.png

Notes
-----
This script attempts robust detection of relevant columns (e.g., 'INCENTIVE AMOUNT', 'COMMUNITY AREA', 'LOCATION'). If your datasets use different column names, update the heuristics at the top of the script accordingly.

"""

import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score

# --------- Configuration ---------
SBIF_PATH = 'Datasets/SBIF_Applicants_Small_Business_Projects_2025.csv'
SOCIO_PATH = 'Datasets/Socioecon_Community_2008_2012.csv'
OUT_DIR = './outputs'
RANDOM_STATE = 42
N_CLUSTERS = 3

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Helpers ----------

def parse_money(x):
    """Turn strings like '$12,345.67' into float 12345.67. Returns np.nan for missing/invalid."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    # remove parentheses for negative numbers, convert to negative
    neg = False
    if '(' in s and ')' in s:
        neg = True
    s = re.sub(r'[\$,%\(\) ]', '', s)
    s = re.sub(r'[^0-9\.\-]', '', s)
    if s == '':
        return np.nan
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return np.nan


def extract_latlon(location_field):
    """Extract latitude and longitude from strings like 'POINT (-87.66 41.96)'. Returns (lat, lon) or (np.nan, np.nan)."""
    if pd.isna(location_field):
        return (np.nan, np.nan)
    s = str(location_field)
    m = re.search(r'POINT\s*\(\s*([\-0-9\.]+)\s+([\-0-9\.]+)\s*\)', s)
    if m:
        lon = float(m.group(1))
        lat = float(m.group(2))
        return lat, lon
    # Try comma-separated 'lat, lon'
    m2 = re.search(r'([\-0-9\.]+)\s*,\s*([\-0-9\.]+)', s)
    if m2:
        return float(m2.group(1)), float(m2.group(2))
    return (np.nan, np.nan)


# ---------- Load datasets ----------
print('Loading datasets...')
sbif = pd.read_csv(SBIF_PATH)
socio = pd.read_csv(SOCIO_PATH)

# strip column whitespace
sbif.columns = sbif.columns.str.strip()
socio.columns = socio.columns.str.strip()

# show a brief preview when run interactively
print('SBIF columns:', sbif.columns.tolist()[:30])
print('Socio columns:', socio.columns.tolist()[:30])

# ---------- Detect community area columns ----------
sbif_comm_col = None
for c in sbif.columns:
    if 'COMMUNITY' in c.upper() and ('AREA' in c.upper() or 'AREA NAME' in c.upper()):
        sbif_comm_col = c
        break
# fallback to any column named COMMUNITY
if sbif_comm_col is None and 'COMMUNITY AREA' in sbif.columns:
    sbif_comm_col = 'COMMUNITY AREA'

socio_name_col = None
for c in socio.columns:
    if 'COMMUNITY' in c.upper() and ('NAME' in c.upper() or 'AREA' in c.upper()):
        socio_name_col = c
        break
if socio_name_col is None:
    # choose first object column as last resort
    obj_cols = socio.select_dtypes(include='object').columns
    socio_name_col = obj_cols[0] if len(obj_cols) else None

if sbif_comm_col is None or socio_name_col is None:
    raise RuntimeError('Could not detect community area columns automatically. Edit sbif_comm_col and socio_name_col.')

print('Detected SBIF community column:', sbif_comm_col)
print('Detected socio community name column:', socio_name_col)

# normalize names
sbif['community_area_name'] = sbif[sbif_comm_col].astype(str).str.strip()
socio['community_area_name'] = socio[socio_name_col].astype(str).str.strip()

# ---------- Merge ----------
print('Merging datasets...')
merged = sbif.merge(socio, on='community_area_name', how='left', suffixes=('','_socio'))
merged.to_csv(os.path.join(OUT_DIR, 'merged_money_mapping.csv'), index=False)
print('Saved merged dataset to merged_money_mapping.csv')

# ---------- Parse monetary and numeric columns ----------
# Detect a grant/incentive column
grant_candidates = [c for c in merged.columns if 'GRANT' in c.upper() or 'INCENTIVE' in c.upper()]
if len(grant_candidates) == 0:
    # attempt to detect columns with $ in first 200 rows
    possible = []
    for c in merged.columns:
        sample = merged[c].dropna().astype(str).head(200).to_list()
        if any('$' in s or ',' in s for s in sample):
            possible.append(c)
    grant_candidates = possible

if len(grant_candidates) == 0:
    raise RuntimeError('No grant/incentive-like column found. Please inspect column names and update the script.')

TARGET_COL = grant_candidates[0]
print('Using target column:', TARGET_COL)
merged[TARGET_COL] = merged[TARGET_COL].apply(parse_money)

# Parse total project cost if available
if 'TOTAL PROJECT COST' in merged.columns:
    merged['TOTAL_PROJECT_COST_NUM'] = merged['TOTAL PROJECT COST'].apply(parse_money)

# Extract lat/lon
loc_col = None
for c in merged.columns:
    if 'LOCATION' in c.upper():
        loc_col = c
        break
if loc_col:
    merged[['lat','lon']] = merged[loc_col].apply(lambda x: pd.Series(extract_latlon(x)))
else:
    merged['lat'] = np.nan; merged['lon'] = np.nan

# ---------- Choose features ----------
# Socioeconomic features we like to use (present in your socio file)
socio_candidates = ['HARDSHIP INDEX','PER CAPITA INCOME','PERCENT HOUSEHOLDS BELOW POVERTY',
                    'PERCENT AGED 16+ UNEMPLOYED','PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA']
socio_feats = [c for c in socio_candidates if c in merged.columns]

proj_feats = []
if 'TOTAL_PROJECT_COST_NUM' in merged.columns:
    proj_feats.append('TOTAL_PROJECT_COST_NUM')
# jobs columns
for c in merged.columns:
    if 'JOB' in c.upper() and ('CREAT' in c.upper() or 'RETAIN' in c.upper()):
        proj_feats.append(c)

features = socio_feats + proj_feats
print('Selected features:', features)

# ---------- Regression: Predict incentive amount ----------
reg_df = merged[features + [TARGET_COL]].copy()
# drop rows without target
reg_df = reg_df.dropna(subset=[TARGET_COL]).copy()
# convert features numeric
for c in features:
    reg_df[c] = pd.to_numeric(reg_df[c], errors='coerce')
# fill NA with median
for c in features:
    if c in reg_df.columns:
        reg_df[c] = reg_df[c].fillna(reg_df[c].median())

# log-transform monetary columns to stabilize
reg_df['log_target'] = np.log1p(reg_df[TARGET_COL])
if 'TOTAL_PROJECT_COST_NUM' in reg_df.columns:
    reg_df['log_total_cost'] = np.log1p(reg_df['TOTAL_PROJECT_COST_NUM'])
    if 'TOTAL_PROJECT_COST_NUM' in features:
        features.remove('TOTAL_PROJECT_COST_NUM')
    features = ['log_total_cost'] + socio_feats + [c for c in proj_feats if c not in ['TOTAL_PROJECT_COST_NUM']]
else:
    features = socio_feats + proj_feats

model_features = [f for f in features if f in reg_df.columns]
print('Model features used:', model_features)

X = reg_df[model_features].values
y = reg_df['log_target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# scale for linear model
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Linear regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

# Random forest (use unscaled features for tree models)
rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, max_depth=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# evaluation helpers
def eval_metrics_log(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

metrics_lr = eval_metrics_log(y_test, y_pred_lr)
metrics_rf = eval_metrics_log(y_test, y_pred_rf)
metrics_df = pd.DataFrame([metrics_lr, metrics_rf], index=['LinearRegression','RandomForest'])
metrics_df.to_csv(os.path.join(OUT_DIR,'regression_metrics.csv'))
print('Saved regression metrics to regression_metrics.csv')

# feature importances
feat_imp = pd.DataFrame({'feature': model_features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
feat_imp.to_csv(os.path.join(OUT_DIR,'feature_importances.csv'), index=False)
print('Saved feature importances to feature_importances.csv')

# save residuals for further analysis
# compute residuals on the test set
residuals_df = pd.DataFrame({
    'y_true_log': y_test,
    'y_pred_rf_log': y_pred_rf
})
residuals_df['y_true'] = np.expm1(residuals_df['y_true_log'])
residuals_df['y_pred_rf'] = np.expm1(residuals_df['y_pred_rf_log'])
residuals_df['residual'] = residuals_df['y_true'] - residuals_df['y_pred_rf']
residuals_df.to_csv(os.path.join(OUT_DIR,'residuals_by_project.csv'), index=False)
print('Saved residuals to residuals_by_project.csv')

# plot actual vs predicted
plt.figure(figsize=(6,5))
plt.scatter(residuals_df['y_true'], residuals_df['y_pred_rf'], alpha=0.6)
plt.plot([residuals_df['y_true'].min(), residuals_df['y_true'].max()], [residuals_df['y_true'].min(), residuals_df['y_true'].max()], linestyle='--')
plt.xlabel('Actual incentive amount ($)')
plt.ylabel('Predicted incentive amount ($)')
plt.title('RandomForest: Actual vs Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'actual_vs_pred.png'))
print('Saved plot actual_vs_pred.png')

# feature importance plot
plt.figure(figsize=(6,4))
plt.bar(feat_imp['feature'], feat_imp['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('RandomForest Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'feature_importance.png'))
print('Saved plot feature_importance.png')

# ---------- Community-level clustering ----------
print('Running community-level clustering...')
agg = merged.groupby('community_area_name').agg({
    TARGET_COL: 'mean',
    'PROJECT NAME': 'count'
}).rename(columns={'PROJECT NAME':'num_projects'}).reset_index()
# merge socio means
for c in socio_feats:
    if c in merged.columns:
        agg[c] = merged.groupby('community_area_name')[c].mean().values

cluster_features = [c for c in [TARGET_COL] + socio_feats if c in agg.columns]
cluster_df = agg.dropna(subset=cluster_features).copy()

if len(cluster_df) >= 3:
    Z = StandardScaler().fit_transform(cluster_df[cluster_features].values)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(Z)
    cluster_df['cluster'] = labels
    sil = silhouette_score(Z, labels)
    print('Silhouette score:', sil)
else:
    cluster_df['cluster'] = 0
    sil = np.nan

cluster_df.to_csv(os.path.join(OUT_DIR,'community_clusters.csv'), index=False)
print('Saved community clusters to community_clusters.csv')

print('All done. Outputs saved in', OUT_DIR)

# Optionally, return or display top underfunded communities by residual
try:
    # Attach residuals back to merged rows: map using index of test set
    # Note: we created residuals on shuffled test split; here we recompute residuals for entire dataset for interpretability
    merged_full = merged.copy()
    # create model predictions for all rows (use rf)
    full_features = [f for f in model_features if f in merged_full.columns]
    merged_full = merged_full.dropna(subset=full_features)
    X_full = merged_full[full_features].values
    preds_full_log = rf.predict(X_full)
    merged_full['predicted_incentive'] = np.expm1(preds_full_log)
    # ensure numeric target
    merged_full[TARGET_COL] = pd.to_numeric(merged_full[TARGET_COL], errors='coerce')
    merged_full['residual'] = merged_full[TARGET_COL] - merged_full['predicted_incentive']
    # community-level residual summary
    resid_by_comm = merged_full.groupby('community_area_name')['residual'].mean().reset_index().sort_values('residual')
    resid_by_comm.to_csv(os.path.join(OUT_DIR,'residuals_by_community.csv'), index=False)
    print('Saved residuals_by_community.csv (negative -> underfunded on average)')
except Exception as e:
    print('Could not compute full residuals:', e)

# End of script

