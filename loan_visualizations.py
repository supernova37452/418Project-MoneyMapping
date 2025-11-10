import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#just setting plot styles to make graphs look nicer
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

#load illinois loan data and filter for cook county (chicago)
loan_data = pd.read_excel("Datasets/Illinois-CRA-Compilation-2023.xlsx")
chicago_loans = loan_data[loan_data['County FIPS'] == 17031].copy()

#tract numbers r like 101.00, 102.01 etc -> need to convert to match chi_data format (10100, 10201)
chicago_loans['tract_clean'] = (chicago_loans['Tract'] * 100).astype(int).astype(str)

chi_tract = pd.read_csv("Datasets/chi_data_tract.csv")
chi_tract['tract_clean'] = chi_tract['tract'].astype(str)

#merge to get community area names for each census tract
chicago_loans = chicago_loans.merge(
    chi_tract[['tract_clean', 'community_area', 'income_level', 'med_income_msa_shr', 'non_white_hh_share']],
    on='tract_clean',
    how='left'
)

#only keep tracts that matched to chicago community areas (removes suburbs)
chicago_loans = chicago_loans[chicago_loans['community_area'].notna()].copy()

#total loan count = sum of all loan sizes
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

#grouping by community area to get totals for each neighborhood
loans_by_ca = chicago_loans.groupby('community_area').agg({
    'total_loan_count': 'sum',
    'total_loan_amount': 'sum',
    'non_white_hh_share': 'mean',
    'med_income_msa_shr': 'mean'
}).round(2)

loans_by_ca_clean = loans_by_ca.reset_index()
loans_by_ca_clean['community_area'] = loans_by_ca_clean['community_area'].str.strip().str.lower()

#merge with socio data to get demographics
socio = pd.read_csv("Datasets/chi_data.csv")
socio['community_area'] = socio['community_area'].str.strip().str.lower()

merged = socio.merge(
    loans_by_ca_clean[['community_area', 'total_loan_count', 'total_loan_amount']],
    on='community_area',
    how='left'
)

#calculating loans per 1k households to normalize by population
merged['loans_per_1k_hh'] = (merged['total_loan_count'] / merged['total_hh']) * 1000

print("Creating loan visualizations...")

# Visualization 1: Top and Bottom 15 neighborhoods by loan count
#this shows which neighborhoods get the most vs least loans
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

top_15 = loans_by_ca.nlargest(15, 'total_loan_count')
bottom_15 = loans_by_ca.nsmallest(15, 'total_loan_count')

#left side = top 15 neighborhoods
ax1.barh(range(len(top_15)), top_15['total_loan_count'], color='steelblue')
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15.index, fontsize=10)
ax1.set_xlabel('Total Loan Count', fontsize=12)
ax1.set_title('Top 15 Neighborhoods by Loan Count (2023)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

#right side = bottom 15 neighborhoods
ax2.barh(range(len(bottom_15)), bottom_15['total_loan_count'], color='coral')
ax2.set_yticks(range(len(bottom_15)))
ax2.set_yticklabels(bottom_15.index, fontsize=10)
ax2.set_xlabel('Total Loan Count', fontsize=12)
ax2.set_title('Bottom 15 Neighborhoods by Loan Count (2023)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('visualizations/loan_viz_1_top_bottom_neighborhoods.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/loan_viz_1_top_bottom_neighborhoods.png")

# Visualization 2: Loans per 1000 households by income level
#comparing loan access for low vs moderate vs upper income neighborhoods
fig, ax = plt.subplots(figsize=(10, 6))

income_groups = merged.dropna(subset=['loans_per_1k_hh']).groupby('income_level')['loans_per_1k_hh'].agg(['mean', 'median', 'count'])
income_groups = income_groups.reindex(['LMI', 'MI', 'UI'])

x = range(len(income_groups))
#showing both mean and median to see if there r outliers
ax.bar([i - 0.2 for i in x], income_groups['mean'], 0.4, label='Mean', color='steelblue', alpha=0.8)
ax.bar([i + 0.2 for i in x], income_groups['median'], 0.4, label='Median', color='coral', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(['Low/Moderate\nIncome (LMI)', 'Moderate\nIncome (MI)', 'Upper\nIncome (UI)'], fontsize=11)
ax.set_ylabel('Loans per 1,000 Households', fontsize=12)
ax.set_title('Small Business Loan Access by Income Level', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

#n= shows how many neighborhoods r in each category
for i, count in enumerate(income_groups['count']):
    ax.text(i, 5, f'n={int(count)}', ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('visualizations/loan_viz_2_loans_by_income.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/loan_viz_2_loans_by_income.png")

# Visualization 3: Loans per 1000 households by predominant race
fig, ax = plt.subplots(figsize=(12, 6))

#categorizing neighborhoods by dominant racial group (which group has highest share)
merged_clean = merged.dropna(subset=['loans_per_1k_hh']).copy()

#this determines which racial group is dominant in each neighborhood
def get_dominant_race(row):
    if row['white_hh_share'] > 0.5:
        return 'Majority White'
    elif row['black_hh_share'] > 0.5:
        return 'Majority Black'
    elif row['latino_hh_share'] > 0.5:
        return 'Majority Latino'
    else:
        return 'Mixed/Other'

merged_clean['dominant_race'] = merged_clean.apply(get_dominant_race, axis=1)

#group by dominant race and calc mean/median
race_groups = merged_clean.groupby('dominant_race')['loans_per_1k_hh'].agg(['mean', 'median', 'count'])
race_order = ['Majority White', 'Mixed/Other', 'Majority Latino', 'Majority Black']
race_groups = race_groups.reindex([r for r in race_order if r in race_groups.index])

x = range(len(race_groups))
ax.bar([i - 0.2 for i in x], race_groups['mean'], 0.4, label='Mean', color='steelblue', alpha=0.8)
ax.bar([i + 0.2 for i in x], race_groups['median'], 0.4, label='Median', color='coral', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(race_groups.index, fontsize=11)
ax.set_ylabel('Loans per 1,000 Households', fontsize=12)
ax.set_title('Small Business Loan Access by Neighborhood Racial Composition', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

#add neighborhood count labels
for i, count in enumerate(race_groups['count']):
    ax.text(i, 5, f'n={int(count)}', ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('visualizations/loan_viz_3_loans_by_race.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/loan_viz_3_loans_by_race.png")

print("\n✓ All visualizations created successfully!")
print("3 PNG files saved in visualizations/ folder")
