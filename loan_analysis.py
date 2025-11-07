import pandas as pd
import numpy as np

#loading the 2023 CRA small business loan data for illinois
loan_data = pd.read_excel("Datasets/Illinois-CRA-Compilation-2023.xlsx")

print(f"\nOriginal loan data: {loan_data.shape[0]:,} rows and {loan_data.shape[1]} cols")
print("Columns:", loan_data.columns.tolist())

#we only want cook county data (chicago area) -> county FIPS 17031
chicago_loans = loan_data[loan_data['County FIPS'] == 17031].copy()
print(f"\nFiltered to Cook County: {len(chicago_loans):,} census tracts")

#the tract column is like 101.00, 102.01 etc, need to convert to match chi_data_tract format
#multiply by 100 to shift decimal: 101.00 -> 10100, 102.01 -> 10201
chicago_loans['tract_clean'] = (chicago_loans['Tract'] * 100).astype(int).astype(str)

#load the socioeconomic data with tract-to-community area mapping
chi_tract = pd.read_csv("Datasets/chi_data_tract.csv")
chi_tract['tract_clean'] = chi_tract['tract'].astype(str)

#merge to get community areas for each census tract
chicago_loans = chicago_loans.merge(
    chi_tract[['tract_clean', 'community_area', 'income_level', 'med_income_msa_shr', 'non_white_hh_share']],
    on='tract_clean',
    how='left'
)

#filter to only tracts that matched to chicago community areas (not suburbs)
chicago_loans = chicago_loans[chicago_loans['community_area'].notna()].copy()
print(f"Matched to Chicago community areas: {len(chicago_loans):,} tracts")
print(f"Community areas represented: {chicago_loans['community_area'].nunique()}")

#calculating total loan counts and amounts (sum of all size categories)
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

#calculating avg loan size per tract (amount / count)
chicago_loans['avg_loan_size'] = chicago_loans['total_loan_amount'] / chicago_loans['total_loan_count'].replace(0, np.nan)

print("\nLoan distribution summary:")
print(chicago_loans[['total_loan_count', 'total_loan_amount']].describe())

#top 20 tracts by loan count
print("\nTop 20 census tracts by loan count:")
print(chicago_loans.nlargest(20, 'total_loan_count')[['Tract', 'community_area', 'total_loan_count', 'total_loan_amount']])

#aggregate by community area (same as grant analysis)
loans_by_ca = chicago_loans.groupby('community_area').agg({
    '<100K Number': 'sum',
    '<100K Amount': 'sum',
    '100-<250K Number': 'sum',
    '100-<250K Amount': 'sum',
    '>250K Number': 'sum',
    '>250K Amount': 'sum',
    'GR<1M Number': 'sum',
    'GR<1M Amount': 'sum',
    'total_loan_count': 'sum',
    'total_loan_amount': 'sum',
    'non_white_hh_share': 'mean',
    'med_income_msa_shr': 'mean'
}).round(2)

loans_by_ca = loans_by_ca.sort_values('total_loan_count', ascending=False)

print("\nTop 20 community areas by loan count:")
print(loans_by_ca[['total_loan_count', 'total_loan_amount', 'non_white_hh_share']].head(20))

print("\nBottom 20 community areas by loan count:")
print(loans_by_ca[['total_loan_count', 'total_loan_amount', 'non_white_hh_share']].tail(20))

#loading full socio data for merging
socio = pd.read_csv("Datasets/chi_data.csv")
socio['community_area'] = socio['community_area'].str.strip().str.lower()

#resetting loans_by_ca index and cleaning community area names
loans_by_ca_clean = loans_by_ca.reset_index()
loans_by_ca_clean['community_area'] = loans_by_ca_clean['community_area'].str.strip().str.lower()

#rename columns to avoid conflicts with chi_data.csv (which has old loan data)
loans_by_ca_clean = loans_by_ca_clean.rename(columns={
    'total_loan_count': 'cra_loan_count_2023',
    'total_loan_amount': 'cra_loan_amount_2023',
    '<100K Number': 'small_loans_count',
    'GR<1M Number': 'loans_to_small_biz'
})

#merge with socioeconomic data
merged = socio.merge(
    loans_by_ca_clean[['community_area', 'cra_loan_count_2023', 'cra_loan_amount_2023', 'small_loans_count', 'loans_to_small_biz']],
    on='community_area',
    how='left'
)

print("\nLoan access by neighborhood demographics:")
print(merged[['community_area', 'cra_loan_count_2023', 'non_white_hh_share', 'ami_shr', 'income_level']].head(20).to_string(index=False))

#loan access and neighborhood characteristics correlations
loan_corr = merged[[
    'white_hh_share', 'black_hh_share', 'latino_hh_share',
    'non_white_hh_share', 'ami_shr', 'cra_loan_count_2023'
]].corr()['cra_loan_count_2023'].drop('cra_loan_count_2023')

print("\nLoan count correlations with neighborhood factors:")
print(loan_corr.to_string())

#loan amount correlations (drop nulls first)
amount_corr = merged[[
    'white_hh_share', 'black_hh_share', 'latino_hh_share',
    'non_white_hh_share', 'ami_shr', 'cra_loan_amount_2023'
]].dropna().corr()['cra_loan_amount_2023'].drop('cra_loan_amount_2023')

print("\nLoan amount correlations with neighborhood factors:")
print(amount_corr.to_string())

#calculate loans per 1000 households (similar to how we might calculate grant access per capita)
merged['cra_loans_per_1k_hh'] = (merged['cra_loan_count_2023'] / merged['total_hh']) * 1000

print("\nLoans per 1000 households by income level:")
print(merged.dropna(subset=['cra_loans_per_1k_hh']).groupby('income_level')['cra_loans_per_1k_hh'].agg(['count', 'mean', 'median']).to_string())

print("\nLoans per 1000 households by racial composition:")
#split into quartiles by non_white_hh_share (only for rows with loan data)
merged_clean = merged.dropna(subset=['cra_loans_per_1k_hh'])
merged_clean['non_white_quartile'] = pd.qcut(merged_clean['non_white_hh_share'], q=4, labels=['Q1 (least)', 'Q2', 'Q3', 'Q4 (most)'])
print(merged_clean.groupby('non_white_quartile')['cra_loans_per_1k_hh'].agg(['count', 'mean', 'median']).to_string())
