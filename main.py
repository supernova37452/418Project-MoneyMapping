import pandas as pd
import numpy as np

#this is just a checker to make sure the csv's r being loaded, disregard
def read_csv_any(p):
    try:
        return pd.read_csv(p, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="latin-1", low_memory=False)

paths = { 
    "cmi_loans": "Datasets/CMI_Microloans.csv",
    "cmi_demo": "Datasets/CMI_Microloans_By_Ethnicity_Gender.csv",
    "nof": "Datasets/NOF_Small_Projects_2025.csv",
    "sbif": "Datasets/SBIF_Applicants_Small_Business_Projects_2025.csv",
    "socio_old": "Datasets/Socioecon_Community_2008_2012.csv",
    "socio_new": "Datasets/Socioeconomic_Neighborhoods_2025.csv.csv",
    "socio_by_ca": "Datasets/chi_data_tract.csv",#ok this is just an update social_old, same df structure
}

dfs = {k: read_csv_any(p) for k,p in paths.items()}

for name, df in dfs.items():
    print(f"\n{name}: {df.shape[0]:,} rows and {df.shape[1]} cols")
    print(df.columns.tolist()[:12])
    print(df.head(3)) 
    

#these are grant programs:  sbif and nof grants , this is to see the grant/community 
#obervations: more sbif usage, ill look more into other columns to see why this is the case but its more sbif are long term, nof are pre project


for prog in ["sbif", "nof"]:
    dfs[prog].columns = dfs[prog].columns.str.strip()

#cleaning all the "$some number" to be nums. curr strings
def to_numeric_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )

#grant ratios = INCENTIVE AMOUNT ÷ TOTAL PROJECT COST, this tells us what share of each project’s total request was covered by the EACH grant
for prog in ["sbif", "nof"]:
    df = dfs[prog]
    needed = ["INCENTIVE AMOUNT", "TOTAL PROJECT COST", "COMMUNITY AREA"]
    if "INCENTIVE AMOUNT" in df.columns:
        df["INCENTIVE AMOUNT"] = to_numeric_clean(df["INCENTIVE AMOUNT"])
    if "TOTAL PROJECT COST" in df.columns:
        df["TOTAL PROJECT COST"] = to_numeric_clean(df["TOTAL PROJECT COST"])

    #some projects dont actually have a cost so its $0.... if thats the case we dont want to devide lol
    if {"INCENTIVE AMOUNT", "TOTAL PROJECT COST"}.issubset(df.columns):
        denom = df["TOTAL PROJECT COST"].replace(0, np.nan)
        df["GRANT_RATIO"] = df["INCENTIVE AMOUNT"] / denom
    else:
        df["GRANT_RATIO"] = np.nan
    
    df["GRANT_Percent"] = (df["GRANT_RATIO"] * 100).round(1) #just turning the ratios into % for easier looks

print("SBIF GRANT_Percent summary (%):\n", dfs["sbif"]["GRANT_Percent"].dropna().describe())
print("NOF GRANT_Percent summary (%):\n", dfs["nof"]["GRANT_Percent"].dropna().describe())

print(dfs["sbif"]["GRANT_Percent"].dropna().map(lambda x: f"{x:.1f}%").head())

#community areas BY project count
print("SBIF:top 20 community areas by count:\n")
print(dfs["sbif"]["COMMUNITY AREA"].value_counts().head(20))


print("NOF:top 20 community areas by count:\n")
print(dfs["nof"]["COMMUNITY AREA"].value_counts().head(20))

#this is just for missing stuff
# print("SBIF missing (top 10):\n")
# print(dfs["sbif"].isna().sum().sort_values(ascending=False).head(10))

# print("NOF missing(top 10):\n")
# print(dfs["nof"].isna().sum().sort_values(ascending=False).head(10))

#this is just checking if the people who applied for sbif maybe also did nof from a parituclar community : SBIF covers the majority of chicago neighborhoods
sbif_areas = set(dfs["sbif"]["COMMUNITY AREA"].dropna().astype(str).str.strip().unique()) if "COMMUNITY AREA" in dfs["sbif"].columns else set()
nof_areas  = set(dfs["nof"]["COMMUNITY AREA"].dropna().astype(str).str.strip().unique())  if "COMMUNITY AREA" in dfs["nof"].columns  else set()

#idk this might be useful, leave this here
print("\nAreas in both SBIF & NOF:", len(sbif_areas & nof_areas))
print("Areas only in SBIF:", len(sbif_areas - nof_areas))
print("Areas only in NOF:",  len(nof_areas - sbif_areas))


# here, we are removing any nulls for community area and %, grouping by commuity area so they show up once and calculating median and mean of those grant %'s
sbif_by_ca = (
    dfs["sbif"].dropna(subset=["COMMUNITY AREA","GRANT_Percent"]).groupby("COMMUNITY AREA")["GRANT_Percent"]
      .agg(count="count", mean="mean", median="median")
      .sort_values("mean", ascending=False)
)

sbif_dollars = (
    dfs["sbif"].dropna(subset=["COMMUNITY AREA","INCENTIVE AMOUNT"]).groupby("COMMUNITY AREA")["INCENTIVE AMOUNT"].sum()
      .sort_values(ascending=False)
)

nof_dollars = (
    dfs["nof"].dropna(subset=["COMMUNITY AREA","INCENTIVE AMOUNT"]).groupby("COMMUNITY AREA")["INCENTIVE AMOUNT"].sum()
      .sort_values(ascending=False)
)
print("\nTotal SBIF dollars by neighborhood (top 20):")
print(sbif_dollars.head(20))

print("\nTotal NOF dollars by neighborhood (top 20):")
print(nof_dollars.head(20))

print("who gets the biggest % coverage by neighborhood")
print(sbif_by_ca.head(30))

socio = pd.read_csv("Datasets/chi_data.csv")

# come back to this later loan stuff later

# socio["loan_amount_per_hh"] = socio["total_loan_amount"] / socio["total_hh"]
# socio["loans_per_1k_hh"] = socio["total_loans"] / (socio["total_hh"] / 100)

# print(socio[["loan_amount_per_hh","non_white_hh_share","ami_shr"]].corr())

socio["community_area"] = socio["community_area"].str.strip().str.lower()
dfs["sbif"]["COMMUNITY AREA"] = dfs["sbif"]["COMMUNITY AREA"].astype(str).str.strip().str.lower()
dfs["nof"]["COMMUNITY AREA"]  = dfs["nof"]["COMMUNITY AREA"].astype(str).str.strip().str.lower()


#just avg of grant % coverage by community area
sbif = dfs["sbif"].groupby("COMMUNITY AREA")["GRANT_Percent"].mean().reset_index()
nof  = dfs["nof"].groupby("COMMUNITY AREA")["GRANT_Percent"].mean().reset_index()

#merging the community areas for sbif and nof-> primary key
merged = socio.merge(sbif, left_on="community_area", right_on="COMMUNITY AREA", how="left").merge(nof, left_on="community_area", right_on="COMMUNITY AREA", how="left")

# note i split these up bc the printing was a mess and it was hard to see, all of this is explained in the readme
print("\ndemographic shares!!!!!")
print(merged[["community_area","white_hh_share","black_hh_share","latino_hh_share","non_white_hh_share"]].head(20).to_string(index=False))

print("\nownership shares!!!!")
print(merged[["community_area","white_own_shr","black_own_shr","latino_own_shr","non_white_own_shr"]].head(20).to_string(index=False))

print("\nincome & grant coverage!!!!")
print(merged[["community_area","ami_shr","income_level","low_inc","GRANT_Percent_x","GRANT_Percent_y"]].head(20).to_string(index=False))


#grant coverage and neighborhood characteristics relatiosbusp

#SBIF correlations (GRANT_Percent_x)
sbif_corr = merged[[
    "white_hh_share","black_hh_share","latino_hh_share",
    "non_white_hh_share","ami_shr","GRANT_Percent_x"
]].corr()["GRANT_Percent_x"].drop("GRANT_Percent_x")

print("\nSBIF correlations with neighborhood factors:")
print(sbif_corr.to_string())

# NOF correlations 
nof_corr = merged[[
    "white_hh_share","black_hh_share","latino_hh_share",
    "non_white_hh_share","ami_shr","GRANT_Percent_y"
]].corr()["GRANT_Percent_y"].drop("GRANT_Percent_y")

print("\nNOF correlations with neighborhood factors:")
print(nof_corr.to_string())


# print(df[["loan_amount_per_hh","non_white_hh_share","ami_shr"]].corr())
# print(merged[["non_white_hh_share","ami_shr","sbif_mean","nof_mean"]].corr())





# print("whats this about")
# print(dfs["socio_by_ca"][['tract', 'community_area', 'total_loans', 'amount_non_white', 'income_level']].head(30))

# # normalize names once so merges line up
# dfs["socio_by_ca"]["community_area"] = dfs["socio_by_ca"]["community_area"].str.strip().str.lower()
# dfs["sbif"]["COMMUNITY AREA"] = dfs["sbif"]["COMMUNITY AREA"].astype(str).str.strip().str.lower()
# dfs["nof"]["COMMUNITY AREA"]  = dfs["nof"]["COMMUNITY AREA"].astype(str).str.strip().str.lower()

# # socioeconomic averages by community area
# socio = dfs["socio_by_ca"].copy()
# socio_ca = (
#     socio.groupby("community_area")[["non_white_hh_share", "white_hh_share"]]
#          .mean()
#          .reset_index()
# )

# # SBIF / NOF average grant coverage by community area
# sbif_grants = (
#     dfs["sbif"].groupby("COMMUNITY AREA")["GRANT_Percent"]
#                .mean()
#                .reset_index()
#                .rename(columns={"COMMUNITY AREA":"community_area", "GRANT_Percent":"sbif_mean"})
# )
# nof_grants = (
#     dfs["nof"].groupby("COMMUNITY AREA")["GRANT_Percent"]
#               .mean()
#               .reset_index()
#               .rename(columns={"COMMUNITY AREA":"community_area", "GRANT_Percent":"nof_mean"})
# )

# # merge
# merged = socio_ca.merge(sbif_grants, on="community_area", how="left")\
#                  .merge(nof_grants, on="community_area", how="left")

# print("\ncommunity-level socioeconomic and grant comparison:\n")
# print(merged.head(20).to_string(index=False))

# print("\ncorrelations between demographics and grant coverage:")
# print(merged[["non_white_hh_share", "white_hh_share", "sbif_mean", "nof_mean"]].corr())




#getting rid of any nulls, i dont see any eprsonally but just in case
# print("SBIF GRANT_RATIO summary cleaned:\n")
# print(dfs["sbif"]["GRANT_RATIO"].dropna().describe())

# print("NOF GRANT_RATIO summary cleaned:\n")
# print(dfs["nof"]["GRANT_RATIO"].dropna().describe())

#NOTE: im not personally working on this part w cmi loans but if we do keep this data set, some cleaning is here


# loan programs: cmi loan, a look into the ethnicities and genders

dfs["cmi_demo"]["Borrower Gender"] = dfs["cmi_demo"]["Borrower Gender"].map({
    "Female": "Female",
    "F": "Female",
    "F ": "Female",
    "F/M": "Female",
    "F / M": "Female",
    "M/F": "Female",
    "Male": "Male",
    "M": "Male",
    "M/M": "Male",
    " M": "Male",
}).fillna("Other")

dfs["cmi_demo"]["Borrower Ethnicity"] = (
    dfs["cmi_demo"]["Borrower Ethnicity"]
    .astype(str)
    .str.strip() #fortrailing spaces
    .replace({
        "": "Not Specified",
        "nan": "Not Specified",
        "(blank)": "Not Specified",
    })
)

# print(dfs["cmi_demo"]['Borrower Ethnicity'].value_counts())
# print(dfs["cmi_demo"]['Borrower Gender'].value_counts())


#understanding ecnomic factors / community - 2008-2012, note: im thinking we disregard this dataset...outdated + duplicate columns exist in 2025 anyway w the same statistics, lets try and find something more recent


# print(dfs["socio_old"][["COMMUNITY AREA NAME","PERCENT HOUSEHOLDS BELOW POVERTY","PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA"]].head(20))


# print(dfs["socio_old"][["COMMUNITY AREA NAME","PERCENT AGED 16+ UNEMPLOYED", "PERCENT AGED UNDER 18 OR OVER 64"]].head(20))

#understanding ecnomic factors / community - 2025

# print(dfs["socio_new"][["COMMUNITY AREA NAME","PER CAPITA INCOME ","HARDSHIP INDEX"]].head(20))


# print(dfs["socio_new"][["COMMUNITY AREA NAME","PERCENT HOUSEHOLDS BELOW POVERTY"]].head(20))
