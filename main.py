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
    "socio_new": "Datasets/Socioeconomic_Neighborhoods_2025.csv.csv" #ok this is just an update social_old, same df structure
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

print("SBIF grant ratio (non-null):", dfs["sbif"]["GRANT_RATIO"].notna().sum())
print("NOF grant ratio (non-null):",  dfs["nof"]["GRANT_RATIO"].notna().sum())

#community areas BY project count
print("SBIF:top 20 community areas by count:\n")
print(dfs["sbif"]["COMMUNITY AREA"].value_counts().head(20))

print("NOF:top 20 community areas by count:\n")
print(dfs["nof"]["COMMUNITY AREA"].value_counts().head(20))

#this is just for missing stuff
print("SBIF missing (top 10):\n")
print(dfs["sbif"].isna().sum().sort_values(ascending=False).head(10))

print("NOF missing(top 10):\n")
print(dfs["nof"].isna().sum().sort_values(ascending=False).head(10))

#this is just checking if the people who applied for sbif maybe also did nof from a parituclar community : SBIF covers the majority of chicago neighborhoods
sbif_areas = set(dfs["sbif"]["COMMUNITY AREA"].dropna().astype(str).str.strip().unique()) if "COMMUNITY AREA" in dfs["sbif"].columns else set()
nof_areas  = set(dfs["nof"]["COMMUNITY AREA"].dropna().astype(str).str.strip().unique())  if "COMMUNITY AREA" in dfs["nof"].columns  else set()

#idk this might be useful
print("\nAreas in both SBIF & NOF:", len(sbif_areas & nof_areas))
print("Areas only in SBIF:", len(sbif_areas - nof_areas))
print("Areas only in NOF:",  len(nof_areas - sbif_areas))

#getting rid of any nulls, i dont see any eprsonally but just in case
print("SBIF GRANT_RATIO summary cleaned:\n")
print(dfs["sbif"]["GRANT_RATIO"].dropna().describe())

print("NOF GRANT_RATIO summary cleaned:\n")
print(dfs["nof"]["GRANT_RATIO"].dropna().describe())

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

print(dfs["cmi_demo"]['Borrower Ethnicity'].value_counts())
print(dfs["cmi_demo"]['Borrower Gender'].value_counts())



sbif_by_ca = (dfs["sbif"]
    .dropna(subset=["COMMUNITY AREA","GRANT_RATIO"])
    .groupby("COMMUNITY AREA")["GRANT_RATIO"].agg(["count","mean","median"]).sort_values("mean", ascending=False))
sbif_by_ca.head(10)

print("who gets the biggest % coverage by neightborhood", sbif_by_ca.head(30))
#understanding ecnomic factors / community - 2008-2012, note: im thinking we disregard this dataset...outdated + duplicate columns exist in 2025 anyway w the same statistics, lets try and find something more recent


# print(dfs["socio_old"][["COMMUNITY AREA NAME","PERCENT HOUSEHOLDS BELOW POVERTY","PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA"]].head(20))


# print(dfs["socio_old"][["COMMUNITY AREA NAME","PERCENT AGED 16+ UNEMPLOYED", "PERCENT AGED UNDER 18 OR OVER 64"]].head(20))

#understanding ecnomic factors / community - 2025

# print(dfs["socio_new"][["COMMUNITY AREA NAME","PER CAPITA INCOME ","HARDSHIP INDEX"]].head(20))


# print(dfs["socio_new"][["COMMUNITY AREA NAME","PERCENT HOUSEHOLDS BELOW POVERTY"]].head(20))
