from main import dfs
import matplotlib.pyplot as plt
import pandas as pd

socio = pd.read_csv("Datasets/chi_data.csv")
socio["community_area"] = socio["community_area"].str.strip().str.lower()

sbif = dfs["sbif"].groupby("COMMUNITY AREA")["GRANT_Percent"].mean().reset_index()
nof  = dfs["nof"].groupby("COMMUNITY AREA")["GRANT_Percent"].mean().reset_index()

merged = socio.merge(sbif, left_on="community_area", right_on="COMMUNITY AREA", how="left") \
              .merge(nof, left_on="community_area", right_on="COMMUNITY AREA", how="left")

merged.rename(columns={"GRANT_Percent_x": "SBIF_Grant%", "GRANT_Percent_y": "NOF_Grant%"}, inplace=True)

print(merged.head(10).to_string(index=False))

plt.figure(figsize=(8,6))
plt.scatter(merged["ami_shr"], merged["SBIF_Grant%"], label="SBIF", alpha=0.7, color='skyblue')
plt.scatter(merged["ami_shr"], merged["NOF_Grant%"], label="NOF", alpha=0.7, color='salmon')
plt.title("Grant % Coverage vs Income (AMI Share)")
plt.xlabel("Area Median Income Share (AMI)")
plt.ylabel("Average Grant % Coverage")
plt.legend()
plt.tight_layout()
plt.show(block=True)

plt.figure(figsize=(8,6))
plt.scatter(merged["black_hh_share"], merged["SBIF_Grant%"], label="SBIF", color='teal', alpha=0.7)
plt.scatter(merged["black_hh_share"], merged["NOF_Grant%"], label="NOF", color='orange', alpha=0.7)
plt.title("Grant % vs % Black Households by Community Area")
plt.xlabel("Share of Black Households")
plt.ylabel("Average Grant % Coverage")
plt.legend()
plt.tight_layout()
plt.show(block=True)

sbif_dollars = dfs["sbif"].dropna(subset=["COMMUNITY AREA","INCENTIVE AMOUNT"]) \
    .groupby("COMMUNITY AREA")["INCENTIVE AMOUNT"].sum().sort_values(ascending=False).head(10)
nof_dollars = dfs["nof"].dropna(subset=["COMMUNITY AREA","INCENTIVE AMOUNT"]) \
    .groupby("COMMUNITY AREA")["INCENTIVE AMOUNT"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(sbif_dollars.index, sbif_dollars.values, color='royalblue', label="SBIF")
plt.bar(nof_dollars.index, nof_dollars.values, color='tomato', alpha=0.7, label="NOF")

plt.title("Top 10 Communities by Total Grant Dollars")
plt.xlabel("Community Area")
plt.ylabel("Total Incentive Amount ($)")
plt.xticks(rotation=45, ha='right')
plt.legend()

ticks = [i * 1e6 for i in range(1, 11)]  
labels = [f"{i}M" for i in range(1, 11)]
plt.yticks(ticks, labels)

plt.tight_layout()
plt.show(block=True)

