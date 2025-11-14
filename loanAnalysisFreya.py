from main import dfs
import matplotlib.pyplot as plt

dfs = dfs["socio_by_data"]
# understand the data --> read the readme to understand the relationship
# then to show the important relationship present the graphs 
# 1. graph by the community
dfs['total_loan_amount_calc'] = (
    dfs['amount_white'] + dfs['amount_black'] + dfs['amount_latino'] +
    dfs['amount_asian'] + dfs['amount_oth'] + dfs['amount_non_white'] +
    dfs['amount_miss']
)

# Group by community area
loan_by_community = dfs.groupby("community_area")["total_loan_amount"].sum().reset_index()

loan_by_community = loan_by_community.sort_values(by="total_loan_amount", ascending=False)

print(loan_by_community[["community_area", "total_loan_amount"]].to_string(index=False))
plt.figure(figsize=(14,6))
plt.bar(loan_by_community["community_area"], loan_by_community["total_loan_amount"], color='skyblue')
plt.xticks(rotation=90)
plt.title("Total Loan Amount by Community Area")
plt.xlabel("Community Area")
plt.ylabel("Total Loan Amount")

ax = plt.gca()
y_ticks = ax.get_yticks()
ax.set_yticklabels([f'{int(y/1000)}K' for y in y_ticks])

plt.tight_layout()
plt.show()

race_cols = [
    "loans_white_shr", "loans_black_shr", "loans_latino_shr",
    "loans_asian_shr", "loans_oth_shr"
]

dfs[race_cols] = dfs[race_cols].div(dfs[race_cols].sum(axis=1), axis=0) * 100

loan_by_race = dfs.groupby("community_area")[race_cols].mean().reset_index()

loan_by_race = loan_by_race.sort_values(by="community_area")

colors = {
    "loans_white_shr": "#425a6b",   
    "loans_black_shr": "#a25b1c",   
    "loans_latino_shr": "#63b963", 
    "loans_asian_shr": "#9467bd",   
    "loans_oth_shr": "#8c564b"      
}

# Plot stacked percentage bar chart
plt.figure(figsize=(14,7))

bottom = None
for col in race_cols:
    plt.bar(
        loan_by_race["community_area"],
        loan_by_race[col],
        bottom=bottom,
        color=colors[col],
        label=col.replace("loans_", "").replace("_shr", "").capitalize()
    )
    bottom = loan_by_race[col] if bottom is None else bottom + loan_by_race[col]
\
plt.title("Loan Share by Race (as % of Total Loans per Community Area)", fontsize=14)
plt.xlabel("Community Area", fontsize=12)
plt.ylabel("Percentage of Total Loans", fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0, 105)  # Headroom so bars don't touch the top
plt.legend(title="Race", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show(block=True)
#region the community areas by east,west,south, north and west
#than show the population in each region and draw the connection
# region_map = {
#     'Loop': 'North',
#     'Gage Park': 'South',
#     'Avalon Park': 'South',
#     'Lincoln Park': 'North',
#     'West Englewood': 'West',
#     'Bridgeport': 'South',
# }

# dfs['region'] = dfs['community_area'].map(region_map)
# loan_by_region = dfs.groupby("region")["total_loan_amount"].sum().reset_index()

# # Plot
# plt.figure(figsize=(8,5))
# plt.bar(loan_by_region["region"], loan_by_region["total_loan_amount"], color='lightcoral')
# plt.title("Total Loan Amount by Region")
# plt.xlabel("Region")
# plt.ylabel("Total Loan Amount")
# plt.tight_layout()
# plt.show(block=True)


#Loan vs Population
#Normalize loan by population to see which communities are underserved.

# When examining the loan distribution across Chicago’s community areas, we found that neighborhoods with predominantly low-income and minority populations — such as Riverdale, Englewood, and West Garfield Park — receive significantly fewer total loans and smaller loan amounts compared to wealthier, predominantly White areas like Edison Park or Mount Greenwood.
# At first glance, this pattern seems logical: lenders typically base decisions on financial risk, and lower-income borrowers are viewed as higher-risk clients with less collateral or repayment capacity. However, this “rational” lending behavior exposes a deeper systemic inequity. By continually limiting credit access in already disadvantaged communities, the system reinforces the very inequalities it responds to — creating a feedback loop of disinvestment, lower business growth, and reduced opportunities for wealth-building.
# This means that while income differences partly explain why certain areas receive fewer loans, the pattern cannot be separated from the city’s racial and economic segregation. Neighborhoods that have been historically underfunded continue to face barriers to financial inclusion, even when controlling for population size.