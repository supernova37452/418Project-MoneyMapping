# Part 3: Small Business Loan Analysis (2023 CRA Data)

This section explores how small business loans from banks and credit unions are distributed across Chicago neighborhoods using 2023 Community Reinvestment Act (CRA) data.

# Data Source and Cleaning

The Illinois-CRA-Compilation-2023.xlsx dataset contains loan data for all Illinois counties (3,255 census tracts total). We filtered for Cook County only (County FIPS = 17031, which is 1,331 tracts) and matched census tracts to Chicago's 77 community areas using the chi_data_tract.csv mapping. Successfully matched 786 census tracts to Chicago community areas.

The dataset includes loan counts and dollar amounts broken down by size:
- <100K loans → small business loans under $100,000
- 100-<250K loans → medium loans between $100K-$250K
- >=250K loans → large loans over $250K
- GR<1M → loans to businesses with gross revenue under $1M

Overall loan distribution: mean = 82 loans per tract, median = 57 loans per tract

# Key Metrics Calculated

total_loan_count = sum of all loan size categories by community area
total_loan_amount = total dollar value of all loans by community area
loans_per_1k_hh = (total_loan_count / total_households) × 1000 → standardizes loan access per capita

# Top Neighborhoods by Loan Access

The downtown and North Side neighborhoods dominate loan access:

Near North Side → 5,499 loans ($214M), 27% non-white
Loop → 4,224 loans ($266M), 40% non-white
Near West Side → 3,654 loans ($148M), 62% non-white
West Town → 3,531 loans ($107M), 33% non-white

In contrast, South and West Side neighborhoods receive fewer loans despite high populations:

Austin → 1,442 loans ($45M), 95% non-white
Englewood → 238 loans ($3M), 99% non-white
West Garfield Park → 234 loans ($10M), 99% non-white

Bottom 5 neighborhoods by loan count:
Fuller Park → 42 loans ($2M), 99% non-white
Burnside → 42 loans ($1M), 100% non-white
Riverdale → 54 loans ($2M), 99% non-white
Pullman → 71 loans ($3M), 87% non-white
Hegewisch → 96 loans ($1M), 55% non-white

# Loan Access and Neighborhood Characteristics

Correlations between loan counts and demographics (77 community areas):

non_white_hh_share → -0.525 (strong negative correlation)
ami_shr (income) → +0.556 (strong positive correlation)
white_hh_share → +0.525 (strong positive correlation)
black_hh_share → -0.376 (moderate negative correlation)
latino_hh_share → -0.114 (weak negative correlation)

Loan amount correlations show similar patterns:
non_white_hh_share → -0.430
ami_shr (income) → +0.514

Loans per 1,000 households by income level:
UI (Upper Income) → 82.4 loans per 1k households (10 neighborhoods)
MI (Moderate Income) → 68.2 loans per 1k households (12 neighborhoods)
LMI (Low/Moderate Income) → 43.3 loans per 1k households (54 neighborhoods)

Loans per 1,000 households by racial composition (quartiles):
Q1 (least non-white) → 79.9 loans per 1k households
Q2 → 57.8 loans per 1k households
Q3 → 40.0 loans per 1k households
Q4 (most non-white) → 36.1 loans per 1k households

This shows that even after controlling for population, majority non-white neighborhoods get less than HALF the loan access of majority white neighborhoods.

# What this means:

Unlike grants (where coverage % was fairly even across neighborhoods at ~50-60%), loan ACCESS shows MUCH clearer disparities. The correlations are strong:

SBIF grant coverage vs. non_white_hh_share → -0.001 (essentially zero, meaning grants are neutral)
CRA loan count vs. non_white_hh_share → -0.525 (strong negative, meaning loans favor white neighborhoods)

Downtown and affluent North Side areas receive significantly more small business loans than South and West Side neighborhoods, even after controlling for population (loans per 1k households). Upper income neighborhoods get nearly DOUBLE the per-capita loan access compared to low/moderate income neighborhoods (82 vs 43 loans per 1k households).

This suggests barriers to loan access include:
- Fewer bank branches in majority non-white neighborhoods (legacy of redlining)
- Stricter lending criteria that disadvantage lower-income applicants or those with less credit history
- Less established business density in certain areas, creating a self-reinforcing cycle
- Geographic discrimination in lending patterns that persists despite CRA requirements

# Comparison to Grants:

The pattern is fundamentally DIFFERENT from what we found with grants:

Grants → Even coverage % across all neighborhoods (participation barriers, not program bias)
Loans → Systematically lower access in non-white and low-income areas (structural barriers + possible discrimination)

This means:
- Public grant programs (SBIF/NOF) appear to have neutral formulas
- Private sector lending shows strong geographic and demographic biases
- South/West Side neighborhoods face barriers to BOTH grants (fewer applications) AND loans (fewer approvals/originations)

The loan data reveals that Chicago's small business financing landscape has significant equity gaps that grants alone cannot solve.

# Next Steps

1. Compare loan + grant access together by community area to identify neighborhoods facing dual barriers
2. Analyze whether neighborhoods with low loan access have higher grant participation (compensation effect)
3. Investigate specific industries/business types in underserved areas to understand capital access needs
