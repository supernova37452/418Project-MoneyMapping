# 418Project-MoneyMapping
This is a data science project for the purpose of exploring how how loans/grants in Chicago are distributed across neighborhoods.

# Part 1: Grant Precent and Findings: Looking at Amount of Coverage

This is essentially a new column made specifically for the grant programs:

GRANT_RATIO = INCENTIVE AMOUNT ÷ TOTAL PROJECT COST

This represents the share of each project’s total cost that was actually covered by the grant. It’s a key measure for understanding how much financial support projects receive from each program and will likely help us look at how amount of funding may be based on neighborhood socioeconomic dynamics (we will test this 2nd part later).

Here are some helpful statistics for the two types of grants below:

For SBIF Grants:

count = 2,150 projects with valid values (others had missing cost or incentive data)
mean ≈ 0.601 -> on average, ~60% of each project’s cost is covered by SBIF
median ≈ 0.572 -> about half of projects have ≤ ~57% coverage
IQR (25%–75%) = 0.50–0.75 -> the middle half of projects get 50–75% of costs covered
max = 1.0 -> some projects are fully covered (grant = total cost)
min ≈ 0.033 -> tiny coverage outliers (~3% subsidized)

NOTE: none of these exceed 1.0, which makes sense since grants can’t surpass total costs.

# What this means:

SBIF is clearly the more generous and widely used program. The average coverage (~60%) shows strong city participation in funding local business improvements. Many projects receive over half their total costs funded, suggesting SBIF is designed to make small upgrades financially realistic for business owners.

FOR NOF Grants:

count = 119 projects with valid data
mean ≈ 0.546 -> average coverage of ~55%
IQR (25%–75%) = 0.48–0.69 -> typical NOF projects cover ~48–69% of total costs
max = 1.0
min ≈ 0.017 → same range, much fewer high coverage projects


We have also displayed the total sum of $ that each neighborhood gets from both grants. this is just extra information and may be helpful.

# Part 2: Understanding chi_data.csv and Combining it with Grant Percent

Now, here is where we try to make sense of the data in chi_data.csv, which is a dataset that combines neighborhood level data. I have made 3 different datasets that are split for context clearness.

# Dataset 1: Demographic Shares

These columns describe the racial and ethnic makeup of each neighborhood.

white_hh_share, black_hh_share, latino_hh_share -> share of all households by race or ethnicity.
Example: black_hh_share = 0.95 means 95% of households are Black.

non_white_hh_share -> combinew all non-white households (Black, Latino, Asian, Other).
for this column, high values indicate majority non-white neighborhoods.

# Dataset 2: Ownership Shares

These columns describe who owns homes in each area.

white_own_shr, black_own_shr, latino_own_shr -> share of homeowners by race or ethnicity.

non_white_own_shr —> share of all homeowners who are people of color.

This helps show whether ownership patterns differ from who actually lives in the neighborhood.

# Dataset 3: Income and Grant Coverage (using the %!)

These columns connect neighborhood income levels with grant program outcomes:

ami_shr —> ratio of neighborhood median income compared to the regional median (Area Median Income, or AMI).

1.0 means equal to the regional median
values below 1.0 indicate lower income
values above 1.0 indicate higher income

income_level —> general income classification (LMI = Low/Moderate Income, MI = Moderate Income, UI = Upper Income).

low_inc —>if the area is low income.

GRANT_Percent_x —> average percent of project costs covered by SBIF grants.

GRANT_Percent_y —> average percent of project costs covered by NOF grants.

These indicators help assess whether lower income or majority non white neighborhoods receive higher or lower levels of financial support from city grant programs.

# What this means:

neighborhood demographics summary: Race and income are closely tied. Neighborhoods with higher non-white household shares tend to have much lower median incomes. As for ownership, white-majority neighborhoods have higher rates of homeownership and income. Predominantly Black and Latino areas on the South and West Sides show lower ownership and income levels.

Not shocking as this just reflects the patterns of segregation and unequal economic opportunity in Chicago.

This isn't super important right now but a good thing to keep in mind

# Grants and Neighborhood relationship! (this is important)

# What the SBIF numbers show

white_hh_share: +0.001 -> no relationship
black_hh_share: +0.045 -> tiny positive link
latino_hh_share: −0.045 -> tiny negative link
non_white_hh_share: −0.001 -> basically zero lol
ami_shr (income): +0.041 -> slightly higher coverage in higher-income areas, but a bit weak

# What this means:
SBIF coverage doesn’t vary by race or income...rich, poor, white, Black, Latino. The numbers show everyone’s getting roughly the same grant % of their total project cost.

meaning, the SBIF formula is neutral, it doesn’t give higher or lower coverage depending on where you are or who lives there.

# What the NOF numbers show

white_hh_share: −0.093 -> a bit lower coverage in whiter areas
black_hh_share: −0.164 -> a bit lower coverage in majority-Black areas
latino_hh_share: +0.237 -> somewhat higher coverage in Latino neighborhoods
non_white_hh_share: +0.093 -> slightly higher coverage overall in non-white neighborhoods
ami_shr (income): −0.003 -> no real relationship to income

# What this means:
NOF coverage is a little more generous (higher %) in Latino or mixed neighborhoods, but the correlation is weak ...it’s not a clear policy trend either.

# Summary of current findings and next steps (also very important)

The data shows that Chicago’s major grant programs are pretty even in how they allocate financial support. Even across neighborhoods with very different racial and income makeups, the % of a projects cost that is covered by a grant remains pretty consistent(between 50% and 70%.)

So, whether a community is mainly white, Black, or Latino, or whether its average income is above or below the city median, the level of support for each approved project looks nearly the same. As of right now, there is almost no correlation between grant coverage and demographic/ income. This likely mesans that the city’s grant formulas is not explicitly biased towards a group or area.

HOWEVER, wealthier or more active neighborhoods have more total projects, more applicants, and therefore receive a larger total dollar amount in city grants, even if each project is funded at the same rate as those in lower income areas. If we look at the count of communities for the grants, the South and West Sides, which face greater economic hardship according to the income levels, appear less frequently in the data. 

What does this mean? I would argue theres barriers in participation rather than unfair treatment in the grant structure itself. So yes , Chicagos grants are fair in how much they cover, but not everyone has the same level of access to the application or process of the grants themselves.
