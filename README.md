# 418Project-MoneyMapping
This is a data science project for the purpose of exploring how how loans/grants in Chicago are distributed across neighborhoods.

Explaining current code and findings:

This is essentially a new column I made specifically for the grant programs (SBIF and NOF):

GRANT_RATIO = INCENTIVE AMOUNT ÷ TOTAL PROJECT COST

This is important because it tells us what share of each project's TOTAL cost was actually covered by the grant (the statistics for both NOF and SBIF are below).

For SBIF Grants:
count = 2,150 projects have both INCENTIVE AMOUNT ÷ TOTAL PROJECT COST, the rest were null
mean ≈ 0.601 -> on average, ~60% of a project’s cost is covered by SBIF
median ≈ 0.572 -> about half of projects have ≤ ~57% coverage
25% = 0.50; 75% = 0.75 -> middle half of projects get 50%–75% of costs covered
max = 1.0 -> some projects are fully covered! (grant equals total cost)
min ≈ 0.033 -> tiny coverage outliers (like 3.3% subsidized)

FOR NOF Grants:

count = 119
mean ≈ 0.546 -> so average ~55% coverage
Quartiles ~0.48–0.69 -> typical NOF projects cover ~48–69%
max = 1.0 -> some fully covered as well
min ≈ 0.017 -> tiny coverage cases exist here too

Note: because im calculating ratio here, none of these should be over > 1.00

So it looks like the majority of applicants are getting some form of coverage but this grant ratio column is useful because now we can see which neighborhoods get better coverage and start looking at the socioeconomic statuses accross these neighborhoods.

More importantly:

SBIF tends to subsidize a larger share of project costs (remember the median ~57%, IQR 50–75%) than NOF (median ~52%, IQR ~48–69%). Not only do the numbers back this up but the # of applicants themselves. this is the more well known grant.

Although I did filter out null SBIF rows, we have more than enough to still do community-level comparisons (so doing maybe the same for CMI mircoloans and then start looking at the socioenomic stuff)
