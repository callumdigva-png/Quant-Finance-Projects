import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\callu\OneDrive\Documents\Coding Projects\Loan_Data\Task_3_and_4_Loan_Data.csv")

# aggregate by FICO score
fico_grouped = df.groupby('fico_score').agg(
    n=('default', 'count'),
    k=('default', 'sum')
).reset_index()

def log_likelihood_quantization(fico_scores, n_counts, k_defaults, n_buckets):

    sorted_indices = np.argsort(fico_scores)
    scores = np.array(fico_scores)[sorted_indices]
    n_counts = np.array(n_counts)[sorted_indices]
    k_defaults = np.array(k_defaults)[sorted_indices]
    n = len(scores)

    ll_table = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            n_bucket = np.sum(n_counts[i:j+1])   # FIXED
            k = np.sum(k_defaults[i:j+1])
            p = k / n_bucket
            if p == 0 or p == 1:
                ll_table[i][j] = -np.inf
            else:
                ll_table[i][j] = n_bucket * np.log(p) + (n_bucket - k) * np.log(1 - p)

    dp = np.full((n_buckets + 1, n + 1), -np.inf)
    dp[0][0] = 0
    dp_split = np.zeros((n_buckets + 1, n + 1), dtype=int)

    for k in range(1, n_buckets + 1):
        for i in range(k, n + 1):
            for j in range(k-1, i):
                cost = dp[k-1][j] + ll_table[j][i-1]
                if cost > dp[k][i]:
                    dp[k][i] = cost
                    dp_split[k][i] = j

    boundaries = []
    i = n
    for k in range(n_buckets, 0, -1):
        j = dp_split[k][i]
        boundaries.append(scores[j])
        i = j
    boundaries = sorted(boundaries)

    return dp[n_buckets][n], boundaries

# run on full aggregated data
result, boundaries = log_likelihood_quantization(
    fico_grouped['fico_score'],
    fico_grouped['n'],
    fico_grouped['k'],
    5
)

all_boundaries = [fico_grouped['fico_score'].min()] + boundaries + [fico_grouped['fico_score'].max()]
print("Rating Map with Default Rates:")
for i in range(len(all_boundaries)-1):
    mask = (df['fico_score'] >= all_boundaries[i]) & (df['fico_score'] < all_boundaries[i+1])
    n = mask.sum()
    k = df['default'][mask].sum()
    if n > 0:
        print(f"  Rating {i+1}: FICO {all_boundaries[i]:.0f}-{all_boundaries[i+1]:.0f} | {n} borrowers | {k/n:.0%} default rate")

low_fico = fico_grouped[fico_grouped['fico_score'] < 600]
high_fico = fico_grouped[fico_grouped['fico_score'] >= 600]

result_low, boundaries_low = log_likelihood_quantization(
    low_fico['fico_score'], low_fico['n'], low_fico['k'], 3)

result_high, boundaries_high = log_likelihood_quantization(
    high_fico['fico_score'], high_fico['n'], high_fico['k'], 3)

all_boundaries = sorted(boundaries_low + boundaries_high)
print(f"Combined boundaries: {all_boundaries}")