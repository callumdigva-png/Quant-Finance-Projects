import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\callu\OneDrive\Documents\Coding Projects\Loan_Data\Task_3_and_4_Loan_Data.csv")

def mse_quantization(fico_scores, n_buckets):
    
    # Step 1: sort the scores
    scores = np.sort(fico_scores)
    n = len(scores)
    
    # Step 2: precompute MSE for every possible bucket
    mse_table = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            bucket = scores[i:j+1]
            mean = np.mean(bucket)
            mse_table[i][j] = np.sum((bucket - mean) ** 2)
    
    # Step 3: dynamic programming tables
    dp = np.full((n_buckets + 1, n + 1), np.inf)
    dp[0][0] = 0
    dp_split = np.zeros((n_buckets + 1, n + 1), dtype=int)  # NEW
    
    # Step 4: fill in the table
    for k in range(1, n_buckets + 1):
        for i in range(k, n + 1):
            for j in range(k-1, i):
                cost = dp[k-1][j] + mse_table[j][i-1]  # CHANGED
                if cost < dp[k][i]:                      # CHANGED
                    dp[k][i] = cost                      # CHANGED
                    dp_split[k][i] = j                   # NEW

    # Step 5: traceback to find boundaries
    boundaries = []
    i = n
    for k in range(n_buckets, 0, -1):
        j = dp_split[k][i]
        boundaries.append(scores[j])
        i = j
    boundaries = sorted(boundaries)
    
    return dp[n_buckets][n], boundaries

# test on small sample first
sample_scores = df['fico_score'].sample(100, random_state=42)
result, boundaries = mse_quantization(sample_scores, 5)
print(f"Minimum MSE: {result:.2f}")
print(f"Bucket boundaries: {boundaries}")
print("\nRating Map:")
boundaries = [df['fico_score'].min()] + boundaries + [df['fico_score'].max()]
for i in range(len(boundaries)-1):
    print(f"  Rating {i+1}: FICO {boundaries[i]:.0f} - {boundaries[i+1]:.0f}")