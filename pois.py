import scipy.stats as stats

# Given values
lambda_rate = 400
k = 600

# Calculate the probability
probability = stats.poisson.pmf(k, lambda_rate)
print(f"Probability that 600 customers come in on any given Friday night: {probability:.10f}")