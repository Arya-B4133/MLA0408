import numpy as np

np.random.seed(42)

true_mean = 50
true_std = 5

data = np.random.normal(true_mean, true_std, 1000)

estimated_mean = np.mean(data)
estimated_variance = np.var(data)

print("Actual Mean:", true_mean)
print("Estimated Mean:", round(estimated_mean,2))

print("Actual Variance:", true_std**2)
print("Estimated Variance:", round(estimated_variance,2))
