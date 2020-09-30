import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as api


def generate_autocorrelated_data(theta, mu, sigma, N):
    X = np.zeros((N,1))
    
    for t in range(1,N):
        X[t] = theta*X[t-1] + np.random.normal(mu, sigma)
    return X

sample_means = np.zeros(200-1)
for i in range(1,200):
    X = generate_autocorrelated_data(0.5, 0, 1 , i*10)
    sample_means[i-1] = np.mean(X)

plt.bar(range(1,200), sample_means)
plt.show()