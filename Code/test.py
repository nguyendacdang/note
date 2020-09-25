import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as api


np.random.seed(123)
X = np.random.normal(loc = 0.0, scale = 0.1 , size = 100)
print(X.mean())
print(X.std())