import numpy as np

np.random.seed(123)
test = np.random.rand(6)
print(test)
a = (test[3] + test[-1])/2
print(a)
print(np.median(test))