import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma, norm, bernoulli
import seaborn as sns

n = 1000
pi = 0.33

sigmas = np.zeros(n,)
betas = np.zeros(n,)

x = np.linspace(-2, 2, n)
y = pi*norm.pdf(x)

y[int(n / 2)] = (1-pi)
plt.plot(x, y)
plt.savefig("spsl.pdf", format='pdf', dpi='figure')