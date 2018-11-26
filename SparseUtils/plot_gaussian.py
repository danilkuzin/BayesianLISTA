import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma, norm, bernoulli
import seaborn as sns

n = 1000

sigmas = np.zeros(n,)
betas = np.zeros(n,)

x = np.linspace(-2, 2, n)
y = norm.pdf(x)

plt.plot(x, y)
plt.savefig("gauss.pdf", format='pdf', dpi='figure')