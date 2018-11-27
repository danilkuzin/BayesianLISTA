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
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axvline(color='k', linestyle='--')
ax.axhline(color='k', linestyle='--')
ax.plot(x, y, linewidth=5)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
plt.tight_layout()
plt.savefig("spsl.pdf", format='pdf', dpi='figure')