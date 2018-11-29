import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    n = 1000

    x = np.linspace(-2, 2, n)
    y = np.zeros_like(x)

    y[int(n / 2)] = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axvline(color='k', linestyle='--')
    ax.axhline(color='k', linestyle='--')
    ax.plot(x, y, linewidth=5)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.tight_layout()
    plt.savefig("delta.pdf", format='pdf', dpi='figure')