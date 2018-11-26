import numpy as np
import matplotlib.pyplot as plt


def soft_thresholding(x, lam):
    return np.sign(x) * (np.abs(x) - lam) * (np.abs(x) > lam)


if __name__=="__main__":
    x = np.linspace(-2, 2, 100)
    lam = 0.75
    y = soft_thresholding(x, lam)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axvline(color='k', linestyle='--')
    ax.axhline(color='k', linestyle='--')
    ax.plot(x, y, linewidth=5)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.tight_layout()
    plt.savefig("s_thr.svg")