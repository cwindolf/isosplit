import numpy as np
import matplotlib.pyplot as plt

from isosplit.isocut5 import isocut5

rg = np.random.default_rng(0)

n_trials = 100

weights = np.ones(1300)

dipscores = np.zeros(n_trials)
cutpoints = np.zeros(n_trials)

for trial in range(n_trials):
    X = np.hstack([
        rg.standard_normal(size=1000),
        5 + rg.standard_normal(size=300),
    ])
    dipscore, cutpoint = isocut5(X, weights)
    dipscores[trial] = dipscore
    cutpoints[trial] = cutpoint

plt.hist(cutpoints)
plt.title(f"cutpoints avg={cutpoints.mean():0.3g}")
plt.show()

plt.hist(dipscores)
plt.title(f"dipscores avg={dipscores.mean():0.3g}")
plt.show()
