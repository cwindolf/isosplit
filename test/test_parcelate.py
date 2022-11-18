import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from isosplit.parcelate2 import parcelate2

rg = np.random.default_rng(0)

z = rg.standard_normal(size=(2, 10000))

labels = parcelate2(z, 100, 100)

sns.scatterplot(x=z[0, :], y=z[1, :], hue=labels)
plt.show()
