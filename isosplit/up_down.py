import numpy as np
from sklearn.isotonic import isotonic_regression


# I don't want to use the name jisotonic since this is
# possibly not identical.

# Probably would be best to do a Cython wrapper of
# `jisotonic5.cpp`, but this is an easy impl relying
# on sklearn's isotonic regression, implementing
# the algorithms on page 25.

# if there is a bottleneck, it would be pava_mse.
# another idea is just to cythonize that function,
# which would be straightforward.


def up_down_isotonic_regression(x, weights=None):
    b = find_optimal_b(x, weights=weights)
    y1 = isotonic_regression(x[:b], increasing=True)
    y2 = isotonic_regression(x[b:], increasing=False)
    return np.vstack([y1, y2])


def down_up_isotonic_regression(x, weights=None):
    return -up_down_isotonic_regression(-x, weights=weights)


def find_optimal_b(x, weights=None):
    x1 = x
    w1 = weights
    x2 = -np.reverse(x)
    w2 = np.reverse(weights)
    mu1 = pava_mse(x1, weights=w1)
    mu2 = np.reverse(pava_mse(x2, weights=w2))
    return np.argmin(mu1 + mu2)


def pava_mse(x, weights=None):
    n = x.size
    i = 0
    j = 0

    count = np.zeros(n, dtype=np.int)
    wcount = np.zeros(n, dtype=np.float)
    sums = np.zeros(n, dtype=np.float)
    sumsqr = np.zeros(n, dtype=np.float)
    mu = np.zeros(n, dtype=np.float)

    wx = x * weights
    wx2 = wx * x

    count[i] = 1
    wcount[i] = weights[j]
    sums[i] = wx[j]
    sumsqr[i] = wx2[j]

    for j in range(1, n):
        i += 1
        count[i] = 1
        wcount[i] = weights[j]
        sums[i] = wx[j]
        sumsqr[i] = wx2[j]
        mu[j] = mu[j - 1]

        while i > 0:
            if sums[i - 1] / count[i] < sums[i] / count[i]:
                break

            mu_before = (
                sumsqr[i - 1]
                - np.square(sums[i - 1]) / count[i - 1]
                + sumsqr[i]
                - np.square(sums[i]) / count[i]
            )
            count[i - 1] += count[i]
            wcount[i - 1] += wcount[i]
            sums[i - 1] += sums[i]
            sumsqr[i - 1] += sumsqr[i]
            mu_after = sumsqr[i - 1] - np.square(sums[i - 1]) / count[i - 1]
            mu[j] += mu_after - mu_before
            i -= 1

    return mu
