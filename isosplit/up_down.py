import numpy as np
from .jisotonic5 import jisotonic5


def up_down_isotonic_regression(x, weights=None):
    # determine switch point
    _, mse1 = jisotonic5(x, weights)
    _, mse2 = jisotonic5(x[::-1].copy(), weights[::-1].copy())
    mse2 = mse2[::-1]
    mse0 = mse1 + mse2
    best_ind = mse0.argmin()

    # regression
    if best_ind > 0:
        y1, _ = jisotonic5(x[:best_ind], weights[:best_ind])
    if best_ind < x.size:
        y2, _ = jisotonic5(-x[best_ind:], weights[best_ind:])
        y2 = -y2

    if best_ind == 0:
        return y2
    elif best_ind == x.size:
        return y1

    return np.hstack([y1, y2])


def down_up_isotonic_regression(x, weights=None):
    return -up_down_isotonic_regression(-x, weights=weights)
