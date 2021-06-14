import numpy as np
from isosplit.up_down import up_down_isotonic_regression


# -- test jisotonic


def is_updown(y):
    ydiff = np.diff(y)
    return any(
        (ydiff[:k] >= 0).all() and (ydiff[k:] <= 0).all()
        for k in range(len(ydiff) + 1)
    )


def test_case(x, w=None):
    x = np.asarray(x, dtype=float)
    if w is None:
        w = np.ones_like(x)
    y = up_down_isotonic_regression(x, w)
    g = "updown" if is_updown(y) else "bad"
    print(f"{x} -> {y}, {g}")


# test
test_case([1, 2, 3, 2, 1])
test_case([1, 1, 3, 2, 1])
test_case([1, 0, 3, 2, 3])
test_case([1, 2, 3, 4, 5])
test_case([5, 4, 3, 2, 1])
test_case([3, 2, 1, 2, 3])
