import numpy as np
from itertools import chain
from .up_down import up_down_isotonic_regression, down_up_isotonic_regression


num_bins_factor = 1


# -- library


def drange(high, low=0, step=-1):
    return range(high + 1, low - 1, step)


def updown_arange(num_bins, dtype=np.int):
    num_bins_1 = int(np.ceil(num_bins / 2))
    num_bins_2 = num_bins - num_bins_1
    return np.fromiter(
        chain(
            range(1, num_bins_1),
            drange(num_bins_2, 1),
        ),
        count=num_bins,
        dtype=dtype,
    )


def compute_ks4(counts1, counts2):
    c1s = counts1.sum()
    c2s = counts2.sum()
    s1 = np.cumsum(counts1)
    s1 /= c1s
    s2 = np.cumsum(counts2)
    s2 /= c2s
    ks = np.abs(s1 - s2).max()
    ks *= np.sqrt((c1s + c2s) / 2)
    return ks


def compute_ks5(counts1, counts2):
    best_ks = -np.inf
    length = counts1.size

    while length >= 4 or length == counts1.size:
        ks = compute_ks4(counts1[0:length], counts2[0:length])
        if ks > best_ks:
            best_ks = ks
            best_length = length
        length //= 2

    return best_ks, best_length


# -- main


def isocut5(samples, sample_weights=None):
    assert samples.ndim == 1
    N = samples.size
    assert N > 0
    num_bins = int(np.ceil(np.sqrt(N / 2) * num_bins_factor))

    if sample_weights is None:
        sample_weights = np.ones(N)

    sort = np.argsort(samples)
    X = samples[sort]
    sample_weights = sample_weights[sort]
    del sort

    while True:
        intervals = updown_arange(num_bins, dtype=np.float)
        alpha = (N - 1) / intervals.sum()
        intervals *= alpha
        # this line is the only one to translate to 0-based
        inds = np.floor(np.hstack([[0], np.cumsum(intervals)])).astype(np.int)
        # N_sub = inds.size
        if intervals.min() >= 1:
            break
        else:
            num_bins -= 1
    del intervals

    cumsum_sample_weights = np.cumsum(sample_weights)
    X_sub = X[inds]
    spacings = np.diff(X_sub)
    multiplicities = np.diff(cumsum_sample_weights[inds])
    densities = multiplicities / spacings

    densities_unimodal_fit = up_down_isotonic_regression(
        densities, multiplicities
    )
    peak_ind = np.argmax(densities_unimodal_fit)
    ks_left, ks_left_ind = compute_ks5(
        multiplicities[0:peak_ind + 1],
        densities_unimodal_fit[0:peak_ind + 1] * spacings[0:peak_ind + 1],
    )
    # possibly this logic with the reversing is a matlab-ism
    # that could be replaced by [peak_ind:] here, but it is
    # kind of hard to follow all of the logic with the ks.
    ks_right, ks_right_ind = compute_ks5(
        multiplicities[peak_ind:][::-1],
        densities_unimodal_fit[peak_ind:][::-1]
        * spacings[peak_ind:][::-1],
    )
    ks_right_ind = spacings.size - ks_right_ind

    if ks_left > ks_right:
        critical_range = range(ks_left_ind)
        dipscore = ks_left
    else:
        critical_range = range(ks_right_ind, spacings.size)
        dipscore = ks_right

    densities_resid = (
        densities[critical_range] - densities_unimodal_fit[critical_range]
    )
    densities_resid_fit = down_up_isotonic_regression(
        densities_resid, spacings[critical_range]
    )
    cutpoint_ind = critical_range.start + np.argmin(densities_resid_fit)
    cutpoint = (X_sub[cutpoint_ind] + X_sub[cutpoint_ind + 1]) / 2

    return dipscore, cutpoint

