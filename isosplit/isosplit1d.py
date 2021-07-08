import numpy as np
from isosplit import isocut


def isosplit1d(x, isocut_threshold=1.0, min_size=10, min_diameter=0.0):
    """Iso-Split 1D

    This is a natural specialization of ISO-SPLIT to the 1d case. The
    idea:     recursively divide the domain of the input at putative
    cut-points, using     ISO-CUT, until none of these are significant
    (or they would be too small according to `min_size` or `min_diameter`,
    see parameter documentation below.)

    This is deterministic and optimal in the case where `min_size` and
    `min_diameter` are 0. If those are nonzero, it is possible that this
    algorithm could miss a significant cut point, since ISO-CUT finds
    only the most significant cut point: if that cut point produces a
    cluster which is too small but there exists another (less, but
    still) significant cut point which would not, that other cut point
    will not be discovered.

    This function will return the `K - 1` `cutpoints` which divide the
    data into `K` clusters, as well as the cluster assignments `y`. The
    `cutpoints` can be used for classification of out-of-bag data.

    Parameters
    ----------
    x : numpy array of shape (N,), (1, N), or (N, 1)
        The one-dimensional data to be clustered. Will be converted to
        double precision if it is not already.
    isocut_threshold : float
        Critical value for the dip score test statistic.
    min_size : int
        Minimum cardinality of output clusters.
    min_diameter : float
        Minimum spatial extent of a cluster, i.e., the minimum distance
        between two neighboring cutpoints.

    Returns
    -------
    y : integer array of cluster assignments
    cutpoints : floating array of cutpoints
    """
    # jisotonic5.pyx expects double precision, so make sure we have it
    x = np.asarray(x, dtype=np.float)
    assert x.ndim == 1 or (x.ndim == 2 and 1 in x.shape)
    x = np.squeeze(x)

    # ith label corresponds to the gap between endpoint i and endpoint i+1
    # labels are implicit since they would need to be adjusted repeatedly
    # to maintain that invariant
    labels_to_process = [0]
    endpoints = [-np.inf, np.inf]

    while labels_to_process:
        # process a label
        i = labels_to_process.pop()
        lo = x > endpoints[i]
        hi = x <= endpoints[i + 1]
        dipscore, cutpoint = isocut(x[lo & hi])
        mid = x <= cutpoint

        # test spatial size criterion of both putative clusters
        wide_enough = (
            (cutpoint - endpoints[i]) >= min_diameter
            and (endpoints[i + 1] - cutpoint) >= min_diameter
        )

        # test cardinality criterion of both putative clusters
        large_enough = (
            (lo & mid).sum() >= min_size
            or (~mid & hi).sum() >= min_size
        )

        # if significant and size tests came back okay...
        if dipscore >= isocut_threshold and wide_enough and large_enough:
            # split the cluster
            # first of all, we need to bump everyone above up
            labels_to_process = [
                j if j < i else j + 1
                for j in labels_to_process
            ]

            # put modified labels back onto the stack
            labels_to_process.append(i)
            labels_to_process.append(i + 1)

            # and, we have a new cut point
            endpoints.insert(i + 1, cutpoint)

    # the K+1 endpoints correspond to K-1 cutpoints, which we will return
    cutpoints = np.array(endpoints[1:-1])

    # and, compute the cluster assignments for the caller
    y = np.zeros(x.size, dtype=np.int)
    lower_mask = np.zeros(x.size, dtype=np.bool)
    for k, upper_bound in enumerate(endpoints[1:]):
        upper_mask = x <= upper_bound
        y[~lower_mask & upper_mask] = k
        lower_mask = upper_mask

    return y, cutpoints
