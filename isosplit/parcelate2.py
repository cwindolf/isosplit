import numpy as np
from scipy.spatial.distance import cdist


# not implementing `final_reassign` since it looks unused


# I wonder why a split factor around 2.71 is in a sense ideal
split_factor = 3


def parcelate2(
    X,
    target_parcel_size,
    target_num_parcels,
):
    assert target_parcel_size >= split_factor
    M, N = X.shape

    # start with one parcel
    centroids = [X.mean(axis=1, keepdims=True)]
    radii = [cdist(centroids[0].T, X.T).max()]
    indices = [np.arange(N)]
    labels = np.zeros(N, dtype=int)

    while len(centroids) < target_num_parcels:
        parcel_sizes = [p.size for p in indices]

        if not any(
            r > 0 and s > target_parcel_size
            for r, s in zip(radii, parcel_sizes)
        ):
            # nothing else will ever be split
            break

        # the target radius is a bit smaller than the largest current
        target_radius = 0.95 * max(
            r for r, s in zip(radii, parcel_sizes) if s > target_parcel_size
        )

        # step through each parcel
        something_changed = False
        p_index = 0
        while p_index < len(centroids):
            inds = indices[p_index]
            rad = radii[p_index]

            if inds.size > target_parcel_size and rad >= target_radius:
                # take the first `split_factor` points as representatives
                pts = X[:, inds[:split_factor]]
                D = cdist(pts.T, X[:, inds].T)
                assignments = D.argmin(axis=0)
                keepers = np.flatnonzero(assignments == 0)
                if keepers.size and not keepers.size == inds.size:
                    something_changed = True

                    # replace current parcel with keepers
                    kinds = indices[p_index] = inds[keepers]
                    centroids[p_index] = X[:, kinds].mean(
                        axis=1, keepdims=True
                    )
                    radii[p_index] = cdist(
                        centroids[p_index].T, X[:, kinds].T
                    ).max()
                    labels[kinds] = p_index

                    # add new parcels
                    for j in range(1, split_factor):
                        jeepers = np.flatnonzero(assignments == j)
                        jinds = inds[jeepers]
                        indices.append(jinds)
                        centroids.append(
                            X[:, jinds].mean(axis=1, keepdims=True)
                        )
                        radii.append(
                            cdist(centroids[-1].T, X[:, jinds].T).max()
                        )
                        labels[jinds] = len(centroids) - 1
                else:
                    print("warn: issue splitting parcel")
            else:
                p_index += 1

        if not something_changed:
            break

    return labels
