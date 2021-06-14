import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import squareform, pdist

from .parcelate2 import parcelate2
from .isocut5 import isocut5


# -- library


def compute_centers(X, labels):
    # python has no accumarray, so heres one way to do it
    K = labels.max() + 1
    # hstack so cluster indexes the columns
    return np.hstack([X[labels == k].mean(axis=1) for k in range(K)])


def get_pairs_to_compare(
    centers, comparisons_made, one_comparison_at_a_time=False
):
    # centers is dimension x n_centers
    n_centers = centers.shape[1]
    inds1 = []
    inds2 = []
    pair_dists = []

    dists = squareform(pdist(centers.T))
    dists[comparisons_made != 0] = np.inf
    np.fill_diagonal(dists, np.inf)

    # take mutual closest pairs
    best_inds = np.argmin(dists, axis=0)
    for j in range(n_centers):
        bij = best_inds[j]
        if bij > j:
            # mutual?
            if best_inds[bij] == j:
                if dists[j, bij] < np.inf:
                    inds1.append(j)
                    inds2.append(bij)
                    pair_dists.append(dists[j, bij])
                    dists[j, :] = dists[:, j] = np.inf
                    dists[bij, :] = dists[:, bij] = np.inf

    if one_comparison_at_a_time and len(inds1) > 1:
        ii = np.argmin(pair_dists)
        inds1 = [inds1[ii]]
        inds2 = [inds2[ii]]

    return inds1, inds2


def compare_pairs(
    X,
    labels,
    k1s,
    k2s,
    isocut_threshold=1,
    min_cluster_size=10,
    whiten_cluster_pairs=True,
):
    Kmax = labels.max() + 1
    clusters_changed_vec = np.zeros(Kmax, dtype=np.bool)
    new_labels = labels + 0

    for k1, k2 in zip(k1s, k2s):
        inds1 = np.flatnonzero(labels == k1)
        inds2 = np.flatnonzero(labels == k2)
        sz1 = inds1.size
        sz2 = inds2.size
        if sz1 > 0 and sz2 > 0:
            if sz1 < min_cluster_size or sz2 < min_cluster_size:
                do_merge = True
            else:
                inds12 = np.vstack([inds1, inds2])
                L12_old = np.zeros(inds12.size, dtype=np.bool)
                L12_old[sz1:] = 1
                do_merge, L12, proj, cutpoint = merge_test(
                    X[:, inds1],
                    X[:, inds2],
                    whiten_cluster_pairs=whiten_cluster_pairs,
                )

            if do_merge:
                new_labels[new_labels == k2] = k1
                clusters_changed_vec[k1] = clusters_changed_vec[k2] = 1
            else:
                # redistribute
                if (L12 != L12_old).any():
                    # recall, L12 is bool array indicating cluster 2 here
                    new_labels[inds12[~L12]] = k1
                    new_labels[inds12[L12]] = k2
                    clusters_changed_vec[k1] = clusters_changed_vec[k2] = 1

    clusters_changed = np.flatnonzero(clusters_changed_vec)
    return new_labels, clusters_changed


def merge_test(X1, X2, isocut_threshold=1, whiten_cluster_pairs=True):
    if whiten_cluster_pairs:
        X1, X2, V = whiten_two_clusters_b(X1, X2)
    else:
        c1 = X1.mean(axis=1)
        c2 = X2.mean(axis=1)
        V = c2 - c1
        V /= np.sqrt(V.T @ V)

    n1 = X1.shape[1]
    n2 = X2.shape[1]
    if n1 == 0 or n2 == 0:
        raise ValueError("error in merge test: empty input")

    proj1 = V @ X1
    proj2 = V @ X2
    proj12 = np.hstack([proj1, proj2])

    dipscore, cutpoint = isocut5(proj12, np.ones(n1 + n2))

    do_merge = dipscore < isocut_threshold
    new_labels = np.zeros(n1 + n2, dtype=np.bool)
    new_labels[proj12 > cutpoint] = 1

    return do_merge, new_labels, proj12, cutpoint


def whiten_two_clusters_b(X1, X2):
    # this is what is used, but it does not seem to me that
    # there is any whitening going on. not totally sure I
    # follow the average covariance thing, but here goes.
    M, N1 = X1.shape
    N2 = X2.shape[1]
    c1 = X1.mean(axis=1, keepdims=True)
    c2 = X2.mean(axis=1, keepdims=True)
    X1c = X1 - c1
    X2c = X2 - c2
    cov1 = (X1c @ X1c.T) / N1
    cov2 = (X2c @ X2c.T) / N2
    avg_cov = (cov1 + cov2) / 2
    V = c2 - c1
    if np.abs(la.det(avg_cov)) > 1e-6:
        V = la.inv(avg_cov) @ V
    V /= np.sqrt(V.T @ V)
    return X1, X2, V


# -- main


def isosplit5(
    X,
    isocut_threshold=1,
    min_cluster_size=10,
    K_init=200,
    refine_clusters=True,
    max_iterations_per_pass=500,
    whiten_cluster_pairs=True,
    initial_labels=None,
    prevent_merge=False,
    one_comparison_at_a_time=False,
):
    # dict to be returned with info about the run
    # info = dict(iterations=[])

    # M is features, N is samples
    M, N = X.shape

    # simplification in 1d
    if M == 1:
        whiten_cluster_pairs = False

    # -- compute initial labels
    if initial_labels is None:
        labels = parcelate2(
            X,
            target_parcel_size=min_cluster_size,
            target_num_parcels=K_init,
        )
    else:
        # matlab code has some logic here; I will defer to the caller to know
        # what is going on.
        labels = np.asarray(initial_labels, dtype=np.int)
        assert labels.shape == (N,)

    # original number of labels
    Kmax = labels.max() + 1

    # -- compute cluster centers
    centers = compute_centers(X, labels)

    # -- main loop
    final_pass = False
    comparisons_made = np.zeros((Kmax, Kmax))
    # outer loop -- passes
    while True:
        # if not, final pass
        something_merged = False

        # keep track of clusters that changed in this pass
        # so that we can update comparisons_made at the end
        clusters_changed_vec_in_pass = np.zeros(Kmax)

        # inner loop -- iterations
        iteration = 0
        while True:
            # for diagnostic purposes
            # old_labels = labels

            iteration += 1
            if iteration > max_iterations_per_pass:
                raise ValueError("max iterations per pass exceeded")

            # active labels are those which are still being used
            active_labels_vec = np.zeros(Kmax, dtype=np.bool)
            active_labels_vec[labels] = 1
            active_labels = np.flatnonzero(active_labels_vec)
            active_centers = centers[:, active_labels]

            # find the pairs to compare on this iteration
            # these will be the closest pairs of active clusters
            # that have not yet been compared in this pass
            inds1, inds2 = get_pairs_to_compare(
                active_centers,
                comparisons_made[active_labels, active_labels],
                one_comparison_at_a_time=one_comparison_at_a_time,
            )

            # if we didn't find any, break from this iteration
            if not inds1:
                break

            # compare the pairs
            labels, clusters_changed = compare_pairs(
                X,
                labels,
                active_labels[inds1],
                active_labels[inds2],
                isocut_threshold=isocut_threshold,
                min_cluster_size=min_cluster_size,
                whiten_cluster_pairs=whiten_cluster_pairs,
            )
            clusters_changed_vec_in_pass[clusters_changed] = 1

            # update which comparisons have been made
            for i1, i2 in zip(inds1, inds2):
                comparisons_made[active_labels[i1], active_labels[i2]] = 1
                comparisons_made[active_labels[i2], active_labels[i1]] = 1

            # recompute centers
            # matlab contains note: maybe this should only apply to
            # those that changed?
            centers = compute_centers(X, labels)

            # for diagnostics, cound the number of changes
            # total_num_label_changes = (labels != old_labels).sum()

            # determine whether something has merged
            # note! matlab has zeros(1,N), should it be zeros(1,Kmax)?
            new_active_labels_vec = np.zeros(Kmax, dtype=np.bool)
            new_active_labels_vec[labels] = 1
            if new_active_labels_vec.sum() < active_labels.size:
                something_merged = True

        # back in outer loop -- passes.

        # zero out the comparisons made matrix only for those
        # that have changed
        clusters_changed = np.flatnonzero(clusters_changed_vec_in_pass)
        for cc in clusters_changed:
            comparisons_made[cc, :] = comparisons_made[:, cc] = 0

        if something_merged:
            final_pass = False

        # this was the final pass and nothing merged
        if final_pass:
            break

        # if we are done, do one last pass for final redistributes
        final_pass = something_merged

    # remap labels to be contiguous
    unique_labels, labels = np.unique(labels, return_inverse=True)
    K = unique_labels.size

    # if the user wants to refine the clusters, then repeat
    # isosplit on each of the new clusters, recursively,
    # unless we only found one cluster
    if refine_clusters and K > 1:
        labels_split = np.zeros(N, dtype=np.int)
        K_split = 0
        for k in range(K):
            inds_k = np.flatnonzero(labels == k)
            labels_k = isosplit5(
                X[inds_k],
                isocut_threshold=isocut_threshold,
                min_cluster_size=min_cluster_size,
                K_init=K_init,
                refine_clusters=refine_clusters,
                max_iterations_per_pass=max_iterations_per_pass,
                whiten_cluster_pairs=whiten_cluster_pairs,
                initial_labels=initial_labels,
                prevent_merge=prevent_merge,
                one_comparison_at_a_time=one_comparison_at_a_time,
            )
            labels_split[inds_k] = K_split + labels_k
            K_split += labels_k.max() + 1

    return labels
