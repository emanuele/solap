"""Sub-Optimal solution to the Linear Assignment Problem (SOLAP) for
large datasets.
"""

import numpy as np
# from scipy.spatial import KDTree as Tree
# from scipy.spatial import cKDTree as Tree
from sklearn.neighbors import KDTree as Tree
# from sklearn.neighbors import BallTree as Tree
from time import time

try:
    from joblib import Parallel, delayed, cpu_count
    joblib_available = True
except ImportError:
    joblib_available = False


tree = None  # container for the shared Tree(B)


def test_unique_rows(a):
    """Test wether an array 'a' has unique rows (no repetitions) or not.
    """
    # See: http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    unique_rows = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1]).shape[0]
    return (a.shape[0] == unique_rows)


def worker_query(A_chunk, k):
    """Basic worker performing the Tree query on the global tree for a
    chunk of data. Useful for tree_parallel_query().
    """
    global tree
    D, I = tree.query(A_chunk, k)
    return D.astype(np.float32), I.astype(np.int32)  # this saves memory for large datasets


def worker_query_radius(A_chunk, r):
    """Basic worker performing the Tree query_radius on the global tree
    for a chunk of data.  Useful for tree_parallel_query().
    """
    global tree
    D, I = tree.query_radius(A_chunk, r=r, return_distance=True)
    return [d.astype(np.float32) for d in D], [i.astype(np.int32) for i in I]  # this saves memory for large datasets


def estimate_radius(tree, A, k, subset_size=1000):
    """Estimate the radius r for a Tree.query_radius(A, r) that will
    return approximately k neighbors per point.
    """
    if A.shape[0] > subset_size:
        A = A[np.random.permutation(A.shape[0])[:subset_size], :]  # subsampling

    d, i = tree_parallel_query(tree, A, k)
    r = d[:, -1].mean()
    print("Estimated radius to get %s neighbors on average: %s" % (k, r))
    return r


def tree_parallel_query(my_tree, A, k=None, r=None, n_jobs=-1, query_radius=False):
    """Parallel query of the global Tree 'tree'.
    """
    global tree
    tree = my_tree
    tmp = cpu_count()
    if (n_jobs is None or n_jobs == -1) and A.shape[0] >= tmp:
        n_jobs = tmp

    if n_jobs > 1:
        tmp = np.linspace(0, A.shape[0], n_jobs + 1).astype(np.int)
    else:  # corner case: joblib detected 1 cpu only.
        tmp = (0, A.shape[0])

    chunks = zip(tmp[:-1], tmp[1:])
    print("chunks: %s" % chunks)
    if query_radius:
        if r is None:
            r = estimate_radius(tree, A, k)

        results = Parallel(n_jobs=n_jobs)(delayed(worker_query_radius)(A[start:stop, :], r) for start, stop in chunks)
        D, I = zip(*results)
        D = np.concatenate(D)
        I = np.concatenate(I)
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(worker_query)(A[start:stop, :], k) for start, stop in chunks)
        worker = worker_query
        D, I = zip(*results)
        D = np.vstack(D)
        I = np.vstack(I)

    return D, I


def compute_D_I(A, B, k=None, r=None, parallel=True, query_radius=False):
    """Compute the distances (D) and the IDs of the neighbors (I) between
    the set of vectors A and the set of vectors B, considering just k
    neighbors or the neighbors within radius r. This computation is
    based on the use of a Tree, e.g. KDTree, BallTree.
    """
    global joblib_available
    global tree
    print("Computing the tree (size=%s)." % B.shape[0])
    t0 = time()
    tree = Tree(B)
    print("%s sec" % (time()-t0))
    print("Computing %s NN queries, each for %s neighbors." % (A.shape[0], k))
    t0 = time()
    if joblib_available and parallel:
        D, I = tree_parallel_query(tree, A, k=k, r=r, query_radius=query_radius)
    else:
        if query_radius:
            D, I = tree.query_radius(A, r=r, return_distance=True)
        else:
            D, I = tree.query(A, k=k)

    print("%s sec" % (time()-t0))
    return D, I


def compute_solap(A, B, k=10, maxvalue=1e6, parallel=True):
    """Compute a scalable (greedy) Sub-Optimal solution to the Linear
    Assignment Problem.
    """
    assert(A.shape[0] <= B.shape[0])
    assert(test_unique_rows(A))
    assert(test_unique_rows(B))
    sizeA = A.shape[0]
    sizeB = B.shape[0]
    D, I = compute_D_I(A, B, k, parallel=parallel)

    print("Suboptimal assignment problem.")
    t0 = time()
    assignment = np.zeros(sizeA, dtype=np.int)
    nA = sizeA
    nB = sizeB
    not_assigned_A = np.ones(nA, dtype=np.bool)
    not_assigned_B = np.ones(nB, dtype=np.bool)
    map_id_A = np.arange(nA, dtype=np.int)
    map_id_B = np.arange(nB, dtype=np.int)
    loss = 0.0
    # for each streamline in A:
    for i in range(nA):
        if (i % 100) == 0:
            print("iteration %s" % i)

        # find the minimum distance in D:
        tmp = D.argmin()
        # find row and column of the minimum in D. The row (bestA) is
        # the id of the streamline in A, the column (col) needs to be
        # queried in I in order to find the corresponding id of the
        # stremaline in B (bestB):
        bestA, col = np.unravel_index(tmp, (nA, min(k, nB)))  # min() is clear only after looking below
        bestB = I[bestA, col]
        partial_loss = D[bestA, col]
        # Here we store the assignment just found:
        assignment[map_id_A[bestA]] = map_id_B[bestB]
        loss += partial_loss
        # We remove bestA and bestB from future assigments
        not_assigned_A[bestA] = False
        not_assigned_B[bestB] = False
        # The row of bestA in D is removed by putting high values inside:
        D[bestA, :] = maxvalue
        # All the entries of I poinring to bestB are removed in D:
        removed = np.where(I == bestB)
        D[removed] = maxvalue
        # If one or more of the remaining (still not assigned) rows of
        # D contain ONLY non-valid values (i.e. maxvalue), then the D
        # and I (so the tree) must be recomputed:
        if (D[not_assigned_A, :] == maxvalue).all(axis=1).any() and i < (sizeA - 1): # and we are not at the last iteration...
            print("iteration %s: re-computing the tree etc. (size=%s)." % (i, not_assigned_B.sum()))
            t1 = time()
            # we keep track of the original IDs of the non assigned
            # streamlines:
            map_id_A = map_id_A[not_assigned_A]
            map_id_B = map_id_B[not_assigned_B]
            D, I = compute_D_I(A[map_id_A], B[map_id_B], k=min(k, map_id_B.size), parallel=parallel)  # notice that the remaning streamlines in B may be less than k
            # We update nA and nB:
            nA = map_id_A.size
            nB = map_id_B.size
            # We update the not assigned boolean vectors:
            not_assigned_A = np.ones(nA, dtype=np.bool)
            not_assigned_B = np.ones(nB, dtype=np.bool)
            print("%s sec" % (time()-t1))

    print("Total SOLAP time: %s sec" % (time()-t0))
    print("Loss = %s" % loss)

    print("")
    print("Sanity check:")
    assert(np.unique(assignment).size == sizeA)
    print("OK")
    return assignment


def compute_solap_sort(A, B, k, maxvalue=100, parallel=True, verbose=False):
    """Compute a scalable (greedy) Sub-Optimal solution to the Linear
    Assignment Problem. Implementation using sorting and recursion.
    """
    assert(A.shape[0] <= B.shape[0])
    assert(test_unique_rows(A))
    assert(test_unique_rows(B))
    sizeA = A.shape[0]
    sizeB = B.shape[0]
    D, I = compute_D_I(A, B, k, parallel=parallel)

    print("Suboptimal assignment problem.")
    print("Argsort.")
    t0 = time()
    idxs = D.flatten().argsort()
    print("%s sec" % (time()-t0))

    print("Inverting I.")
    t0 = time()
    I_inverse = dict([(v, []) for v in np.unique(I.flatten())])
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            I_inverse[I[i, j]].append(i)

    print("%s sec" % (time()-t0))

    print("Main loop.")
    t0 = time()
    nA = sizeA
    nB = sizeB
    not_assigned_A = np.ones(nA, dtype=np.bool)
    not_assigned_B = np.ones(nB, dtype=np.bool)
    map_id_A = np.arange(nA, dtype=np.int)
    map_id_B = np.arange(nB, dtype=np.int)
    assignment = np.zeros(nA, dtype=np.int)
    counter = np.ones(nA, dtype=np.int) * k
    loss = 0.0
    sub_loss = 0.0
    for i, idx in enumerate(idxs):
        if (i % 1000) == 0 and verbose:
            print("not_assigned_A: %s" % not_assigned_A.sum())

        a, tmp = np.unravel_index(idx, I.shape)
        b = I[a, tmp]
        if not_assigned_A[a] and not_assigned_B[b]:
            assignment[a] = b
            loss += D.flat[idx]
            not_assigned_A[a] = False
            not_assigned_B[b] = False
            counter[I_inverse[b]] -= 1
            naa = not_assigned_A.sum()
            nab = not_assigned_B.sum()
            if (counter[I_inverse[b]] == 0).any() and naa > 0 and nab > 0:
                print("i: %s" % i)
                print("not_assigned_A: %s" % naa)
                print("not_assigned_B: %s" % nab)
                print("Recompute D, I.")
                map_id_A = map_id_A[not_assigned_A]
                map_id_B = map_id_B[not_assigned_B]
                sub_assignment, sub_loss = compute_solap_sort(A[map_id_A],
                                                              B[map_id_B],
                                                              k=min(k, nab))
                assignment[map_id_A] = map_id_B[sub_assignment]

                break

    loss += sub_loss
    print("%s sec" % (time()-t0))
    print("Loss = %s" % loss)
    return assignment, loss


def compute_solap_radius_new(AA, BB, radius=None, k=None, parallel=True):
    """Compute a scalable (greedy) Sub-Optimal solution to the Linear
    Assignment Problem considering distances till a certain radius.

    if radius is None, then it is estimated in order to provide k
    neighbors on average.
    """
    idxs, distances = compute_D_I(AA, BB, r=radius, k=k,
                                  query_radius=True)

    print("Computing correspondence with compute_solap_radius().")
    b_idxs = np.concatenate(idxs)
    distances = np.concatenate(distances)
    idxs_sorted = distances.argsort()
    a_idxs = np.concatenate([np.ones(len(idx), dtype=np.int32) * i for i, idx in enumerate(idxs)])

    a_expired = set([])
    b_expired = set([])
    correspondence = []
    correspondence_distances = []
    for counter, idx in enumerate(idxs_sorted):
        if (counter % 100000) == 0:
            print(counter)

        a = a_idxs[idx]
        b = b_idxs[idx]
        d = distances[idx]
        if a in a_expired or b in b_expired:
            continue
        else:
            correspondence.append([a, b])
            correspondence_distances.append(d)
            a_expired.add(a)
            b_expired.add(b)

    correspondence = np.array(correspondence, dtype=np.int)
    correspondence_distances = np.array(correspondence_distances)
    return correspondence


def compute_solap_radius(AA, BB, radius=None, k=None, parallel=True):
    """Compute a scalable (greedy) Sub-Optimal solution to the Linear
    Assignment Problem considering distances till a certain radius.

    if radius is None, then it is estimated in order to provide k
    neighbors on average.
    """
    idxs, distances = compute_D_I(AA, BB, r=radius, k=k,
                                  query_radius=True)

    print("Computing correspondence with compute_solap_radius().")
    b_idxs = np.concatenate(idxs)
    distances = np.concatenate(distances)
    a_idxs = np.concatenate([np.ones(len(idx), dtype=np.int) * i for i, idx in enumerate(idxs)])

    tmp = distances.argsort()
    b_idxs = b_idxs[tmp]
    distances = distances[tmp]
    a_idxs = a_idxs[tmp]

    correspondence = []
    counter = 0
    while a_idxs.size > 0:
        if (counter % 1000) == 0:
            print(counter)

        a = a_idxs[0]
        b = b_idxs[0]
        d = distances[0]
        correspondence.append([a, b, d])
        tmp = np.logical_and((a_idxs != a), (b_idxs != b))
        a_idxs = a_idxs[tmp]
        b_idxs = b_idxs[tmp]
        distances = distances[tmp]
        counter += 1

    correspondence = np.array(correspondence)
    distances = correspondence[:, 2]
    correspondence = correspondence[:, :2].astype(np.int)
    return correspondence


def compute_solap_radius_all(A, B, k):
    """Iterate over compute_solap_radius() in order to compute the
    sub-optimal linear assignment solution over all points in A.
    """
    sizeA = A.shape[0]
    sizeB = B.shape[0]
    assert(sizeA <= sizeB)
    correspondence = np.empty(shape=(0, 2))
    A_left = A
    B_left = B
    A_left_idx = np.arange(sizeA, dtype=np.int)
    B_left_idx = np.arange(sizeB, dtype=np.int)
    while A_left_idx.size > 0:
        if A_left_idx.size > k:
            tmp = compute_solap_radius(A_left, B_left, radius=None, k=k)
        else:
            tmp = compute_solap(A_left, B_left, k=A_left_idx.size)
            tmp = np.vstack([range(A_left_idx.size), tmp]).T

        tmp = np.vstack([A_left_idx[tmp[:, 0]], B_left_idx[tmp[:, 1]]]).T
        correspondence = np.vstack([correspondence, tmp])
        print("updating A_left and B_left.")
        A_left_idx = np.array(list(set(A_left_idx.tolist()).difference(set(tmp[:, 0].tolist()))), dtype=np.int)
        B_left_idx = np.array(list(set(B_left_idx.tolist()).difference(set(tmp[:, 1].tolist()))), dtype=np.int)
        A_left = A[A_left_idx]
        B_left = B[B_left_idx]

    return correspondence
