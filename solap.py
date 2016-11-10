"""Sub-Optimal solution to the Linear Assignment Problem (SOLAP) for
large datasets.
"""

import numpy as np
# from scipy.spatial import KDTree
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import KDTree
from time import time


def test_unique_rows(a):
    """Test wether an array 'a' has unique rows (no repetitions) or not.
    """
    # See: http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    unique_rows = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1]).shape[0]
    return (a.shape[0] == unique_rows)


def compute_D_I(A, B, k):
    """Compute the sparse distance matrix (D) and the IDs of the neighbors
    (I) between the set of vectors A and the set of vectors B,
    considering just k neighbors. This computation is based on the use
    of the KDTree.
    """
    print("Computing the kdtree (size=%s)." % B.shape[0])
    t0 = time()
    kdt = KDTree(B)
    print("%s sec" % (time()-t0))
    print("Computing %s NN queries, each for %s neighbors." % (A.shape[0], k))
    t0 = time()
    D, I = kdt.query(A, k=k)
    print("%s sec" % (time()-t0))
    return D, I


def compute_solap(A, B, k=10, maxvalue=100):
    """Compute a scalable (greedy) Sub-Optimal solution to the Linear
    Assignment Problem.
    """
    assert(A.shape[0] <= B.shape[0])
    assert(test_unique_rows(A))
    assert(test_unique_rows(B))
    sizeA = A.shape[0]
    sizeB = B.shape[0]
    D, I = compute_D_I(A, B, k)

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
        # and I (so the kdtree) must be recomputed:
        if (D[not_assigned_A, :] == maxvalue).all(axis=1).any() and i < (sizeA - 1): # and we are not at the last iteration...
            print("iteration %s: re-computing the kdtree etc. (size=%s)." % (i, not_assigned_B.sum()))
            t1 = time()
            # we keep track of the original IDs of the non assigned
            # streamlines:
            map_id_A = map_id_A[not_assigned_A]
            map_id_B = map_id_B[not_assigned_B]
            D, I = compute_D_I(A[map_id_A], B[map_id_B], k=min(k, map_id_B.size))  # notice that the remaning streamlines in B may be less than k
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


def compute_solap_sort(A, B, k, maxvalue=100):
    """Compute a scalable (greedy) Sub-Optimal solution to the Linear
    Assignment Problem. Implementation using sorting and recursion.
    """
    assert(A.shape[0] <= B.shape[0])
    assert(test_unique_rows(A))
    assert(test_unique_rows(B))
    sizeA = A.shape[0]
    sizeB = B.shape[0]
    D, I = compute_D_I(A, B, k)

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
        if (i % 1000) == 0:
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
