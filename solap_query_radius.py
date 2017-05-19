import numpy as np
from time import time
from sklearn.neighbors import KDTree, BallTree
from solap import tree_parallel_query, compute_solap


def compute_solap_radius(AA, BB, radius, k=None):
    """Compute a scalable (greedy) Sub-Optimal solution to the Linear
    Assignment Problem considering distances till a certain radius.
    """
    print("Building Tree and estimating nearest neighbors till radius %s" % radius)
    tree = KDTree(BB)
    # idxs, distances = tree.query_radius(AA, r=radius, return_distance=True)
    idxs, distances = tree_parallel_query(tree, AA, r=radius, k=k, query_radius=True)

    print("Computing correspondence.")
    b_idxs = np.concatenate(idxs)
    distances = np.concatenate(distances)
    a_idxs = np.concatenate([np.ones(len(idx)) * i for i, idx in enumerate(idxs)]).astype(np.int)

    tmp = distances.argsort()
    b_idxs = b_idxs[tmp]
    distances = distances[tmp]
    a_idxs = a_idxs[tmp]

    correspondence = []
    counter = 0
    while b_idxs.size > 0:
        if (counter % 1000) == 0:
            print(counter)
    
        a = a_idxs[0]
        b = b_idxs[0]
        r = distances[0]
        correspondence.append([a_idxs[0], b_idxs[0], distances[0]])
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
    sizeA = A.shape[0]
    sizeB = B.shape[0]
    a_idxs_left = np.arange(sizeA, dtype=np.int)
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
        # updating A_left and B_left:
        A_left_idx = np.array(list(set(A_left_idx.tolist()).difference(set(tmp[:, 0].tolist()))), dtype=np.int)
        B_left_idx = np.array(list(set(B_left_idx.tolist()).difference(set(tmp[:, 1].tolist()))), dtype=np.int)
        A_left = A[A_left_idx]
        B_left = B[B_left_idx]

    return correspondence


if __name__ == '__main__':

    np.random.seed(0)

    sizeA = 10000
    sizeB = 10000
    d = 40
    # radius = 2.0  # it can be estimated with tree.query(k=...)
    k = 10

    print("Generating data.")
    t0 = time()
    A = np.random.uniform(size=(sizeA, d))
    B = np.random.uniform(size=(sizeB, d))
    # A = np.random.multivariate_normal(np.zeros(d),
    #                                   np.eye(d)/2.0, size=(sizeA))
    print("%s sec" % (time()-t0))


    t0 = time()
    correspondence = compute_solap_radius_all(A, B, k)
    print("%s sec" % (time()-t0))
    assert(np.unique(correspondence[:, 0]).size == sizeA)
    assert(np.unique(correspondence[:, 1]).size == sizeB)
    print("Sanity check: OK.")

    assignment = compute_solap(A, B, k=10)
    idx = correspondence[:, 0].argsort()
    assert((assignment == correspondence[:, 1][idx]).all())
    print("Further sanity check: OK.")
