import numpy as np
from time import time
from sklearn.neighbors import KDTree, BallTree
from solap import compute_solap_radius_all, compute_solap

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
