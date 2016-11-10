import numpy as np
from time import time
from solap import compute_solap, compute_solap_sort
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment


def sklearn_rlap(A, B):
    X = pairwise_distances(A, B)
    tmp = linear_assignment(X)
    assignment = np.empty(tmp.shape[0])
    assignment[tmp[:, 0]] = tmp[:, 1]
    return assignment


if __name__ == '__main__':

    np.random.seed(0)

    sizeA = 10000
    sizeB = 10000
    d = 40
    k = 10
    plot = True

    print("Generating data.")
    t0 = time()
    A = np.random.uniform(size=(sizeA, d))
    B = np.random.uniform(size=(sizeB, d))
    # A = np.random.multivariate_normal(np.zeros(d),
    #                                   np.eye(d)/2.0, size=(sizeA))
    print("%s sec" % (time()-t0))

    assignment_solap = compute_solap(A, B, k)
    assignment_solap_sort, loss = compute_solap_sort(A, B, k)
    assert((assignment_solap == assignment_solap_sort).all())
    if sizeA <= 1000 and sizeB <= 1000:
        print("")
        print("Optimal solution to the Rectangular Linear Assignment Problem (RLAP):")
        assignment_rlap = sklearn_rlap(A, B)
        print("Match between SOLAP and RLAP: %s" %
              (assignment_solap == assignment_rlap).mean())

    if plot and d == 2:
        assignment = assignment_solap
        plt.interactive(True)
        delta = 2.0
        plt.figure()
        plt.plot(A[:, 0], A[:, 1], 'ro')
        plt.plot(B[:, 0] + delta, B[:, 1] + delta, 'bo')
        for i in range(sizeA):
            plt.plot([A[i, 0], B[assignment[i], 0] + delta],
                     [A[i, 1], B[assignment[i], 1] + delta], 'k-')

        # plt.show()
