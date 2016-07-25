from __future__ import division
import numpy as np
import numpy.linalg as la


def mvee(points, tol=0.001):
    N, d = points.shape

    Q = np.zeros([N, d+1])
    Q[:, 0:d] = points[0:N, 0:d]
    Q[:, d] = np.ones([1, N])

    Q = np.transpose(Q)
    points = np.transpose(points)
    count = 1
    err = 1
    u = (1/N) * np.ones(shape=(N,))

    while err > tol:

        X = np.dot(np.dot(Q, np.diag(u)), np.transpose(Q))
        M = np.diag(np.dot(np.dot(np.transpose(Q), la.inv(X)), ))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1)/((d+1)*(M[jdx] - 1))
        new_u = (1 - step_size)*u
        new_u[jdx] = new_u[jdx] + step_size
        count = count + 1
        err = la.norm(new_u - u)
        u = new_u

    U = np.diag(u)
    c = np.dot(points, u)
    A = (1/d) * la.inv(np.dot(np.dot(points, U), np.transpose(points)) - np.dot(c, np.transpose(c)))
    return A, np.transpose(c)
