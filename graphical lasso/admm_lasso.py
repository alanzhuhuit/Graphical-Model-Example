# some comment about this program
# The program I wrote is almost the same as the on in alasso.py
# both program tried to implement lasso algorithm by ADMM method.
# The only difference that I can notice is that alasso used sparse matrix
# representation of the coefficients. Since there is a lot of matrix multiplication
# during this method, it might cause instability.


# After careful reread the file I noticed that what should be returned
# from a ADMM scheme is the z not x.
import pdb, time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm, cholesky, solve
from sklearn.linear_model import lars_path
import alasso as ladmm

# shrinkage use to update z
# def lasso_shrinkage(x, kappa):
#     return np.maximum(0., x - kappa) - np.maximum(0., -x - kappa)


#

def lasso_shrinkage(x, kappa):
    return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)
    # y=[]
    # for a in x:
    #     if a > kappa:
    #         y.append(a-kappa)
    #     elif a < -kappa:
    #         y.append(a+kappa)
    #     else:
    #         y.append(0)
    # return np.array(y)


# objective function
def objective(A, b, l1_penalty, x, z):
    return .5 * np.square(A.dot(x) - b).sum() + l1_penalty * norm(z, 1)


# decomposition
def factor(A, rho):
    m, n = A.shape
    if m >= n:
        L = cholesky(A.T.dot(A) + rho * np.eye(n))
    else:
        L = cholesky(np.eye(m) + 1. / rho * (A.dot(A.T)))
    U = L.T
    return L, U


def lasso_admm(A, b, l1penalty=0.01, rho=1, relax_alpha=1., MAX_ITER=1000, ABSTOL=1e-3, RELTOL=1e-2):
    # Data preprocessing
    m, n = A.shape
    # save a matrix-vector multiply
    # b = b[...,np.newaxis]
    Atb = A.T.dot(b)

    # ADMM solver
    # x = np.zeros((n, 1))
    # z = np.zeros((n, 1))
    # u = np.zeros((n, 1))
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    # cache the (Cholesky) factorization
    L, U = factor(A, rho)

    # store the history of the optimization
    # use data type dictionary to store
    h = {}
    h['objval'] = np.zeros(MAX_ITER)
    h['r_norm'] = np.zeros(MAX_ITER)
    h['s_norm'] = np.zeros(MAX_ITER)
    h['eps_pri'] = np.zeros(MAX_ITER)
    h['eps_dual'] = np.zeros(MAX_ITER)

    for k in xrange(MAX_ITER):
        print(k)
        # x-update
        q = (Atb + rho * (z - u))  # (temporary value)
        if m >= n:
            x = solve(U, solve(L, q))
        else:
            ULXq = solve(U, solve(L, A.dot(q)))
            x = (q * 1. / rho) - ((A.T.dot(ULXq)) * 1. / (rho ** 2))

        # z-update without relaxation
        zold = np.copy(z)
        x_hat = relax_alpha * x + (1. - relax_alpha) * zold
        z = lasso_shrinkage(x_hat + u, l1penalty / rho)

        # u-updates
        u += (x - z)

        # diagnostics, reporting, termination checks
        h['objval'][k] = objective(A, b, l1penalty, x, z)
        h['r_norm'][k] = norm(x - z)
        h['s_norm'][k] = norm(-rho * (z - zold))
        h['eps_pri'][k] = np.sqrt(n) * ABSTOL + RELTOL * np.maximum(norm(x), norm(-z))
        h['eps_dual'][k] = np.sqrt(n) * ABSTOL + RELTOL * norm(rho * u)

        # break condition
        if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
            break

    return z, h


if __name__ == "__main__":
    m = 40
    n = 30
    np.random.seed(123)
    x0 = sparse.random(n, 1, density=0.1)
    x0 = np.array(x0.todense()).ravel()
    A = np.random.normal(0, 2, (m, n))
    # b = A.dot(x0)
    b = A.dot(x0) + np.random.normal(0, 0.2, m)
    # l1p = 0.2 * norm(A.T.dot(b), ord=float('Inf'))
    l1p = 100
    # A = np.asmatrix(A)
    # b = np.asmatrix(b)
    x, h = lasso_admm(A, b, l1penalty=l1p)
    # print x.T
    xfil = np.zeros(n)
    for i in range(x.size):
        if abs(x[i]) > 1e-2:
            xfil[i] = x[i]
        else:
            xfil[i] = 0

    # print xfil.T
    # print x0.todense().T
    # outpu = (np.asarray(x0.todense()))



    # test ladmm
    A = np.matrix(A)
    b = np.matrix(b).T
    x, h = ladmm.lasso_admm(A, b, l1p, 1., 1.)
    for myval, origval,hiscode in zip(xfil, x0, x):
        print myval, origval, hiscode
    # _, _, coefs_ = lars_path(A, b, method="lasso", alpha_min=l1p)
    # print "Now is new"
    # for myval, origval in zip(coefs_[:, -1], x0):
    #     print myval, origval

    # print coefs_[:,-1]
    # print abs(x-x0).T
