import scipy.io as spio
import pandas as pd
from admm_lasso import lasso_admm
import numpy as np
import scipy.linalg as scilin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import lars_path


#### example 1
n_features = 20
a = np.ones((1, n_features))[0]
b = np.ones((1, n_features-1))[0]
Q = 5*np.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1)
Q = Q + np.random.normal(0,0.005,(n_features,n_features))
S = np.linalg.inv(Q)

#### example 2
# # load matlab data file
# mat = spio.loadmat('ggm_data.mat', squeeze_me=True)
# Xdata = mat['X']
# n_features = Xdata.shape[1]
# for i in range(n_features):
#     Xdata[:,i] = Xdata[:,i] - np.mean(Xdata[:,i])
# # transform into dataframe
# df = pd.DataFrame(Xdata)
# # get data sample empirical covariance matrix
# S = df.cov()
# S = np.array(S)



# S = np.array(S)
# set parameters
rho = 5

# initialize W
W = np.matrix(S) + rho * np.eye(n_features)

# begin iteration
MAXITER = 20

# get s12
s12all = []
for i in range(n_features):
    temp = [k for k in range(n_features) if k <> i]
    s12all.append([np.matrix(S)[i, k] for k in temp])
s12all = np.asarray(s12all)

# for currIter in range(MAXITER):
#     for i in range(n):
#         # pivoting W
#
#         W[i, :], W[n - 1, :] = W[n - 1, :].copy(), W[i, :].copy()
#         W[:, i], W[:, n - 1] = W[:, n - 1].copy(), W[:, i].copy()
#
#         # take sub-matrix W11
#         W11 = W[0:n - 1, 0:n - 1]
#         A = scilin.sqrtm(W11)
#         b = scilin.solve(A, s12all[i])[..., np.newaxis]
#
#         # implement lasso individually
#         # l1p = 0.1*np.linalg.norm(A.T.dot(b), ord=float('Inf'))
#         beta, h_beta = lasso_admm(A, b, l1penalty=0.01)
#
#         # generate w12
#         w12 = W11.dot(beta)
#         W[n - 1, 0:n - 1] = w12.T
#         W[0:n - 1, n - 1] = w12
#
#         # re-pivoting the W
#         W[i, :], W[n - 1, :] = W[n - 1, :].copy(), W[i, :].copy()
#         W[:, i], W[:, n - 1] = W[:, n - 1].copy(), W[:, i].copy()


def test_convergence( previous_W, new_W, S, t):
    d = S.shape[0]
    x = np.abs( previous_W - new_W ).mean()
    print x - t*( np.abs(S).sum() + np.abs( S.diagonal() ).sum() )/(d*d-d)
    if np.abs( previous_W - new_W ).mean() < t*( np.abs(S).sum() + np.abs( S.diagonal() ).sum() )/(d*d-d):
        return True
    else:
        return False

def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion
    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum()
                    - np.abs(np.diag(precision_)).sum())
    return gap

mle_estimate_ = S
covariance_ = mle_estimate_.copy()
precision_ = np.linalg.pinv(mle_estimate_)
indices = np.arange(n_features)
for i in xrange(MAXITER):
    for n in range(n_features):
        sub_estimate = covariance_[indices != n].T[indices != n]
        row = mle_estimate_[n, indices != n]
        # solve the lasso problem
        # _, _, coefs_ = lars_path(sub_estimate, row, Xy=row, Gram=sub_estimate,
        #                          alpha_min=0.06, copy_Gram=True,
        #                          method="lars")
        # coefs_ = coefs_[:, -1]  # just the last please.
        A = scilin.sqrtm(np.matrix(sub_estimate))
        A = np.array(A)
        b = scilin.solve(A,row)
        coefs_ ,_ = lasso_admm(A, b,0.007)
        # update the precision matrix.
        precision_[n, n] = 1. / (covariance_[n, n] - np.dot(covariance_[indices != n, n][np.newaxis,...], coefs_))
        precision_[indices != n, n] = - precision_[n, n] * coefs_
        precision_[n, indices != n] = - precision_[n, n] * coefs_
        temp_coefs = np.dot(sub_estimate, coefs_)
        covariance_[n, indices != n] = temp_coefs
        covariance_[indices != n, n] = temp_coefs

    # if test_convergence( old_estimate_, new_estimate_, mle_estimate_, convg_threshold):
    if np.abs(_dual_gap(mle_estimate_, precision_, 0.1)) < 1e-5:
        break
else:
    # this triggers if not break command occurs
    print "The algorithm did not coverge. Try increasing the max number of iterations."

# print pd.DataFrame(W)
print pd.DataFrame(precision_)
plt.imshow(precision_, cmap='hot', interpolation='nearest')
plt.show()
plt.savefig("filename.png")

