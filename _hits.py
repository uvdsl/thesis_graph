import numpy as np
from numba import njit, prange

@njit(nogil=True)
def hub_matrix(M):
    return M@M.T

@njit(nogil=True)
def authority_matrix(M):
    return M.T@M

@njit(nogil=True)
def hits_hubs(M, max_iter=500, tol=1.0e-6, normalized=True):
    (n, m) = M.shape  # should be square
    H = hub_matrix(M)
    x = np.ones((n, 1)) / n  # initial guess
    # power iteration on authority matrix
    i = 0
    while True:
        xlast = x
        x = H@x
        x = x / x.max()
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < tol:
            break
        if i > max_iter:
            raise RuntimeError('Power Method exceeded max iterations.')
        i += 1
    h = np.asarray(x).flatten()
    if normalized:
        h = h / h.sum()
    return h

@njit(nogil=True)
def hits_authorities(M, max_iter=500, tol=1.0e-6, normalized=True):
    (n, m) = M.shape  # should be square
    A = authority_matrix(M)
    x = np.ones((n, 1)) / n  # initial guess
    # power iteration on authority matrix
    i = 0
    while True:
        xlast = x
        x = A@x
        x = x / x.max()
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < tol:
            break
        if i > max_iter:
            raise RuntimeError('Power Method exceeded max iterations.')
        i += 1
    a = np.asarray(x).flatten()
    if normalized:
        a = a / a.sum()
    return a


@njit(nogil=True)
def hits(M, max_iter=500, tol=1.0e-6, normalized=True):
    (n, m) = M.shape  # should be square
    H =  hub_matrix(M)  
    x = np.ones((n, 1)) / n  # initial guess
    # power iteration on authority matrix
    i = 0
    while True:
        xlast = x
        x = H@x
        x = x / x.max()
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < tol:
            break
        if i > max_iter:
            raise RuntimeError('Power Method exceeded max iterations.')
        i += 1
    h = np.asarray(x).flatten()
    # h=M*a
    a = np.asarray(M.T@h).flatten()
    if normalized:
        h = h / h.sum()
        a = a / a.sum()
    return h, a
    