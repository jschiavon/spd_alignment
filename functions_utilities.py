import numpy as np

# Functions
_isqrt = lambda x: 1. / np.sqrt(x)
_funs = {'sqrt': np.sqrt,
        'isqrt': _isqrt,
        'log': np.log,
        'exp': np.exp}

def _transform_mat(X, func='sqrt'):
    """
    Applies a transformation to a SPD matrix by means of the eigenvalues.

    This function compute the eigenvalue decomposition of a SPD matrix and
    returns the matrix obtained by applying the transformation func to the 
    eigenvalues before reconstructing the matrix, i.e. returns
    `V * func(Lambda) * V'` where `V * Lambda * V'` is the eigenvalue decomposition
    of `X`
    """
    u, v = np.linalg.eigh(X)
    return np.einsum('...ij,...j,...kj', v, _funs[func](u), v)


def norm_frob_squared(X):
    """
    Computes the squared Frobenius norm of a matrix.
    """
    return np.einsum('...ji,...ji', X, X)


def dist_frob_squared(X, Y):
    """
    Computes the squared distance, induced by frobenius norm, between two matrices.
    """
    return norm_frob_squared(X - Y)


def dist_riem_squared(X, Y):
    """
    Computes the squared distance, induced by riemmanian norm, between two SPD matrices.
    """
    x = _transform_mat(X, 'isqrt')
    mid = np.einsum('...ij,...jk,...kl', x, Y, x)
    return norm_frob_squared(_transform_mat(mid, 'log'))


def norm_riem_squared(X):
    """
    Computes the squared Riemannian norm of a SPD matrix.
    """
    x = _transform_mat(X, 'log')
    return norm_frob_squared(x)


def rotate(X, Omega):
    """
    Rotates a SPD matrix `X` with an orthogonal matrix `Omega`.
    """
    return np.einsum('...ij,...jk,...lk', Omega, X, Omega)
