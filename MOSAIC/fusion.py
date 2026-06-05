"""
Network Fusion Module
Iterative similarity network fusion with adaptive view weighting
and spectral embedding-guided local affinity refinement.

Core algorithm converted from R (NetworkFusion.R) and C (projsplx_R.c) to Python.
"""

import numpy as np
from scipy.sparse import issparse


# ===========================================================================
# Simplex Projection (ported from projsplx_R.c)
# ===========================================================================

def projsplx(y):
    """
    Project each column of y onto the probability simplex.
    Implements the algorithm from Wang & Carreira-Perpiñán (2013).

    Parameters
    ----------
    y : ndarray of shape (m, n)
        Input matrix. Each column is projected independently.

    Returns
    -------
    x : ndarray of shape (m, n)
        Projected matrix where each column sums to 1 and is non-negative.
    """
    m, n = y.shape
    x = np.zeros_like(y, dtype=np.float64)

    for k in range(n):
        s = y[:, k].copy().astype(np.float64)
        means = np.sum(s)
        s = s - (means - 1.0) / m
        mins = np.min(s)

        if mins < 0:
            f_val = 1.0
            lambda_m = 0.0
            ft = 1
            vs = s.copy()
            while abs(f_val) > 1e-10:
                npos = 0
                f_val = 0.0
                vs = s - lambda_m
                for j in range(m):
                    if vs[j] > 0:
                        npos += 1
                        f_val += vs[j]
                if npos > 0:
                    lambda_m += (f_val - 1.0) / npos
                if ft > 100:
                    x[:, k] = np.maximum(vs, 0)
                    break
                ft += 1
            x[:, k] = np.maximum(vs, 0)
        else:
            x[:, k] = s

    return x


# ===========================================================================
# Kernel Normalization
# ===========================================================================

def dn(w, norm_type='ave'):
    """
    Normalize a symmetric kernel matrix.

    Parameters
    ----------
    w : ndarray of shape (n, n)
        Symmetric kernel / similarity matrix.
    norm_type : str
        'ave' → D^{-1} W   (random walk normalization)
        'gph' → D^{-1/2} W D^{-1/2}  (symmetric normalization)

    Returns
    -------
    wn : ndarray of shape (n, n)
        Normalized matrix.
    """
    D = np.sum(w, axis=0)

    if norm_type == 'ave':
        D = 1.0 / D
        D_mat = np.diag(D)
        wn = D_mat @ w
    elif norm_type == 'gph':
        D = 1.0 / np.sqrt(D)
        D_mat = np.diag(D)
        wn = D_mat @ w @ D_mat
    else:
        raise ValueError("Invalid normalization type. Use 'ave' or 'gph'.")

    return wn


# ===========================================================================
# Eigen-decomposition Helper
# ===========================================================================

def eig1(A, c=None, is_max=True, is_sym=True):
    """
    Compute top or bottom c eigenvalues and eigenvectors.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Square matrix (typically a Laplacian).
    c : int or None
        Number of eigenvectors to return. Default: n.
    is_max : bool
        If True, return eigenvectors corresponding to largest eigenvalues.
        If False, return eigenvectors corresponding to smallest eigenvalues.
    is_sym : bool
        If True, use eigh (symmetric solver).

    Returns
    -------
    eigval : ndarray of shape (c,)
        Selected eigenvalues.
    eigvec : ndarray of shape (n, c)
        Selected eigenvectors (real part).
    eigval_full : ndarray of shape (n,)
        All eigenvalues in sorted order.
    """
    n = A.shape[0]
    if c is None or c > n:
        c = n

    if is_sym:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(A)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    if is_max:
        idx = np.argsort(eigenvalues)[::-1]
    else:
        idx = np.argsort(eigenvalues)

    idx1 = idx[:c]
    eigval = eigenvalues[idx1]
    eigvec = eigenvectors[:, idx1]
    eigval_full = eigenvalues[idx]

    return eigval, eigvec, eigval_full


# ===========================================================================
# L2 Distance
# ===========================================================================

def L2_distance_1(a, b):
    """
    Compute pairwise squared L2 distance between columns of a and b.

    Parameters
    ----------
    a : ndarray of shape (d, n1)
    b : ndarray of shape (d, n2)

    Returns
    -------
    d : ndarray of shape (n1, n2)
        Squared L2 distances, with diagonal zeroed out if n1 == n2.
    """
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros(a.shape[1])])
        b = np.vstack([b, np.zeros(b.shape[1])])

    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = a.T @ b

    d1 = np.tile(aa.reshape(-1, 1), (1, len(bb)))
    d2 = np.tile(bb.reshape(1, -1), (len(aa), 1))
    d = d1 + d2 - 2 * ab
    d = np.real(d)
    d = np.maximum(d, 0)

    if d.shape[0] == d.shape[1]:
        np.fill_diagonal(d, 0)

    return d


# ===========================================================================
# Perplexity-based Optimization (umkl)
# ===========================================================================

def Hbeta(D, beta):
    """
    Compute Shannon entropy and softmax distribution for given bandwidth.

    Parameters
    ----------
    D : ndarray
        Distance vector.
    beta : float
        Bandwidth parameter (1 / variance).

    Returns
    -------
    H : float
        Shannon entropy.
    P : ndarray
        Softmax probabilities.
    """
    eps = np.finfo(np.float64).eps
    D = (D - np.min(D)) / (np.max(D) - np.min(D) + eps)
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def umkl(D, beta=None):
    """
    Binary search for beta that achieves a target perplexity (u=20).

    Parameters
    ----------
    D : ndarray
        Distance vector.
    beta : float or None
        Initial bandwidth guess.

    Returns
    -------
    thisP : ndarray
        Optimized probability distribution.
    """
    if beta is None:
        beta = 1.0 / len(D)

    tol = 1e-4
    u = 20.0
    logU = np.log(u)

    H, thisP = Hbeta(D, beta)
    Hdiff = H - logU
    tries = 0
    betamin = -np.inf
    betamax = np.inf

    while abs(Hdiff) > tol and tries < 30:
        if Hdiff > 0:
            betamin = beta
            if np.isinf(betamax):
                beta = beta * 2
            else:
                beta = (beta + betamax) / 2
        else:
            betamax = beta
            if np.isinf(betamin):
                beta = beta * 2
            else:
                beta = (beta + betamin) / 2

        H, thisP = Hbeta(D, beta)
        Hdiff = H - logU
        tries += 1

    return thisP


# ===========================================================================
# Main Network Fusion Algorithm
# ===========================================================================

def network_fusion(X, c, no_dim=None, k=10, niter=30, beta=0.8, verbose=True):
    """
    Iterative similarity network fusion.

    Parameters
    ----------
    X : list of ndarray
        List of view-specific affinity/similarity matrices, each (n, n).
    c : int
        Number of clusters (and default embedding dimension).
    no_dim : int or None
        Embedding dimension. Default: c.
    k : int
        Number of nearest neighbors for local affinity.
    niter : int
        Maximum number of iterations.
    beta : float
        Update rate for affinity matrix blending (0 < beta < 1).
    verbose : bool
        Print progress messages.

    Returns
    -------
    S : ndarray of shape (n, n)
        Fused similarity matrix.
    F_eig1 : ndarray of shape (n, c)
        Spectral embedding (eigenvectors of the final Laplacian).
    """
    if no_dim is None:
        no_dim = c

    num = X[0].shape[0]
    n_views = len(X)

    # ---- Initialization ----
    alphaK = np.ones(n_views) / n_views

    # Weighted sum of kernels
    distX = np.zeros((num, num))
    for i in range(n_views):
        distX += X[i]
    distX /= n_views

    # Sort each row
    idx = np.zeros((num, num), dtype=np.int64)
    for i in range(num):
        sorted_idx = np.argsort(distX[i, :])
        idx[i, :] = sorted_idx

    # Bandwidth parameter r
    k_neighbor = min(k + 1, num - 1)
    di = np.take_along_axis(distX, idx[:, 1:(k_neighbor + 1)], axis=1)
    rr = 0.5 * (k_neighbor * di[:, -1] - np.sum(di[:, :k_neighbor], axis=1))
    r = np.mean(rr)
    if r <= 0:
        r = max(np.mean(rr), 0)

    lambda_val = max(np.mean(rr), 0)

    # Initial S from distX
    S0 = np.max(distX) - distX
    S0 = dn(S0, 'ave')
    S = S0.copy()

    D0 = np.diag(np.sum(S, axis=0))
    L0 = D0 - S
    _, F_eig1, _ = eig1(L0, c, is_max=False)

    converge = []
    S_old = S.copy()

    # ---- Iterative Fusion ----
    for it in range(niter):
        if verbose:
            print(f"  Fusion iteration {it + 1}/{niter}")

        # L2 distance between spectral embeddings
        distf = L2_distance_1(F_eig1.T, F_eig1.T)

        # Build local affinity matrix A
        A = np.zeros((num, num))
        b = idx[:, 1:]
        a = np.tile(np.arange(num).reshape(-1, 1), (1, b.shape[1]))
        inda = (a.ravel(), b.ravel())

        ad = (distX[inda] + lambda_val * distf[inda]) / (2.0 * r)
        ad = ad.reshape(num, b.shape[1])

        # Simplex projection
        ad = projsplx(ad.T).T

        A[inda] = ad.ravel()
        A[np.isnan(A)] = 0
        A = (A + A.T) / 2

        # Update S
        S = (1 - beta) * S + beta * A

        # Recompute Laplacian and eigenvectors
        D = np.diag(np.sum(S, axis=0))
        L = D - S
        ev_eig1, F_eig1, ev_full = eig1(L, c, is_max=False)

        # Update view weights alphaK
        DD = np.zeros(n_views)
        eps = np.finfo(np.float64).eps
        for i in range(n_views):
            temp1 = (eps + X[i]) * (S + eps)
            temp2 = 0.5 * (eps + X[i]) * (X[i] + eps)
            temp = temp1 - temp2
            DD[i] = np.mean(np.sum(temp, axis=0))

        alphaK0 = umkl(DD)
        alphaK0 = alphaK0 / np.sum(alphaK0)
        alphaK = (1 - beta) * alphaK + beta * alphaK0
        alphaK = alphaK / np.sum(alphaK)

        # Convergence check
        c_idx = min(c, len(ev_eig1))
        fn1 = np.sum(ev_eig1[:c_idx])
        fn2 = np.sum(ev_eig1[:(c_idx + 1)]) if c_idx < len(ev_eig1) else fn1
        converge.append(fn2 - fn1)

        # Adaptive lambda / r
        if it < 10 and len(ev_eig1) > 0 and ev_eig1[-1] > 1e-6:
            lambda_val *= 1.5
            r /= 1.01

        # Early stopping
        if it >= 10 and converge[it] > converge[it - 1]:
            S = S_old
            if verbose:
                print(f"  Converged at iteration {it + 1}")
            break

        S_old = S.copy()

        # Update distX with new weights
        distX = X[0] * alphaK[0]
        for i in range(1, n_views):
            distX += X[i] * alphaK[i]

        # Re-sort
        for i in range(num):
            sorted_idx = np.argsort(distX[i, :])
            idx[i, :] = sorted_idx

    return S, F_eig1
