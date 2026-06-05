"""
MOSAIC Utility Functions
Graph construction, distance computation, affinity matrix,
clustering, and evaluation utilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    f1_score,
    silhouette_score,
)

cuda = True if torch.cuda.is_available() else False


# ===========================================================================
# Distance Functions
# ===========================================================================

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """Cosine distance between rows of x1 and x2."""
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def euclidean_distance_torch(x1, x2=None, eps=1e-8):
    """Euclidean distance between rows of x1 and x2."""
    x2 = x1 if x2 is None else x2
    return torch.cdist(x1, x2, p=2)


# ===========================================================================
# Sparse Conversion
# ===========================================================================

def to_sparse(x):
    """Convert a dense tensor to a sparse tensor."""
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


# ===========================================================================
# Graph Construction
# ===========================================================================

def graph_from_dist_tensor(dist, num_class, self_dist=True):
    """Keep top-k smallest distances per row, set others to 1."""
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"

    k = int(dist.shape[0] / num_class + 1)
    new_matrix = np.ones_like(dist)
    for idx, row in enumerate(dist):
        kth_smallest = np.partition(row, k - 1)[:k]
        new_row = np.where(np.isin(row, kth_smallest), row, 1)
        new_matrix[idx] = new_row

    return new_matrix


def gen_adj_mat_tensor(data, num_class, metric="euclidean"):
    """Generate normalized adjacency matrix from data."""
    if metric == "cosine":
        dist = cosine_distance_torch(data, data)
    elif metric == "euclidean":
        dist = euclidean_distance_torch(data, data)
        sigma = torch.median(dist)
        dist = torch.exp(-dist ** 2 / (2 * sigma ** 2 + 1e-8))
    else:
        raise NotImplementedError

    g = graph_from_dist_tensor(dist, num_class, self_dist=True)
    adj = g

    diag_idx = np.diag_indices(adj.shape[0])
    adj[diag_idx[0], diag_idx[1]] = 0

    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1
    row_sums_expanded = np.expand_dims(row_sums, axis=1)
    adj = adj / row_sums_expanded

    adj_T = adj.T
    adj = adj + adj_T
    adj = F.normalize(torch.from_numpy(adj), p=1)
    I = torch.eye(adj.shape[0])
    adj = adj + I
    adj = to_sparse(adj)

    return adj


# ===========================================================================
# Intra-view Contrastive Loss
# ===========================================================================

def knbrsloss(H, k, eps=1e-8):
    """K-nearest neighbor based intra-view contrastive loss."""
    H_norm = H.norm(dim=1, keepdim=True)
    dist_matrix = torch.mm(H, H.t()) / (H_norm * H_norm.t()).clamp(min=eps)

    simMaxNeb, indices = torch.topk(dist_matrix, k, largest=True)
    indices = indices[:, 1:]
    simMaxNeb = simMaxNeb[:, 1:]

    f = lambda x: torch.exp(x)
    refl_sim = f(dist_matrix)

    num = H.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    V = f(simMaxNeb)

    ret = -torch.log(
        V.sum(1) / (refl_sim.sum(1) - refl_sim.diag()))
    ret = ret.mean()

    return ret


# ===========================================================================
# Local Affinity Matrix
# ===========================================================================

def kscale(matrix, k=7, dists=None, minval=0.004):
    """Local scale based on k-th nearest neighbor distances."""
    r, c = matrix.shape
    scale = np.zeros((r, c))
    for i in range(1, k + 1):
        ix = (np.arange(len(matrix)), matrix.argsort(axis=0)[i])
        d = matrix[ix][np.newaxis].T
        dists = (d if dists is None else dists)
        scale1 = dists.dot(dists.T)
        scale = scale1 + scale
    scale = scale / k
    return np.clip(scale, minval, np.inf)


def affinity(matrix, k):
    """Compute local-scaled affinity matrix."""
    scale = kscale(matrix, k)
    msq = matrix * matrix
    scaled = -msq / (0.5 * scale + 0.5 * matrix)
    scaled[np.where(np.isnan(scaled))] = 0.0
    a = np.exp(scaled)
    a.flat[::matrix.shape[0] + 1] = 0.0
    return a


def testaff(matrix, k):
    """Wrapper to compute affinity matrix."""
    k = int(k)
    return affinity(matrix, k)


# ===========================================================================
# Clustering & Evaluation
# ===========================================================================

def cluster_label(x, k):
    """KMeans clustering on fused similarity matrix."""
    k = int(k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    return kmeans.predict(x).tolist()


def evaluate_clustering(x, k, label_path, label_col='label'):
    """
    Evaluate clustering performance against true labels.

    Parameters
    ----------
    x : ndarray
        Similarity matrix or embedding.
    k : int
        Number of clusters.
    label_path : str
        Path to CSV file with true labels.
    label_col : str
        Column name for true labels.

    Returns
    -------
    metrics : dict
        RI, ARI, NMI, Accuracy, F1-score.
    """
    k = int(k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    py_result = kmeans.predict(x)

    true_label = pd.read_csv(label_path)
    true_label_vals = np.array(true_label[label_col].tolist())

    # Remove NaN labels
    nan_idx = np.where(np.isnan(true_label_vals))[0]
    if len(nan_idx) > 0:
        true_label_vals = np.delete(true_label_vals, nan_idx)
        py_result = np.delete(py_result, nan_idx)

    true_label_list = true_label_vals.astype(int).tolist()
    test_label_list = py_result.astype(int).tolist()

    return {
        'RI':  rand_score(true_label_list, test_label_list),
        'ARI': adjusted_rand_score(true_label_list, test_label_list),
        'NMI': normalized_mutual_info_score(true_label_list, test_label_list),
        'Accuracy': accuracy_score(true_label_list, test_label_list),
        'F1': f1_score(true_label_list, test_label_list, average='micro'),
    }


# ===========================================================================
# Generic Clustering Wrapper
# ===========================================================================
# ===========================================================================

def cluster(method, data_matrix, num_class):
    """Apply a clustering method and return labels as tensor."""
    if method == 'kmeans':
        model = KMeans(n_clusters=num_class, random_state=0)
        labels = torch.tensor(model.fit_predict(data_matrix))
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=num_class, random_state=0)
        labels = torch.tensor(model.fit_predict(data_matrix))
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=num_class)
        labels = torch.tensor(model.fit_predict(data_matrix))
    elif method == 'gmm':
        model = GaussianMixture(n_components=num_class, random_state=0)
        model.fit(data_matrix)
        labels = torch.tensor(model.predict(data_matrix))
    else:
        raise ValueError(
            "Invalid clustering method. "
            "Available: 'kmeans', 'spectral', 'agglomerative', 'gmm'."
        )
    return labels
