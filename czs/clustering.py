"""K-Means zoning of crop stress."""
import numpy as np
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_features(indices, bands, index_names, band_names=()):
    layers = [indices[n] for n in index_names] + [bands[n] for n in band_names]
    if not layers:
        raise ValueError("Select at least one feature.")
    names = list(index_names) + list(band_names)
    h, w = layers[0].shape
    X = np.stack(layers, -1).reshape(-1, len(layers)).astype(np.float64)
    return X, names, (h, w)


def kmeans_zones(X, shape, k=3, ref_col=0, seed=0, smooth=0):
    """Cluster pixels, then order clusters by mean of feature column `ref_col`
    (a greenness index) so the lowest-greenness cluster = most stressed.

    Returns (stress_class_map, cluster_id_map, class_of_cluster, model).
    stress_class: 0 healthy, 1 stressed, 2 diseased (k clusters binned into 3).
    """
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    lab = km.labels_
    means = np.array([X[lab == c, ref_col].mean() if (lab == c).any() else np.inf
                      for c in range(k)])
    rank = np.empty(k, int)
    rank[np.argsort(means)] = np.arange(k)  # rank 0 = lowest greenness (worst)
    stress_level = (k - 1) - rank           # 0 = healthiest
    if k == 2:
        cls_of_cluster = np.where(stress_level == 0, 0, 2)
    elif k == 3:
        cls_of_cluster = stress_level
    else:
        cls_of_cluster = np.minimum((stress_level * 3) // k, 2)
    cls = cls_of_cluster[lab].reshape(shape).astype(np.uint8)
    if smooth > 1:
        cls = median_filter(cls, size=int(smooth)).astype(np.uint8)
    return cls, lab.reshape(shape), cls_of_cluster, km


def k_selection(X, k_range=range(2, 9), seed=0, sample=4000):
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), min(sample, len(Xs)), replace=False)
    out = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Xs)
        out.append((k, km.inertia_, silhouette_score(Xs[idx], km.labels_[idx])))
    return out
