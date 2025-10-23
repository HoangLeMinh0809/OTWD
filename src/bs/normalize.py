import numpy as np
# ============================== Normalization ==============================
def _zscore_pooled(X, Y, eps=1e-8):
    """Z-score per feature, pooled between X and Y (to scale consistently).

    Returns
    -------
    Xz, Yz, stats
        Normalized X and Y and a small stats dict.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    mu = np.mean(np.vstack([X, Y]), axis=0)
    sd = np.std(np.vstack([X, Y]), axis=0, ddof=0)
    sd = np.maximum(sd, eps)
    Xz = (X - mu) / sd
    Yz = (Y - mu) / sd
    stats = {"type": "zscore", "mu": mu, "sd": sd}
    return Xz, Yz, stats

def _robust_zscore_pooled(X, Y, eps=1e-8):
    """Robust z-score using median/MAD, pooled between X and Y.

    Notes
    -----
    MAD is multiplied by 1.4826 to approximate the standard deviation for
    Gaussian data.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Z = np.vstack([X, Y])
    med = np.median(Z, axis=0)
    mad = np.median(np.abs(Z - med), axis=0)
    sigma = 1.4826 * np.maximum(mad, eps)
    Xz = (X - med) / sigma
    Yz = (Y - med) / sigma
    stats = {"type": "robust", "median": med, "sigma": sigma}
    return Xz, Yz, stats

def _pca_whiten_pooled(X, Y, eps=1e-6, center="zscore", k_keep=None, energy=None):
    """PCA-whitening pooled between X and Y, with optional dimensionality reduction.

    Parameters
    ----------
    k_keep : int or None
        Keep exactly k principal components (if None, use `energy`).
    energy : float or None
        Minimum cumulative variance ratio to keep (e.g. 0.95).

    Returns
    -------
    Xw, Yw, stats
        Whitened projections of X and Y and stats dict.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if center == "zscore":
        Xc, Yc, _ = _zscore_pooled(X, Y, eps=eps)
        Zc = np.vstack([Xc, Yc])  # mean ~ 0
    else:
        Z = np.vstack([X, Y])
        mu = np.mean(Z, axis=0)
        Xc = X - mu; Yc = Y - mu
        Zc = np.vstack([Xc, Yc])

    # SVD on Zc (already centered)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = (S**2) / max(Zc.shape[0]-1, 1)  # eigenvalues of the covariance along V
    cum = np.cumsum(var) / (var.sum() + 1e-12)

    if k_keep is None:
        if energy is None:
            # keep the full rank
            k = len(S)
        else:
            k = int(np.searchsorted(cum, float(energy)) + 1)
    else:
        k = int(k_keep)
    k = max(1, min(k, len(S)))  # safe

    V_k = Vt[:k, :].T         # (d, k)
    inv_sqrt = 1.0 / np.sqrt(var[:k] + eps)  # (k,)
    P = V_k * inv_sqrt        # (d, k)  -- whitening + reduce to k dims

    Xw = Xc @ P
    Yw = Yc @ P
    stats = {"type": "whiten-pca", "proj": P, "k": k, "energy": energy, "pre": center}
    return Xw, Yw, stats

def normalize_pair(X, Y, mode="zscore"):
    """Normalize two arrays X and Y according to the selected mode.

    mode in {"none", "zscore", "robust", "whiten"}.
      - "zscore": pooled z-score per feature.
      - "robust": pooled median/MAD (robust to outliers).
      - "whiten": PCA-whitening after pooled z-score (reduces feature correlations).
    """
    if mode is None or mode == "none":
        return np.asarray(X, float), np.asarray(Y, float), {"type": "none"}
    if mode == "zscore":
        return _zscore_pooled(X, Y)
    if mode == "robust":
        return _robust_zscore_pooled(X, Y)
    if mode == "whiten":
        return _pca_whiten_pooled(X, Y, center="zscore")
    raise ValueError("normalize_pair: invalid mode.")
