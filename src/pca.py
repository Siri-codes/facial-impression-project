import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def fit_pca(rating_matrix):
    """(n_stimuli, n_attributes) -> fitted PCA on standardized ratings."""
    return PCA().fit(StandardScaler().fit_transform(rating_matrix))

def match_components(pca_ref, pca_model, k=5):
    """For each of ref's first k PCs, find model's best-matching PC.
    Returns list of (ref_idx, model_idx, abs_r, ref_var, model_var).
    Sign-invariant: PCA component signs are arbitrary."""
    out = []
    for i in range(k):
        rs = [abs(np.corrcoef(pca_ref.components_[i], c)[0, 1])
              for c in pca_model.components_]
        j = int(np.argmax(rs))
        out.append((i, j, rs[j],
                    pca_ref.explained_variance_ratio_[i],
                    pca_model.explained_variance_ratio_[j]))
    return out