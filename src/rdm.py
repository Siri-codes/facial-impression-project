import numpy as np
from scipy import stats

def build_rdm(rating_matrix):
    """
    Takes a (1004, 34) rating matrix and returns a 34x34 RDM, where dissimilarity is defined as {1 - correlation} between each pair of attribute columns.
    """

    rm_transposed = np.transpose(rating_matrix)
    correlations = np.corrcoef(rm_transposed)
    dissimilarities = np.subtract(1, correlations)
    
    return dissimilarities

def get_unique_correlations(rdm):
    """
    Extracts the upper triangle (excluding the diagonal) from an RDM and returns it as a flat vector, so it can later be used to get the final human-vs-model RSA score.
    """

    rows, cols = rdm.shape

    indices = np.triu_indices(n=rows, k=1, m=cols) # k=1 excludes the diagonal
    flattened = rdm[indices] # extracts the upper triangle as a flat vector

    return flattened

def compare_rdms(rdm_1, rdm_2):
    """
    Returns the correlation coefficient for the two RDMS.
    """

    rdm_1_vector = get_unique_correlations(rdm_1)
    rdm_2_vector = get_unique_correlations(rdm_2)

    spearman_corr, p_val = stats.spearmanr(rdm_1_vector, rdm_2_vector)
    
    return spearman_corr, p_val