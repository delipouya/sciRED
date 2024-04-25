
import numpy as np

def get_factor_libsize_correlation(factor_scores, library_size):
    '''
    Calculate the correlation between the factor scores and the library size
    factor_scores: numpy array of the factor scores (n_cells, n_factors)
    library_size: numpy array of the library size (n_cells,)
    '''
    factor_libsize_correlation = np.zeros(factor_scores.shape[1])
    for i in range(factor_scores.shape[1]):
        factor_libsize_correlation[i] = np.corrcoef(factor_scores[:,i], library_size)[0,1]
    return factor_libsize_correlation
