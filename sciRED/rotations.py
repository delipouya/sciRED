
import numpy as np
import numpy as np
from scipy.linalg import svd
from numpy.linalg import LinAlgError

####  python implementation of base R varimax function
def varimax(x, normalize=True, eps=1e-05):
    '''
    varimax rotation
    x: the factor loading matrix
    normalize: whether to normalize the factor loading matrix
    eps: the tolerance for convergence
    '''
    nc = x.shape[1]
    if nc < 2:
        return x
    
    if normalize:
        sc = np.sqrt(np.sum(x**2, axis=1))
        x = x / sc[:, np.newaxis]
    
    p = x.shape[0]
    TT = np.eye(nc)
    d = 0
    for i in range(1, 1001):
        z = np.dot(x, TT)
        B = np.dot(x.T, z**3 - np.dot(z, np.diag(np.sum(z**2, axis=0)) / p))
        u, sB, vh = svd(B, full_matrices=False)
        TT = np.dot(u, vh)
        dpast = d
        d = np.sum(sB)
        
        if d < dpast * (1 + eps):
            break
    
    z = np.dot(x, TT)
    
    if normalize:
        z = z * sc[:, np.newaxis]
    
    return {'rotloading': z, 'rotmat': TT}


####  python implementation of base R promax function
def promax(x, m=4, normalize=True, eps=1e-05):
    '''
    promax rotation
    x: the factor loading matrix
    m: the power of the objective function
    normalize: whether to normalize the factor loading matrix
    eps: the tolerance for convergence
    '''

    if x.shape[1] < 2:
        return x
    
    # Varimax rotation
    xx = varimax(x, normalize)
    x = xx['rotloading']
    
    # Calculate Q matrix
    q = x * np.abs(x)**(m - 1)
    
    try:
        # Perform linear regression
        u = np.linalg.lstsq(x, q, rcond=None)[0]
        d = np.diag(np.linalg.inv(np.dot(u.T, u)))
        u = np.dot(u, np.diag(np.sqrt(d)))
    except LinAlgError:
        print('Error: Singular matrix. The loadings cannot be rotated using promax.')
        return x  # Return the original loadings if there is an error
    
    # Calculate the rotated loadings
    z = np.dot(x, u)
    u = np.dot(xx['rotmat'], u)
    print('promax was performed.')
    
    return {'rotloading': z, 'rotmat': u}


## method: scale(original pc scores) %*% rotmat
def get_rotated_scores(factor_scores, rotmat):
    '''
    calculate the rotated factor scores
    factor_scores: the factor scores matrix
    rotmat: the rotation matrix
    '''
    return np.dot(factor_scores, rotmat)


